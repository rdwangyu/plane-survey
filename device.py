"""设备抽象层：把「旋转电机 + 激光测距仪」封装成统一接口。

本模块提供：

* :class:`BaseDevice`  —— 设备抽象基类，定义旋转 / 测距的最小接口。
* :class:`MockDevice`  —— 无硬件时的模拟设备，按房间轮廓生成距离数据。
* :class:`BleDevice`   —— 基于 bleak 的蓝牙实现（通讯协议待定，内含 TODO）。
* :class:`BleScanner`  —— 后台线程中扫描 BLE 设备，供设置对话框使用。
* :func:`create_device` —— 按配置挑选并构造设备（蓝牙不可用时自动回退到模拟）。

角度约定：0° 指向 +x 轴，逆时针增大；距离单位为米。
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import random
import threading
import time
from typing import Any, Iterable, Sequence

from PyQt6.QtCore import QThread, pyqtSignal

__all__ = [
    "DeviceError",
    "BaseDevice",
    "MockDevice",
    "SimulatedDevice",
    "BleDevice",
    "BleScanner",
    "create_device",
    "MODE_AUTO",
    "MODE_MOCK",
    "MODE_BLE",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 设备模式常量
# --------------------------------------------------------------------------- #

MODE_AUTO = "auto"  # 优先蓝牙，失败或未配置地址时回退到模拟设备
MODE_MOCK = "mock"  # 强制使用模拟设备
MODE_BLE = "ble"    # 强制使用蓝牙设备


class DeviceError(RuntimeError):
    """设备连接或通讯失败时抛出。"""


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #


class BaseDevice(abc.ABC):
    """旋转电机 + 激光测距仪的统一接口。

    一次典型测量流程为::

        device.connect()
        device.rotate_to(angle)
        distance = device.read_distance()
        device.disconnect()

    子类需要实现 :meth:`connect` / :meth:`disconnect` / :meth:`rotate_to` /
    :meth:`read_distance` 四个方法。
    """

    def __init__(self, *, speed_rpm: float = 12.0, sample_interval: float = 0.05) -> None:
        """
        :param speed_rpm: 电机转速（转/分钟），用于估算转到目标角度的耗时。
        :param sample_interval: 两次采样之间的时间间隔（秒）。
        """
        self.speed_rpm: float = float(speed_rpm)
        self.sample_interval: float = float(sample_interval)
        self._connected: bool = False

    # -- 状态 -------------------------------------------------------------- #

    @property
    def connected(self) -> bool:
        """设备是否已连接。"""
        return self._connected

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """设备显示名称，用于状态栏提示。"""

    @property
    def degrees_per_sample(self) -> float:
        """按当前转速与采样间隔估算的角度步进（度）。"""
        return max(0.1, self.speed_rpm * 360.0 / 60.0 * self.sample_interval)

    # -- 生命周期 ---------------------------------------------------------- #

    @abc.abstractmethod
    def connect(self) -> None:
        """建立连接。失败时抛出 :class:`DeviceError`。"""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并释放资源。重复调用应当安全。"""

    # -- 动作 -------------------------------------------------------------- #

    @abc.abstractmethod
    def rotate_to(self, angle_deg: float) -> None:
        """把电机旋转到指定角度（度）。"""

    @abc.abstractmethod
    def read_distance(self) -> float:
        """读取一次激光测距结果，返回米。"""

    def set_speed(self, speed_rpm: float) -> None:
        """设置电机转速（转/分钟）。"""
        self.speed_rpm = max(0.1, float(speed_rpm))

    # -- 上下文管理 -------------------------------------------------------- #

    def __enter__(self) -> "BaseDevice":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.disconnect()


# --------------------------------------------------------------------------- #
# 模拟设备
# --------------------------------------------------------------------------- #


class MockDevice(BaseDevice):
    """无硬件时使用的模拟设备。

    内部维护一个闭合的房间多边形（矩形 + 一个壁龛），从原点按当前角度做射线
    求交，得到该方向上的距离。数据叠加少量高斯噪声与偶发跳变，便于演示平滑
    滤波与分段直线拟合的效果。
    """

    #: 房间轮廓顶点（米），逆时针闭合多边形；原点 (0, 0) 位于房间内部。
    DEFAULT_ROOM: tuple[tuple[float, float], ...] = (
        (4.0, 3.0),
        (-4.0, 3.0),
        (-4.0, -3.0),
        (1.0, -3.0),
        (1.0, -1.5),
        (4.0, -1.5),
    )

    def __init__(
        self,
        *,
        speed_rpm: float = 12.0,
        sample_interval: float = 0.05,
        room: Sequence[tuple[float, float]] | None = None,
        noise_std: float = 0.012,
        outlier_rate: float = 0.02,
        seed: int | None = None,
    ) -> None:
        """
        :param room: 自定义房间多边形，默认使用 :attr:`DEFAULT_ROOM`。
        :param noise_std: 高斯噪声标准差（米）。
        :param outlier_rate: 单点跳变出现的概率，用于检验中值滤波。
        :param seed: 随机数种子，便于复现。
        """
        super().__init__(speed_rpm=speed_rpm, sample_interval=sample_interval)
        self._room: tuple[tuple[float, float], ...] = tuple(
            room if room is not None else self.DEFAULT_ROOM
        )
        if len(self._room) < 3:
            raise DeviceError("模拟房间至少需要 3 个顶点")

        self.noise_std = float(noise_std)
        self.outlier_rate = float(outlier_rate)
        self._rng = random.Random(seed)
        self._angle: float = 0.0
        self._distance: float = self._ray_cast(0.0)

    # -- BaseDevice 接口 --------------------------------------------------- #

    @property
    def name(self) -> str:
        return "模拟设备（Mock）"

    def connect(self) -> None:
        """模拟设备总是可以连接，仅做少量延时以贴近真实手感。"""
        time.sleep(0.05)
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def rotate_to(self, angle_deg: float) -> None:
        """记录目标角度，并按转速模拟转动耗时。"""
        if not self._connected:
            raise DeviceError("设备未连接，无法旋转")

        target = float(angle_deg)
        delta = abs(target - self._angle)
        # 取一圈内的最短路径耗时，避免跨 360° 时等待过久
        travel = min(delta, 360.0 - (delta % 360.0)) if delta % 360.0 else delta
        angular_speed = max(1e-3, self.speed_rpm * 6.0)  # 度/秒
        time.sleep(min(travel / angular_speed, 0.5))

        self._angle = target
        self._distance = self._ray_cast(target)

    def read_distance(self) -> float:
        """返回当前角度上的距离，附带噪声与偶发跳变。"""
        if not self._connected:
            raise DeviceError("设备未连接，无法测距")

        time.sleep(min(self.sample_interval, 0.2))

        value = self._distance + self._rng.gauss(0.0, self.noise_std)
        if self._rng.random() < self.outlier_rate:
            # 模拟反光 / 遮挡导致的跳变点
            value += self._rng.choice((-1.0, 1.0)) * self._rng.uniform(0.08, 0.25)
        return max(0.05, value)

    # -- 内部工具 ---------------------------------------------------------- #

    def _ray_cast(self, angle_deg: float) -> float:
        """从原点沿给定角度发射射线，返回与房间轮廓的最近交点距离。"""
        theta = math.radians(angle_deg)
        dx, dy = math.cos(theta), math.sin(theta)
        best: float | None = None

        n = len(self._room)
        for i in range(n):
            x1, y1 = self._room[i]
            x2, y2 = self._room[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1

            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:  # 射线与边平行
                continue

            # 解 origin + t * dir = p1 + s * edge
            t = (x1 * ey - y1 * ex) / denom
            s = (x1 * dy - y1 * dx) / denom
            if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
                best = t if best is None else min(best, t)

        if best is None:  # 理论上不会发生，兜底避免程序崩溃
            return 1.0
        return best


class SimulatedDevice(MockDevice):
    """「模拟运行」设备：完全不进行真实通信，只返回模拟数据。

    与 :class:`MockDevice` 的差别在于**语义**和**可观测性**，而不只是换个名字：

    * 它代表「本来要跟某台真实设备通信」的那种运行方式，因此会把**本应下发的
      指令**逐条记录下来并写入日志，方便在没有硬件时核对指令流与时序；
    * 它绝不会打开蓝牙连接，因此不会因为适配器缺失、设备不在范围内、协议
      未确定而失败——勾选「模拟运行」后点「绘制」必定能跑完流程。

    测距值仍由 :class:`MockDevice` 的房间轮廓生成，因此绘制出来的图形与真实
    房间拓扑一致，适合演示与回归测试。
    """

    def __init__(
        self,
        *,
        target: str = "",
        speed_rpm: float = 12.0,
        sample_interval: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """
        :param target: 本应通信的目标设备描述，仅用于提示与日志。
        """
        super().__init__(speed_rpm=speed_rpm, sample_interval=sample_interval, seed=seed)
        self.target: str = target.strip() or "（未指定设备）"
        #: 本次运行中「本应下发」的指令，按顺序记录
        self.command_log: list[str] = []

    # -- BaseDevice 接口 --------------------------------------------------- #

    @property
    def name(self) -> str:
        return f"模拟运行（不通信）→ {self.target}"

    def connect(self) -> None:
        """不建立任何真实连接，仅重置指令记录。"""
        self.command_log.clear()
        logger.info("模拟运行：跳过真实连接，目标设备 %s", self.target)
        super().connect()

    def rotate_to(self, angle_deg: float) -> None:
        """记录本应下发的电机指令，然后按模拟角度继续。"""
        # TODO: 协议确定后，这里可以改成「把指令交给 BleDevice 编码但不发送」，
        #       用来验证真实报文长度、字节序与超时设置。
        self.command_log.append(f"MOTOR -> {angle_deg:.2f} deg")
        logger.debug("模拟下发指令 #%d：MOTOR -> %.2f deg", len(self.command_log), angle_deg)
        super().rotate_to(angle_deg)

    def read_distance(self) -> float:
        """返回模拟测距值。"""
        value = super().read_distance()
        logger.debug("模拟返回测距值：%.4f m", value)
        return value

    @property
    def commands_sent(self) -> int:
        """已记录的指令条数。"""
        return len(self.command_log)


# --------------------------------------------------------------------------- #
# 蓝牙设备
# --------------------------------------------------------------------------- #


class BleDevice(BaseDevice):
    """基于 ``bleak`` 的蓝牙设备实现。

    .. warning::
       真实设备的 GATT 服务 / 特征 UUID、写入命令格式与测距报文解析方式尚未
       确定，相关位置均以 ``TODO`` 标注。在协议补齐之前，:meth:`connect` 会
       抛出 :class:`DeviceError`，由 :func:`create_device` 回退到
       :class:`MockDevice`，因此界面仍可正常演示。

    ``bleak`` 是异步 API，这里在后台线程中常驻一个事件循环，对外暴露同步方法，
    方便被扫描线程直接调用。
    """

    # ------------------------------------------------------------------ #
    # TODO: 协议确定后填写以下常量
    # ------------------------------------------------------------------ #
    #: 主服务 UUID
    SERVICE_UUID: str | None = None
    #: 电机控制特征 UUID（写入）
    MOTOR_CHAR_UUID: str | None = None
    #: 测距数据特征 UUID（通知 / 读取）
    RANGE_CHAR_UUID: str | None = None
    #: 使能激光的特征 UUID（可选）
    LASER_CHAR_UUID: str | None = None
    #: 一次写入命令的超时时间（秒）
    WRITE_TIMEOUT: float = 5.0
    #: 单次读取测距值的超时时间（秒）
    READ_TIMEOUT: float = 5.0

    def __init__(
        self,
        address: str,
        *,
        speed_rpm: float = 12.0,
        sample_interval: float = 0.05,
        connect_timeout: float = 15.0,
    ) -> None:
        """
        :param address: 蓝牙设备地址，例如 ``AA:BB:CC:DD:EE:FF``。
        :param connect_timeout: 连接超时时间（秒）。
        """
        super().__init__(speed_rpm=speed_rpm, sample_interval=sample_interval)
        if not address or not address.strip():
            raise DeviceError("未配置蓝牙设备地址")

        self.address: str = address.strip()
        self.connect_timeout = float(connect_timeout)

        self._client: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._latest_distance: float | None = None  # 由通知回调写入
        self._lock = threading.Lock()

    # -- BaseDevice 接口 --------------------------------------------------- #

    @property
    def name(self) -> str:
        return f"蓝牙设备 {self.address}"

    def connect(self) -> None:
        """连接设备并订阅测距通知。"""
        self._ensure_loop()
        self._run(self._connect_async(), timeout=self.connect_timeout)
        self._connected = True

    def disconnect(self) -> None:
        """断开连接并关闭后台事件循环。"""
        if self._loop is not None and self._client is not None:
            try:
                self._run(self._disconnect_async(), timeout=5.0)
            except Exception:  # noqa: BLE001 - 断开失败不应阻塞退出
                pass
        self._connected = False
        self._client = None
        self._shutdown_loop()

    def rotate_to(self, angle_deg: float) -> None:
        """向电机写入目标角度。

        TODO: 协议确定后替换为真实指令，例如::

            payload = struct.pack("<f", math.radians(angle_deg))
            await client.write_gatt_char(self.MOTOR_CHAR_UUID, payload, response=True)
            await asyncio.sleep(该角度所需转动时间)  # 或等待电机到位通知
        """
        self._require_connected()
        if self.MOTOR_CHAR_UUID is None:
            raise DeviceError("蓝牙电机控制 UUID 未配置（协议待定）")

        self._run(self._rotate_async(angle_deg), timeout=self.WRITE_TIMEOUT)

    def read_distance(self) -> float:
        """读取一次测距结果（米）。

        TODO: 协议确定后替换为真实报文解析，例如::

            raw = await client.read_gatt_char(self.RANGE_CHAR_UUID)
            return int.from_bytes(raw[2:4], "little") / 1000.0
        """
        self._require_connected()
        if self.RANGE_CHAR_UUID is None:
            raise DeviceError("蓝牙测距 UUID 未配置（协议待定）")

        value = self._run(self._read_async(), timeout=self.READ_TIMEOUT)
        return float(value)

    # -- 异步实现 ---------------------------------------------------------- #

    async def _connect_async(self) -> None:
        """建立 BLE 连接并订阅通知。"""
        try:
            from bleak import BleakClient  # 延迟导入，缺少依赖时不影响模拟模式
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise DeviceError(f"未安装 bleak，无法使用蓝牙设备：{exc}") from exc

        try:
            client = BleakClient(self.address, timeout=self.connect_timeout)
            await client.connect()
        except Exception as exc:  # noqa: BLE001 - bleak 会抛出多种异常
            raise DeviceError(f"连接蓝牙设备失败：{exc}") from exc

        if not client.is_connected:
            raise DeviceError(f"连接蓝牙设备失败：{self.address} 未进入已连接状态")

        self._client = client

        # TODO: 协议确定后校验服务与特征是否存在，例如
        #   services = client.services
        #   if self.SERVICE_UUID not in [s.uuid for s in services]:
        #       raise DeviceError("目标服务不存在")
        if self.RANGE_CHAR_UUID is not None:
            try:
                await client.start_notify(self.RANGE_CHAR_UUID, self._on_notify)
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(f"订阅测距通知失败：{exc}") from exc

    async def _disconnect_async(self) -> None:
        client = self._client
        if client is None:
            return
        if self.RANGE_CHAR_UUID is not None:
            try:
                await client.stop_notify(self.RANGE_CHAR_UUID)
            except Exception:  # noqa: BLE001 - 忽略取消订阅失败
                pass
        await client.disconnect()

    async def _rotate_async(self, angle_deg: float) -> None:
        """TODO: 按真实协议写入电机指令。"""
        client = self._client
        if client is None:
            raise DeviceError("蓝牙设备未连接")

        # 协议待定：保留写入骨架，便于协议确定后直接补全。
        payload = str(angle_deg).encode("ascii")  # 占位载荷，勿用于真实设备
        await client.write_gatt_char(self.MOTOR_CHAR_UUID, payload, response=True)
        await asyncio.sleep(self.sample_interval)

    async def _read_async(self) -> float:
        """TODO: 解析真实测距报文。"""
        client = self._client
        if client is None:
            raise DeviceError("蓝牙设备未连接")

        with self._lock:
            if self._latest_distance is not None:
                return self._latest_distance

        raw = await client.read_gatt_char(self.RANGE_CHAR_UUID)
        return self.parse_distance(bytes(raw))

    # -- 回调与解析 -------------------------------------------------------- #

    def _on_notify(self, _sender: Any, data: bytearray) -> None:
        """测距通知回调：解析后缓存最新值。"""
        try:
            value = self.parse_distance(bytes(data))
        except Exception:  # noqa: BLE001 - 单条报文异常不应中断扫描
            return
        with self._lock:
            self._latest_distance = value

    @staticmethod
    def parse_distance(raw: bytes) -> float:
        """把原始报文解析为距离（米）。

        TODO: 协议确定后按真实格式实现，例如::

            return int.from_bytes(raw[2:4], "little") / 1000.0
        """
        raise DeviceError("测距报文解析尚未实现（协议待定）")

    # -- 事件循环管理 ------------------------------------------------------ #

    def _ensure_loop(self) -> None:
        """在后台线程中启动常驻事件循环。"""
        if self._loop is not None and self._loop.is_running():
            return

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_runner, name="ble-event-loop", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread

    def _shutdown_loop(self) -> None:
        loop, thread = self._loop, self._loop_thread
        self._loop, self._loop_thread = None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=2.0)
        if not loop.is_running():
            loop.close()

    def _run(self, coro: Any, *, timeout: float) -> Any:
        """在后台事件循环中同步执行协程，并返回结果。"""
        loop = self._loop
        if loop is None or not loop.is_running():
            raise DeviceError("蓝牙事件循环未启动")

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except DeviceError:
            raise
        except TimeoutError as exc:
            future.cancel()
            raise DeviceError(f"蓝牙操作超时（{timeout:.0f} 秒）") from exc
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"蓝牙操作失败：{exc}") from exc

    def _require_connected(self) -> None:
        if not self._connected:
            raise DeviceError("设备未连接")


# --------------------------------------------------------------------------- #
# BLE 扫描线程
# --------------------------------------------------------------------------- #


class BleScanner(QThread):
    """在后台线程扫描 BLE 设备，避免阻塞界面。

    信号：
        * ``device_found(address, name)`` —— 每发现一个设备发一次。
        * ``scan_finished(count)``        —— 扫描正常结束。
        * ``scan_failed(message)``        —— 扫描失败（缺少依赖、无适配器等）。
    """

    device_found = pyqtSignal(str, str)
    scan_finished = pyqtSignal(int)
    scan_failed = pyqtSignal(str)

    def __init__(self, timeout: float = 5.0, parent: Any = None) -> None:
        super().__init__(parent)
        self.timeout = float(timeout)
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """请求尽快结束扫描。"""
        self._cancelled.set()

    def run(self) -> None:  # noqa: D102 - QThread 入口
        if self._cancelled.is_set():
            self.scan_finished.emit(0)
            return

        try:
            from bleak import BleakScanner
        except ImportError as exc:
            self.scan_failed.emit(f"未安装 bleak，无法扫描蓝牙设备：{exc}")
            return

        async def _discover() -> Iterable[Any]:
            return await BleakScanner.discover(timeout=self.timeout)

        try:
            devices = asyncio.run(_discover())
        except Exception as exc:  # noqa: BLE001 - 蓝牙栈可能抛出多种异常
            self.scan_failed.emit(f"蓝牙扫描失败：{exc}")
            return

        count = 0
        for item in devices:
            if self._cancelled.is_set():
                break
            address = getattr(item, "address", "") or ""
            name = getattr(item, "name", None) or "(未知设备)"
            self.device_found.emit(address, name)
            count += 1

        self.scan_finished.emit(count)


# --------------------------------------------------------------------------- #
# 工厂函数
# --------------------------------------------------------------------------- #


def create_device(
    mode: str = MODE_AUTO,
    *,
    address: str = "",
    speed_rpm: float = 12.0,
    sample_interval: float = 0.05,
    simulate: bool = False,
) -> BaseDevice:
    """按配置构造设备实例。

    ``auto`` 模式下优先尝试蓝牙；地址为空或蓝牙初始化失败时回退到
    :class:`MockDevice`，保证无硬件时界面依然可用。

    :param mode: ``auto`` / ``mock`` / ``ble`` 之一。
    :param address: 蓝牙设备地址，``mock`` 模式下忽略。
    :param simulate: 勾选「模拟运行」时为 ``True``：直接返回
        :class:`SimulatedDevice`，**完全不做真实通信**，``mode`` 与 ``address``
        仅用于在提示里说明「本来会连哪台设备」。
    :returns: 设备实例（尚未连接）。
    """
    if simulate:
        target = address.strip() or f"按「{mode}」模式自动选择的设备"
        return SimulatedDevice(
            target=target, speed_rpm=speed_rpm, sample_interval=sample_interval
        )

    if mode == MODE_MOCK:
        return MockDevice(speed_rpm=speed_rpm, sample_interval=sample_interval)

    if mode == MODE_BLE:
        # 显式选择蓝牙时不再回退，让错误直接暴露给用户
        return BleDevice(address, speed_rpm=speed_rpm, sample_interval=sample_interval)

    # auto
    if address.strip():
        try:
            return BleDevice(address, speed_rpm=speed_rpm, sample_interval=sample_interval)
        except DeviceError:
            pass
    return MockDevice(speed_rpm=speed_rpm, sample_interval=sample_interval)
