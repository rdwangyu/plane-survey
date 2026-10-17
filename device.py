"""设备抽象层：蓝牙激光测距仪 + 模拟设备。

通讯协议
--------
测距仪每转一圈返回 ``points_per_turn`` 帧，每帧为 ``{点序号, 距离}``::

    device.connect()
    device.start_scan(points_per_turn)     # 下发启动指令，测距仪开始转
    for _ in range(points_per_turn):
        frame = device.read_frame()        # 阻塞读取一帧
        angle = frame_angle_deg(frame.index, points_per_turn)
    device.stop_scan()                     # 下发停止指令

:meth:`BaseDevice.query_info` 用于「关于」页：发送查询指令并解析测距仪信息。

角度约定：0° 指向 +x 轴，逆时针增大；距离单位为米。
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import queue
import random
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from PyQt6.QtCore import QThread, pyqtSignal

__all__ = [
    "DeviceError",
    "Frame",
    "frame_angle_deg",
    "BaseDevice",
    "MockDevice",
    "SimulatedDevice",
    "BleDevice",
    "BleScanner",
    "DEFAULT_POINTS_PER_TURN",
]

logger = logging.getLogger(__name__)

#: 「测距仪转一圈共返回多少个点」的默认值
DEFAULT_POINTS_PER_TURN = 720


class DeviceError(RuntimeError):
    """设备连接或通讯失败时抛出。"""


# --------------------------------------------------------------------------- #
# 数据帧
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Frame:
    """测距仪返回的一帧：``{点序号, 距离}``。"""

    index: int
    """点序号，由测距仪给出。"""

    distance: float
    """该点测得的距离（米）。"""


def frame_angle_deg(index: int, points_per_turn: int) -> float:
    """点序号 → 角度（度）：``点序号 / 一圈点数 * 360``。

    点序号按一圈取模，兼容测距仪从 0 或 1 开始计数、以及跨圈累加的实现。
    """
    pts = max(1, int(points_per_turn))
    return (int(index) % pts) / pts * 360.0


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #


class BaseDevice(abc.ABC):
    """激光测距仪的统一接口。

    一次完整采集为 ``connect`` → ``start_scan`` → 逐帧 ``read_frame`` →
    ``stop_scan`` → ``disconnect``。连接是长期持有的：同一个设备对象可以连续
    发起多次采集，因此 :meth:`stop_scan` 只停转，不断开。
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._points_per_turn: int = DEFAULT_POINTS_PER_TURN

    # -- 状态 -------------------------------------------------------------- #

    @property
    def connected(self) -> bool:
        """设备是否已连接。"""
        return self._connected

    @property
    def points_per_turn(self) -> int:
        """本次（或最近一次）采集设定的一圈点数。"""
        return self._points_per_turn

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """设备显示名称，用于状态栏提示。"""

    # -- 生命周期 ---------------------------------------------------------- #

    @abc.abstractmethod
    def connect(self) -> None:
        """建立连接。失败时抛出 :class:`DeviceError`。"""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并释放资源。重复调用应当安全。"""

    # -- 采集 -------------------------------------------------------------- #

    @abc.abstractmethod
    def start_scan(self, points_per_turn: int) -> None:
        """下发启动指令，让测距仪按 ``points_per_turn`` 个点转一圈。"""

    @abc.abstractmethod
    def read_frame(self) -> Frame:
        """阻塞读取下一帧。超时或失败时抛出 :class:`DeviceError`。"""

    def stop_scan(self) -> None:
        """下发停止指令。默认什么都不做。"""
        return None

    # -- 设备信息 ---------------------------------------------------------- #

    def query_info(self) -> dict[str, str]:
        """查询测距仪信息（关于页展示）。默认返回空字典。"""
        return {}

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

    内部维护一个闭合的房间多边形（矩形 + 一个壁龛），从原点按角度做射线求交，
    得到该方向上的距离。数据叠加少量高斯噪声与偶发跳变，便于演示平滑滤波与
    分段直线拟合的效果。
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
        room: Sequence[tuple[float, float]] | None = None,
        noise_std: float = 0.012,
        outlier_rate: float = 0.02,
        frame_interval: float = 0.004,
        seed: int | None = None,
    ) -> None:
        """
        :param room: 自定义房间多边形，默认使用 :attr:`DEFAULT_ROOM`。
        :param noise_std: 高斯噪声标准差（米）。
        :param outlier_rate: 单点跳变出现的概率，用于检验中值滤波。
        :param frame_interval: 每帧之间的模拟耗时（秒）。
        :param seed: 随机数种子，便于复现。
        """
        super().__init__()
        self._room: tuple[tuple[float, float], ...] = tuple(
            room if room is not None else self.DEFAULT_ROOM
        )
        if len(self._room) < 3:
            raise DeviceError("模拟房间至少需要 3 个顶点")

        self.noise_std = float(noise_std)
        self.outlier_rate = float(outlier_rate)
        self.frame_interval = max(0.0, float(frame_interval))
        self._rng = random.Random(seed)
        self._index: int = 0

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

    def start_scan(self, points_per_turn: int) -> None:
        self._require_connected()
        self._points_per_turn = max(1, int(points_per_turn))
        self._index = 0

    def read_frame(self) -> Frame:
        """产生下一帧：按点序号折算角度，再对房间轮廓做射线求交。"""
        self._require_connected()

        if self.frame_interval:
            time.sleep(self.frame_interval)

        index = self._index
        self._index += 1
        angle = frame_angle_deg(index, self._points_per_turn)

        value = self._ray_cast(angle) + self._rng.gauss(0.0, self.noise_std)
        if self._rng.random() < self.outlier_rate:
            # 模拟反光 / 遮挡导致的跳变点
            value += self._rng.choice((-1.0, 1.0)) * self._rng.uniform(0.08, 0.25)
        return Frame(index=index, distance=max(0.05, value))

    # -- 内部工具 ---------------------------------------------------------- #

    def _require_connected(self) -> None:
        if not self._connected:
            raise DeviceError("设备未连接")

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
      未确定而失败——勾选「模拟运行」后点「绘制 → 开始」必定能跑完流程。

    测距值仍由 :class:`MockDevice` 的房间轮廓生成，因此绘制出来的图形与真实
    房间拓扑一致，适合演示与回归测试。
    """

    def __init__(self, *, target: str = "", **kwargs: Any) -> None:
        """
        :param target: 本应通信的目标设备描述，仅用于提示与日志。
        """
        super().__init__(**kwargs)
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

    def start_scan(self, points_per_turn: int) -> None:
        """记录本应下发的启动指令，然后进入模拟采集。"""
        # TODO: 协议确定后，这里可以改成「把指令交给 BleDevice 编码但不发送」，
        #       用来验证真实报文长度、字节序与超时设置。
        self.command_log.append(f"START points_per_turn={int(points_per_turn)}")
        logger.debug("模拟下发指令 #%d：START", len(self.command_log))
        super().start_scan(points_per_turn)

    def read_frame(self) -> Frame:
        frame = super().read_frame()
        logger.debug("模拟返回第 %d 帧：%.4f m", frame.index, frame.distance)
        return frame

    def stop_scan(self) -> None:
        self.command_log.append("STOP")
        logger.debug("模拟下发指令 #%d：STOP", len(self.command_log))
        super().stop_scan()

    def query_info(self) -> dict[str, str]:
        """模拟运行下不发送任何查询指令，直接返回一组固定的设备信息。"""
        self.command_log.append("INFO")
        return {
            "设备": "模拟测距仪",
            "型号": "SIM-2D",
            "固件版本": "0.0.0-sim",
            "一圈点数": str(self._points_per_turn),
            "说明": "模拟运行，数据由内置房间轮廓生成，未进行任何真实通信。",
        }

    @property
    def commands_sent(self) -> int:
        """已记录的指令条数。"""
        return len(self.command_log)


# --------------------------------------------------------------------------- #
# 蓝牙设备
# --------------------------------------------------------------------------- #


class BleDevice(BaseDevice):
    """基于 ``bleak`` 的蓝牙测距仪实现。

    .. warning::
       真实设备的 GATT 特征 UUID、启动 / 停止 / 查询指令的报文格式以及
       「点序号 + 距离」帧的解析方式尚未确定，相应位置均以 ``TODO`` 标注。
       在协议确定之前：

       * UUID 留空时会**自动发现**——从设备已发布的服务里挑第一个可写特征作为
         控制通道、第一个 notify/indicate 特征作为数据通道、第一个可读特征作为
         信息通道，因此拿到真机即可先行联调；
       * :meth:`start_scan` / :meth:`stop_scan` / :meth:`query_info` 的指令载荷
         与 :meth:`parse_frame` / :meth:`parse_info` 采用**文档化的默认格式**，
         协议确定后只需替换这几个位置。

    ``bleak`` 是异步 API，这里在后台线程中常驻一个事件循环，对外暴露同步方法，
    方便被采集线程直接调用。
    """

    # ------------------------------------------------------------------ #
    # TODO: 协议确定后填写以下 UUID（留 None 表示自动发现）
    # ------------------------------------------------------------------ #
    #: 主服务 UUID
    SERVICE_UUID: str | None = None
    #: 控制特征 UUID（写入启动 / 停止 / 查询指令）
    CONTROL_CHAR_UUID: str | None = None
    #: 数据特征 UUID（notify 推送 {点序号, 距离} 帧）
    FRAME_CHAR_UUID: str | None = None
    #: 信息特征 UUID（查询响应）
    INFO_CHAR_UUID: str | None = None

    # ------------------------------------------------------------------ #
    # TODO: 协议确定后替换为真实指令报文
    # ------------------------------------------------------------------ #
    #: 启动采集指令（后接 uint16 小端的一圈点数）
    START_CMD: bytes = b"START"
    #: 停止采集指令
    STOP_CMD: bytes = b"STOP"
    #: 查询设备信息指令
    INFO_CMD: bytes = b"INFO"

    #: 一次写入命令的超时时间（秒）
    WRITE_TIMEOUT: float = 5.0
    #: 等待一帧 / 一次查询响应的超时时间（秒）
    READ_TIMEOUT: float = 5.0
    #: 接收缓冲上限（帧）。暂停接收时通知仍会到达，超出后丢弃最旧的帧。
    MAX_BACKLOG_FRAMES: int = 8192

    def __init__(
        self,
        address: str,
        *,
        connect_timeout: float = 15.0,
    ) -> None:
        """
        :param address: 蓝牙设备地址，例如 ``AA:BB:CC:DD:EE:FF``。
        :param connect_timeout: 连接超时时间（秒）。
        """
        super().__init__()
        if not address or not address.strip():
            raise DeviceError("未配置蓝牙设备地址")

        self.address: str = address.strip()
        self.connect_timeout = float(connect_timeout)

        self._client: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # 实际使用的特征 UUID（协议未定时由自动发现填充）
        self._control_uuid: str | None = None
        self._frame_uuid: str | None = None
        self._info_uuid: str | None = None

        self._frames: queue.Queue[Frame] = queue.Queue()
        self._notify_active: bool = False
        self._info_reply: bytes | None = None
        self._info_event = threading.Event()

    # -- BaseDevice 接口 --------------------------------------------------- #

    @property
    def name(self) -> str:
        return f"蓝牙测距仪 {self.address}"

    def connect(self) -> None:
        """连接设备、解析特征并订阅数据通知。"""
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
        self._notify_active = False
        self._client = None
        self._shutdown_loop()
        with self._lock:
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break

    def start_scan(self, points_per_turn: int) -> None:
        """下发启动指令，让测距仪按指定点数转一圈。"""
        self._require_connected()
        if self._control_uuid is None:
            raise DeviceError("未找到可写的控制特征，无法下发启动指令")

        self._points_per_turn = max(1, int(points_per_turn))
        with self._lock:
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break

        payload = self.START_CMD + struct.pack("<H", min(65535, self._points_per_turn))
        self._run(self._write_async(payload), timeout=self.WRITE_TIMEOUT)
        logger.info("已向 %s 下发启动指令（一圈 %d 点）", self.address, self._points_per_turn)

    def stop_scan(self) -> None:
        """下发停止指令；失败只记录日志，不打断界面收尾。"""
        if not self._connected or self._control_uuid is None:
            return
        try:
            self._run(self._write_async(self.STOP_CMD), timeout=self.WRITE_TIMEOUT)
        except DeviceError as exc:
            logger.warning("下发停止指令失败：%s", exc)

    def read_frame(self) -> Frame:
        """阻塞读取下一帧 ``{点序号, 距离}``。

        设备推送通知时从缓冲区取；设备只支持读取时退回主动读特征值。
        """
        self._require_connected()

        if self._notify_active:
            try:
                return self._frames.get(timeout=self.READ_TIMEOUT)
            except queue.Empty:
                raise DeviceError(
                    f"等待测距帧超时（{self.READ_TIMEOUT:.0f} 秒未收到数据）"
                ) from None

        if self._frame_uuid is None:
            raise DeviceError("未找到测距数据特征（notify 或 read）")
        raw = self._run(self._read_char_async(self._frame_uuid), timeout=self.READ_TIMEOUT)
        return self.parse_frame(bytes(raw))

    def query_info(self) -> dict[str, str]:
        """发送查询指令并解析测距仪信息（关于页使用）。"""
        self._require_connected()
        if self._control_uuid is None:
            raise DeviceError("未找到可写的控制特征，无法下发查询指令")

        self._info_reply = None
        self._info_event.clear()
        self._run(self._write_async(self.INFO_CMD), timeout=self.WRITE_TIMEOUT)
        logger.info("已向 %s 下发信息查询指令，等待响应…", self.address)

        if self._info_event.wait(timeout=self.READ_TIMEOUT):
            return self.parse_info(self._info_reply or b"")

        # 设备不推送响应时，退回直接读取信息特征
        if self._info_uuid is not None:
            try:
                raw = self._run(
                    self._read_char_async(self._info_uuid), timeout=self.READ_TIMEOUT
                )
                return self.parse_info(bytes(raw))
            except DeviceError as exc:
                logger.debug("直接读取设备信息失败：%s", exc)

        raise DeviceError(f"未收到测距仪信息响应（等待 {self.READ_TIMEOUT:.0f} 秒）")

    # -- 异步实现 ---------------------------------------------------------- #

    async def _connect_async(self) -> None:
        """建立 BLE 连接、解析特征并订阅数据通知。"""
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
        self._resolve_characteristics(client)
        logger.info(
            "已连接 %s（控制=%s 数据=%s 信息=%s）",
            self.address, self._control_uuid, self._frame_uuid, self._info_uuid,
        )

        if self._frame_uuid is not None:
            try:
                await client.start_notify(self._frame_uuid, self._on_frame_notify)
                self._notify_active = True
            except Exception as exc:  # noqa: BLE001
                # 该特征可能不支持通知，退回主动读取
                logger.warning("订阅测距数据通知失败，将改用主动读取：%s", exc)
                self._notify_active = False

        if self._info_uuid is not None and self._info_uuid != self._frame_uuid:
            try:
                await client.start_notify(self._info_uuid, self._on_info_notify)
            except Exception:  # noqa: BLE001 - 只读特征不支持通知，查询时退回读取
                logger.debug("信息特征不支持通知，查询时将直接读取")

    async def _disconnect_async(self) -> None:
        client = self._client
        if client is None:
            return
        for uuid in {self._frame_uuid, self._info_uuid} - {None}:
            try:
                await client.stop_notify(uuid)
            except Exception:  # noqa: BLE001 - 忽略取消订阅失败
                pass
        await client.disconnect()

    async def _write_async(self, payload: bytes) -> None:
        """把指令写入控制特征。"""
        client = self._client
        if client is None:
            raise DeviceError("蓝牙设备未连接")
        # response=None：由 bleak 按特征属性自动选择带响应 / 不带响应写入
        await client.write_gatt_char(self._control_uuid, payload, response=None)

    async def _read_char_async(self, uuid: str) -> bytes:
        client = self._client
        if client is None:
            raise DeviceError("蓝牙设备未连接")
        raw = await client.read_gatt_char(uuid)
        return bytes(raw)

    # -- 特征解析 ---------------------------------------------------------- #

    def _resolve_characteristics(self, client: Any) -> None:
        """确定控制 / 数据 / 信息三个特征。

        类常量已配置时直接采用；留空时从设备已发布的服务里自动发现，这样在协议
        UUID 尚未确定的情况下也能先连上真机联调。
        """
        self._control_uuid = self.CONTROL_CHAR_UUID
        self._frame_uuid = self.FRAME_CHAR_UUID
        self._info_uuid = self.INFO_CHAR_UUID
        if self._control_uuid and self._frame_uuid:
            return

        try:
            services = list(client.services)
        except Exception as exc:  # noqa: BLE001 - 服务枚举失败时保留已配置的 UUID
            logger.warning("枚举蓝牙服务失败，无法自动发现特征：%s", exc)
            return

        for service in services:
            service_uuid = str(getattr(service, "uuid", "") or "")
            if self.SERVICE_UUID and service_uuid.lower() != self.SERVICE_UUID.lower():
                continue
            for char in getattr(service, "characteristics", []) or []:
                uuid = str(getattr(char, "uuid", "") or "")
                if not uuid:
                    continue
                properties = {str(p).lower() for p in getattr(char, "properties", []) or []}
                if not self._control_uuid and properties & {"write", "write-without-response"}:
                    self._control_uuid = uuid
                if not self._frame_uuid and properties & {"notify", "indicate"}:
                    self._frame_uuid = uuid
                if not self._info_uuid and "read" in properties:
                    self._info_uuid = uuid

    # -- 回调与解析 -------------------------------------------------------- #

    def _on_frame_notify(self, _sender: Any, data: bytearray) -> None:
        """数据通知回调：解析后入队，供 :meth:`read_frame` 取用。"""
        try:
            frame = self.parse_frame(bytes(data))
        except Exception:  # noqa: BLE001 - 单帧报文异常不应中断采集
            logger.debug("忽略无法解析的数据帧", exc_info=True)
            return

        with self._lock:
            if self._frames.qsize() >= self.MAX_BACKLOG_FRAMES:
                # 暂停接收期间通知仍会到达，只保留最新的一批
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            self._frames.put(frame)

    def _on_info_notify(self, _sender: Any, data: bytearray) -> None:
        """信息响应回调。"""
        self._info_reply = bytes(data)
        self._info_event.set()

    @staticmethod
    def parse_frame(raw: bytes) -> Frame:
        """把数据帧报文解析为 :class:`Frame`。

        TODO: 协议确定后按真实格式实现。默认布局为 6 字节小端::

            偏移 0..1  uint16  点序号
            偏移 2..5  uint32  距离（毫米）
        """
        if len(raw) < 6:
            raise DeviceError(f"数据帧长度不足：{len(raw)} 字节")
        index, millimeters = struct.unpack_from("<HI", raw, 0)
        return Frame(index=int(index), distance=int(millimeters) / 1000.0)

    @staticmethod
    def parse_info(raw: bytes) -> dict[str, str]:
        """把查询响应解析成「字段 → 取值」。

        TODO: 协议确定后按真实格式实现。默认按 UTF-8 文本解析，支持以换行或
        分号分隔的 ``键=值`` 形式；若整段没有 ``=``，则原样作为一条响应返回。
        """
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            raise DeviceError("测距仪信息响应为空")

        info: dict[str, str] = {}
        for chunk in re.split(r"[;\r\n]+", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            key, sep, value = chunk.partition("=")
            if sep:
                info[key.strip()] = value.strip()
        return info or {"原始响应": text}

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

    def __init__(self, timeout: float = 6.0, parent: Any = None) -> None:
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
