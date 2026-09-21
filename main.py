"""程序入口与主窗口。

界面结构::

    ┌──────────────────────────────────────────────┐
    │ [绘制] [导出] [设置]      ← QToolBar（左上角） │
    ├──────────────────────────────────────────────┤
    │                                              │
    │         GLViewWidget 3D 扫描视图              │
    │                                              │
    ├──────────────────────────────────────────────┤
    │ 状态提示            [进度条]      交互提示     │
    └──────────────────────────────────────────────┘

扫描在 :class:`ScanWorker` 线程中执行，通过信号实时把每个采样点回传主线程绘制，
因此界面不会卡死。平滑算法（中值滤波 + 滑动均值 + 分段最小二乘直线拟合）也放在
工作线程内完成，避免大点集阻塞界面。
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStyle,
    QToolBar,
)

from device import MODE_AUTO, MODE_MOCK, BaseDevice, DeviceError, MockDevice, create_device
from exporter import FORMAT_PENDING_MESSAGE, export_scan, suggested_filename
from plot3d import ScanView
from settings_dialog import ScanConfig, SettingsDialog, load_config

__all__ = ["MainWindow", "ScanWorker", "main"]

logger = logging.getLogger(__name__)

#: 实时预览的刷新间隔（毫秒），避免每个采样点都重建图元
PREVIEW_INTERVAL_MS = 40

#: 分段直线拟合允许的最短段点数
MIN_SEGMENT = 5


# --------------------------------------------------------------------------- #
# 坐标换算
# --------------------------------------------------------------------------- #


def polar_to_points(
    angles_deg: Sequence[float] | np.ndarray,
    distances: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """把极坐标采样转换为 3D 坐标。

    ``x = d * cos(θ)``，``y = d * sin(θ)``，``z = 0``。

    :param angles_deg: 角度序列（度）。
    :param distances: 距离序列（米）。
    :returns: 形状 ``(N, 3)`` 的坐标数组。
    """
    angle_arr = np.asarray(angles_deg, dtype=np.float64).ravel()
    dist_arr = np.asarray(distances, dtype=np.float64).ravel()
    if angle_arr.size != dist_arr.size:
        raise ValueError(f"角度与距离长度不一致：{angle_arr.size} != {dist_arr.size}")
    if angle_arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    rad = np.radians(angle_arr)
    x = dist_arr * np.cos(rad)
    y = dist_arr * np.sin(rad)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))


# --------------------------------------------------------------------------- #
# 平滑算法
# --------------------------------------------------------------------------- #


def _window_filter(
    values: np.ndarray,
    window: int,
    *,
    reducer: str,
    causal: bool = False,
) -> np.ndarray:
    """滑动窗口滤波的统一实现。

    :param values: 一维序列。
    :param window: 窗口长度（自动修正为不小于 3 的奇数）。
    :param reducer: ``"median"`` 或 ``"mean"``。
    :param causal: 为 ``True`` 时只使用当前点及之前的样本（用于实时预览）。
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    size = arr.size
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    if size < 3 or win <= 1:
        return arr.copy()

    half = win // 2
    out = np.empty_like(arr)
    for i in range(size):
        lo = max(0, i - half)
        hi = i + 1 if causal else min(size, i + half + 1)
        chunk = arr[lo:hi]
        if chunk.size == 0:
            out[i] = arr[i]
        elif reducer == "median":
            out[i] = float(np.median(chunk))
        else:
            out[i] = float(np.mean(chunk))
    return out


def median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    """中值滤波：有效抑制测距跳变等孤立异常点。"""
    return _window_filter(values, window, reducer="median")


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """一维滑动均值滤波：平滑高频噪声。

    .. warning::
       这是**距离序列**上的一维均值滤波，:func:`smooth_scan` 刻意没有用它。
       墙面在 xy 平面上是直线，其距离随角度却是 ``d(θ) = r / cos(θ - φ)``
       这样的曲线，对距离取平均等于在曲线上取弦，会把点系统性地拉离墙面
       （角度步进越粗越严重）。扫描流程请改用 :func:`_mean_filter_xy`。
       本函数仅适用于本身近似线性的序列，例如实时预览。
    """
    return _window_filter(values, window, reducer="mean")


def causal_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """因果平滑（只用历史样本），用于扫描过程中的实时预览。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 3:
        return arr.copy()
    step1 = _window_filter(arr, window, reducer="median", causal=True)
    return _window_filter(step1, window, reducer="mean", causal=True)


def _principal_direction(xy: np.ndarray) -> np.ndarray | None:
    """用 PCA 求点集的主方向（单位向量）。点数不足或退化时返回 ``None``。"""
    if xy.shape[0] < 2:
        return None
    centered = xy - xy.mean(axis=0)
    if not np.isfinite(centered).all():
        return None
    cov = centered.T @ centered / centered.shape[0]
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    return direction / norm


class _RunningLine:
    """增量式直线拟合。

    通过累加一阶/二阶统计量，用 2×2 协方差矩阵主特征向量的闭式解求方向，
    使得「加入一个点」与「计算某点到当前直线的垂距」都是 O(1)，从而能够对
    上万点的扫描序列做自底向上的直线段生长。
    """

    __slots__ = ("_n", "_sx", "_sy", "_sxx", "_syy", "_sxy")

    def __init__(self) -> None:
        self._n = 0
        self._sx = self._sy = 0.0
        self._sxx = self._syy = self._sxy = 0.0

    def add(self, x: float, y: float) -> None:
        """加入一个点。"""
        self._n += 1
        self._sx += x
        self._sy += y
        self._sxx += x * x
        self._syy += y * y
        self._sxy += x * y

    def __len__(self) -> int:
        return self._n

    @property
    def centroid(self) -> np.ndarray:
        """当前点集的重心。"""
        if self._n == 0:
            return np.zeros(2, dtype=np.float64)
        return np.array([self._sx / self._n, self._sy / self._n], dtype=np.float64)

    def direction(self) -> np.ndarray | None:
        """当前点集的主方向（单位向量）；点数不足时返回 ``None``。"""
        if self._n < 2:
            return None
        count = float(self._n)
        mx, my = self._sx / count, self._sy / count
        a = self._sxx / count - mx * mx
        c = self._syy / count - my * my
        b = self._sxy / count - mx * my
        # 2×2 对称矩阵 [[a, b], [b, c]] 的主特征值
        eigenvalue = 0.5 * (a + c + math.hypot(a - c, 2.0 * b))
        if abs(b) > 1e-12:
            vector = np.array([eigenvalue - c, b], dtype=np.float64)
        elif a >= c:
            vector = np.array([1.0, 0.0], dtype=np.float64)
        else:
            vector = np.array([0.0, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else None

    def residual(self, x: float, y: float) -> float:
        """点 ``(x, y)`` 到当前拟合直线的垂直距离。"""
        direction = self.direction()
        if direction is None:
            return 0.0
        center = self.centroid
        return abs(direction[0] * (y - center[1]) - direction[1] * (x - center[0]))


def _estimate_residual_tolerance(xy: np.ndarray) -> float:
    """由相邻三点的垂距中位数（MAD）稳健估计噪声，得到分段阈值。

    拐角只占少数样本，因此中位数不受其影响；阈值取噪声的若干倍，既能容纳
    正常抖动，又能在方向发生真实变化处断开。
    """
    total = xy.shape[0]
    if total < 5:
        return 0.05
    prev, cur, nxt = xy[:-2], xy[1:-1], xy[2:]
    chord = nxt - prev
    length = np.linalg.norm(chord, axis=1)
    length[length < 1e-12] = 1e-12
    dev = np.abs(chord[:, 0] * (cur[:, 1] - prev[:, 1]) - chord[:, 1] * (cur[:, 0] - prev[:, 0]))
    dev /= length
    median = float(np.median(dev))
    mad = float(np.median(np.abs(dev - median)))
    sigma = 1.4826 * mad
    return float(np.clip(2.5 * sigma, 0.015, 0.30))


def _grow_segments(xy: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """自底向上生长直线段：逐点计算垂距，超阈值即断开。

    对**孤立跳变**做了容错：若某个点偏离当前直线，但紧接着的下一个点又回到
    直线附近，就认为该点只是测距毛刺——跳过它（不参与拟合）继续生长。这样
    毛刺被留在段内，随后由段内中值滤波与直线投影共同修正，而不是把一面墙
    硬生生切成两段。
    """
    total = xy.shape[0]
    segments: list[tuple[int, int]] = []
    start = 0

    while start < total - 1:
        line = _RunningLine()
        line.add(float(xy[start, 0]), float(xy[start, 1]))
        line.add(float(xy[start + 1, 0]), float(xy[start + 1, 1]))
        end = start + 2

        while end < total:
            if line.residual(float(xy[end, 0]), float(xy[end, 1])) <= tolerance:
                line.add(float(xy[end, 0]), float(xy[end, 1]))
                end += 1
                continue
            # 孤立跳变：下一点回到直线附近则跳过该点，继续生长
            if end + 1 < total and line.residual(float(xy[end + 1, 0]), float(xy[end + 1, 1])) <= tolerance:
                end += 1
                continue
            break

        segments.append((start, end))
        start = end

    return segments


def _merge_segments(
    xy: np.ndarray,
    segments: list[tuple[int, int]],
    angle_threshold_deg: float,
) -> list[tuple[int, int]]:
    """合并方向几乎一致的相邻段，修复生长阶段被噪声切碎的过分割。

    只按方向夹角合并，**不**因为某段点数少就强行并入邻居——真实的短墙
    （例如壁龛）角度与邻居相差很大，强行合并会让一条直线横跨两面墙，
    误差可达米级。

    例外是**单点段**：一个点无法确定方向，几乎总是测距跳变造成的孤立点，
    把它并回相邻段，后续段内中值滤波就会将其剔除。
    """
    merged = list(segments)

    # 1) 吸收单点段
    absorbed = True
    while absorbed and len(merged) > 1:
        absorbed = False
        for index, (start, end) in enumerate(merged):
            if end - start >= 2:
                continue
            if index > 0:
                merged[index - 1] = (merged[index - 1][0], end)
                del merged[index]
            else:
                merged[1] = (start, merged[1][1])
                del merged[0]
            absorbed = True
            break

    # 2) 按方向合并
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index in range(len(merged) - 1):
            start_a, end_a = merged[index]
            start_b, end_b = merged[index + 1]
            direction_a = _principal_direction(xy[start_a:end_a])
            direction_b = _principal_direction(xy[start_b:end_b])
            if direction_a is None or direction_b is None:
                continue
            cos_value = float(np.clip(abs(float(np.dot(direction_a, direction_b))), 0.0, 1.0))
            if math.degrees(math.acos(cos_value)) <= angle_threshold_deg:
                merged[index] = (start_a, end_b)
                del merged[index + 1]
                changed = True
                break
    return merged


def _segment_runs(
    xy: np.ndarray,
    residual_tolerance: float | None,
    angle_threshold_deg: float,
) -> list[tuple[int, int]]:
    """把点序列切成若干直线段，返回 ``[start, end)`` 区间列表。"""
    tolerance = (
        float(residual_tolerance)
        if residual_tolerance is not None
        else _estimate_residual_tolerance(xy)
    )
    grown = _grow_segments(xy, tolerance)
    return _merge_segments(xy, grown, angle_threshold_deg)


def piecewise_line_fit(
    points: np.ndarray,
    *,
    angle_threshold_deg: float = 10.0,
    min_segment: int = 5,
    residual_tolerance: float | None = None,
) -> np.ndarray:
    """分段最小二乘直线拟合，把墙面线条拉直。

    步骤：

    1. **分段**：自底向上生长直线段——从相邻两点确定初始直线，逐点计算垂距，
       超过阈值就断开，因此段边界正好落在真实拐角处；
    2. **合并**：把方向几乎一致或过短的相邻段合并，修复过分割；
    3. **拟合**：对每段做总体最小二乘（PCA）拟合，段内点全部投影到拟合直线上。

    拐角处刻意**不做**交点吸附：把端点强行拉到相邻两段拟合直线的交点，会把
    真实的测量点挪走（角度分辨率越低、偏离越大），反而降低几何精度。投影后
    两段直线自然相接，拐角最多只有一个角度步进的圆角。

    :param points: 形状 ``(N, 3)`` 的点集。
    :param angle_threshold_deg: 合并相邻段时的方向夹角阈值（度）。
    :param min_segment: 点数少于该值时不拟合，直接原样返回。
    :param residual_tolerance: 分段阈值（米）；``None`` 时由数据自动估计。
    :returns: 拟合后的点集，形状与输入一致；点数不足时原样返回。
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2 * max(2, min_segment):
        return arr.copy()

    xy = arr[:, :2]
    segments = _segment_runs(xy, residual_tolerance, angle_threshold_deg)

    result = xy.copy()

    for start, end in segments:
        segment = xy[start:end]
        direction = _principal_direction(segment)
        if direction is None:
            continue
        origin = segment.mean(axis=0)
        # 投影到拟合直线上：段内点被完全拉直
        t = (segment - origin) @ direction
        result[start:end] = origin + np.outer(t, direction)

    out = arr.copy()
    out[:, :2] = result
    out[:, 2] = 0.0
    return out


def _filter_runs(
    values: np.ndarray,
    segments: Sequence[tuple[int, int]],
    median_window: int,
) -> np.ndarray:
    """在每个直线段内部做**中值**滤波，返回滤波后的距离序列。

    中值滤波留在距离空间是正确的：测距跳变是沿射线方向的径向尖刺，在距离
    序列里就是一个孤立的离群值，中值能直接把它剔除；取中值不产生新的数值，
    也就不会引入偏置。

    均值滤波**不在这里做**——见 :func:`_mean_filter_xy`。

    只替换段内部的点，段两端各留出半个窗口不处理：

    * 段边界就是拐角，跨段取窗口会把两面墙的距离混在一起；
    * 即使在段内，窗口在边界处被截断也会让端点的值「向段内平移」——对掠射
      墙面（距离随角度变化很快）这一平移可达十几厘米。端点交给后续的直线
      拟合处理，那里是无偏的。
    """
    out = np.asarray(values, dtype=np.float64).copy()
    margin = (max(3, int(median_window)) - 1) // 2

    for start, end in segments:
        size = end - start
        if size <= 2 * margin:
            continue  # 段太短，整段都不滤波，避免端点在段内占比过高
        run = out[start:end]
        smoothed = median_filter(run, median_window)
        out[start + margin : end - margin] = smoothed[margin : size - margin]
    return out


def _moving_average_rows(values: np.ndarray, window: int) -> np.ndarray:
    """对二维数组按行（``axis=0``）做滑动均值，窗口用边缘值补齐。"""
    arr = np.asarray(values, dtype=np.float64)
    win = max(1, int(window))
    if win <= 1 or arr.shape[0] == 0:
        return arr.copy()
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.cumsum(np.vstack([np.zeros((1, arr.shape[1])), padded]), axis=0)
    return (cumulative[win:] - cumulative[:-win]) / win


def _mean_filter_xy(
    xy: np.ndarray,
    segments: Sequence[tuple[int, int]],
    mean_window: int,
) -> np.ndarray:
    """在每个直线段内部对 **x、y 坐标**（而不是距离）做滑动均值滤波。

    为什么必须放在直角坐标里做：一面墙在 xy 平面上是直线，但它的距离随角度
    变化是 ``d(θ) = r / cos(θ - φ)``，并不线性——尤其掠射时曲率很大。对**距离**
    取滑动平均，等于在一条弯曲的序列上取弦，结果会系统性地偏向房间内侧，而且
    角度步进越粗偏得越多（实测 9° 步进时中位偏差 30 mm，1.8° 时 3.4 mm）。
    直线拟合只能在这份已经偏掉的输入上做最小二乘，修不回来。

    对坐标取滑动平均则没有这个问题：共线点的任何凸组合仍然落在那条直线上，
    所以均值滤波不会把点推离墙面。实测偏差降到 4~6 mm，且与步进粗细基本无关。

    与前一步一致，段两端各留半个窗口不处理。
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    margin = (max(3, int(mean_window)) - 1) // 2

    for start, end in segments:
        size = end - start
        if size <= 2 * margin:
            continue
        run = out[start:end]
        smoothed = _moving_average_rows(run, mean_window)
        out[start + margin : end - margin] = smoothed[margin : size - margin]
    return out


def smooth_scan(
    angles_deg: Sequence[float] | np.ndarray,
    distances: Sequence[float] | np.ndarray,
    *,
    median_window: int = 5,
    mean_window: int = 5,
    corner_threshold_deg: float = 10.0,
) -> np.ndarray:
    """完整平滑流程：分段 → 段内滤波 → 分段最小二乘直线拟合。

    顺序和空间都很关键：**先分段、再滤波；中值在距离空间，均值在 xy 空间**。

    1. 用原始点序列按垂距生长出直线段，边界即真实拐角。必须最先做：如果先对
       整条序列滤波，拐角两侧（属于两面不同的墙）的数据会被混在一起平均，
       反而在拐角处引入最大可达十几厘米的误差，且后续拟合无法修复。
    2. 在每个段内部对**距离**做中值滤波，剔除测距跳变。窗口不跨越拐角。
    3. 转成直角坐标，在每个段内部对 **x、y 坐标**做滑动均值滤波压制噪声。
       均值必须在这一步做：墙面在 xy 里是直线、在距离序列里是 ``r/cos(θ-φ)``
       这样的曲线，对距离取平均会系统性地把点拉向房间内侧（角度步进越粗越
       严重）。理由详见 :func:`_mean_filter_xy`。
    4. 重新分段并做总体最小二乘直线拟合，段内点投影到拟合直线上，墙面被拉直。

    :returns: 形状 ``(N, 3)`` 的平滑后点集。
    """
    angle_arr = np.asarray(angles_deg, dtype=np.float64).ravel()
    dist_arr = np.asarray(distances, dtype=np.float64).ravel()
    if dist_arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # 由原始距离序列估计噪声，作为分段阈值
    sigma = _estimate_distance_noise(dist_arr)
    tolerance = float(np.clip(2.5 * sigma, 0.015, 0.30)) if sigma > 0 else None

    raw_points = polar_to_points(angle_arr, dist_arr)
    segments = _segment_runs(raw_points[:, :2], tolerance, corner_threshold_deg)

    # 段内滤波：窗口被限制在同一面墙内，且不触碰段端点。
    # 中值在距离空间（离群点是径向尖刺），均值在 xy 空间（墙在那里才是直线）。
    filtered = _filter_runs(dist_arr, segments, median_window)

    points = polar_to_points(angle_arr, filtered)
    points[:, :2] = _mean_filter_xy(points[:, :2], segments, mean_window)

    return piecewise_line_fit(
        points,
        angle_threshold_deg=corner_threshold_deg,
        min_segment=MIN_SEGMENT,
        residual_tolerance=tolerance,
    )


def _estimate_distance_noise(distances: np.ndarray) -> float:
    """由二阶差分的中值绝对偏差估计距离序列的噪声标准差（米）。"""
    arr = np.asarray(distances, dtype=np.float64).ravel()
    if arr.size < 5:
        return 0.0
    second = np.diff(arr, n=2)
    median = float(np.median(second))
    mad = float(np.median(np.abs(second - median)))
    # 二阶差分的标准差为 sqrt(6)·σ
    return 1.4826 * mad / math.sqrt(6.0)


# --------------------------------------------------------------------------- #
# 扫描线程
# --------------------------------------------------------------------------- #


class ScanWorker(QThread):
    """在后台线程执行一次完整扫描。

    信号：
        * ``progress(int)``              —— 进度百分比 0~100。
        * ``sample_ready(float, float)`` —— 单个采样点（角度°, 距离 m）。
        * ``finished_ok(object, object, str)`` —— 原始点集、平滑点集（未开启平滑
          时为 ``None``）、设备名称。
        * ``failed(str)``                —— 扫描失败原因。
    """

    progress = pyqtSignal(int)
    sample_ready = pyqtSignal(float, float)
    finished_ok = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def __init__(self, config: ScanConfig, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        """请求停止扫描；已采集的数据仍会正常返回。"""
        self._cancelled = True

    # -- 入口 -------------------------------------------------------------- #

    def run(self) -> None:  # noqa: D102 - QThread 入口
        config = self.config
        device: BaseDevice | None = None
        fallback_note = ""

        try:
            device = create_device(
                config.mode,
                address=config.address,
                speed_rpm=config.speed_rpm,
                sample_interval=config.sample_interval,
                simulate=config.simulate,
            )
            try:
                device.connect()
            except DeviceError as exc:
                if config.mode != MODE_AUTO:
                    raise
                # 自动模式：蓝牙不可用时回退到模拟设备，保证界面可用
                logger.warning("蓝牙设备不可用，回退到模拟设备：%s", exc)
                brief = str(exc).strip().splitlines()[0][:40]
                fallback_note = f"（蓝牙不可用，已回退模拟：{brief}）"
                device = MockDevice(
                    speed_rpm=config.speed_rpm, sample_interval=config.sample_interval
                )
                device.connect()
        except DeviceError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程异常导致界面无响应
            logger.exception("设备初始化失败")
            self.failed.emit(f"设备初始化失败：{exc}")
            return

        try:
            self._scan_loop(device, fallback_note)
        finally:
            try:
                device.disconnect()
            except Exception:  # noqa: BLE001 - 断开失败不影响结果
                logger.warning("断开设备时出现异常", exc_info=True)

    # -- 主循环 ------------------------------------------------------------ #

    def _scan_loop(self, device: BaseDevice, fallback_note: str) -> None:
        config = self.config
        total_angle = 360.0 * max(1, int(config.turns))
        step = max(0.1, config.degrees_per_sample())
        count = max(2, int(round(total_angle / step)))
        angles = np.linspace(0.0, total_angle, count, endpoint=False)

        angles_done: list[float] = []
        distances: list[float] = []

        for index, angle in enumerate(angles):
            if self._cancelled:
                logger.info("扫描被用户取消，已完成 %d 个采样点", len(distances))
                break
            try:
                device.rotate_to(float(angle))
                distance = float(device.read_distance())
            except DeviceError as exc:
                self.failed.emit(f"扫描中断：{exc}")
                return
            except Exception as exc:  # noqa: BLE001 - 单点异常不应终止整个扫描
                logger.warning("第 %d 个采样点读取失败：%s", index, exc)
                continue

            angles_done.append(float(angle))
            distances.append(distance)
            self.sample_ready.emit(float(angle), distance)
            self.progress.emit(int((index + 1) * 100 / count))

        if not distances:
            self.failed.emit("未采集到任何数据，请检查设备连接。")
            return

        try:
            raw_points = polar_to_points(angles_done, distances)
            smooth_points = (
                smooth_scan(angles_done, distances) if config.smooth else None
            )
        except Exception as exc:  # noqa: BLE001 - 平滑失败时退化为原始点
            logger.exception("平滑处理失败，改用手工原始点")
            raw_points = polar_to_points(angles_done, distances)
            smooth_points = None
            self.failed.emit(f"平滑处理失败，已显示原始点：{exc}")

        self.progress.emit(100)
        self.finished_ok.emit(raw_points, smooth_points, f"{device.name}{fallback_note}")


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    """应用主窗口：工具栏 + 3D 视图 + 状态栏。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("环境测量 · 3D 扫描")
        self.resize(1080, 720)

        self.config: ScanConfig = load_config()

        # -- 中央 3D 视图 -------------------------------------------------- #
        self.view = ScanView(self)
        self.setCentralWidget(self.view)

        # -- 扫描状态 ------------------------------------------------------ #
        self._worker: ScanWorker | None = None
        self._pending_angles: list[float] = []
        self._pending_distances: list[float] = []
        self._points: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        # 定时刷新预览，避免每个采样点都重建图元
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_INTERVAL_MS)
        self._preview_timer.timeout.connect(self._flush_preview)

        self._build_toolbar()
        self._build_statusbar()
        self._apply_config_to_status()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #

    def _build_toolbar(self) -> None:
        """构建左上角工具栏：绘制 / 导出 / 设置。"""
        style = self.style()

        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self.action_draw = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "绘制", self
        )
        self.action_draw.setShortcut(QKeySequence("F5"))
        self.action_draw.setToolTip("开始扫描并绘制 3D 空间平面图（F5）")
        self.action_draw.triggered.connect(self._toggle_scan)

        self.action_export = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "导出", self
        )
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.setToolTip("导出当前扫描结果（Ctrl+E）")
        self.action_export.triggered.connect(self._on_export)

        self.action_settings = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), "设置", self
        )
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.action_settings.setToolTip("蓝牙与扫描参数设置（Ctrl+,）")
        self.action_settings.triggered.connect(self._on_settings)

        for action in (self.action_draw, self.action_export, self.action_settings):
            toolbar.addAction(action)

        # 复位视角不占用工具栏按钮，仅提供快捷键
        self.action_reset_view = QAction("复位视角", self)
        self.action_reset_view.setShortcut(QKeySequence("Ctrl+R"))
        self.action_reset_view.triggered.connect(self._on_reset_view)
        self.addAction(self.action_reset_view)

        self._toolbar = toolbar

    def _build_statusbar(self) -> None:
        """构建状态栏：提示文本 + 进度条 + 操作说明。"""
        bar = self.statusBar()

        self.status_label = QLabel("就绪", self)
        bar.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        bar.addPermanentWidget(self.progress_bar)

        self.count_label = QLabel("点数：0", self)
        bar.addPermanentWidget(self.count_label)

        hint = QLabel("左键拖动旋转 · 滚轮缩放 · 中键平移 · Ctrl+R 复位", self)
        hint.setEnabled(False)
        bar.addPermanentWidget(hint)

    def _apply_config_to_status(self) -> None:
        """把当前配置摘要显示在状态栏。"""
        cfg = self.config
        if cfg.simulate:
            mode_text = "模拟运行（不通信）"
        else:
            mode_text = {
                MODE_AUTO: "自动",
                MODE_MOCK: "模拟设备",
                "ble": "蓝牙设备",
            }.get(cfg.mode, cfg.mode)
        self.status_label.setText(
            f"就绪 · {mode_text} · {cfg.turns} 圈 · "
            f"平滑{'开' if cfg.smooth else '关'} · {cfg.speed_rpm:.1f} RPM"
        )

    # ------------------------------------------------------------------ #
    # 「绘制」
    # ------------------------------------------------------------------ #

    def _toggle_scan(self) -> None:
        """点击「绘制」开始扫描；扫描中再次点击则停止。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.status_label.setText("正在停止扫描…")
            self.action_draw.setEnabled(False)
            return
        self._start_scan()

    def _start_scan(self) -> None:
        """清空视图并启动扫描线程。"""
        self.view.clear_scan()
        self._points = np.zeros((0, 3), dtype=np.float64)
        self._pending_angles.clear()
        self._pending_distances.clear()
        self._update_count_label()

        worker = ScanWorker(self.config, self)
        worker.sample_ready.connect(self._on_sample)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_stopped)
        self._worker = worker

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.action_draw.setText("停止")
        self.action_draw.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.action_draw.setToolTip("停止当前扫描")
        self.action_export.setEnabled(False)
        self.status_label.setText("扫描中…")

        self._preview_timer.start()
        worker.start()

    def _on_sample(self, angle: float, distance: float) -> None:
        """接收单个采样点（主线程槽函数）。"""
        self._pending_angles.append(angle)
        self._pending_distances.append(distance)

    def _flush_preview(self) -> None:
        """把缓存中的采样点绘制到 3D 视图。"""
        if not self._pending_distances:
            return
        try:
            distances = np.asarray(self._pending_distances, dtype=np.float64)
            if self.config.smooth:
                distances = causal_smooth(distances, window=5)
            points = polar_to_points(self._pending_angles, distances)
            self.view.set_scan(points)
            self._points = points
            self._update_count_label()
        except Exception:  # noqa: BLE001 - 预览异常不应中断扫描
            logger.exception("实时预览刷新失败")

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(int(value))

    def _on_finished(self, raw_points: object, smooth_points: object, device_name: str) -> None:
        """扫描完成：保留最终图形。"""
        self._preview_timer.stop()
        self._flush_preview()

        points = smooth_points if smooth_points is not None else raw_points
        try:
            self.view.set_scan(points)
            self.view.reset_view(points)  # 按数据范围自动取景
            self._points = np.asarray(points, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.exception("绘制最终结果失败")
            self.status_label.setText(f"绘制失败：{exc}")
            return

        self._update_count_label()
        smooth_text = "已平滑" if smooth_points is not None else "未平滑"
        self.status_label.setText(
            f"扫描完成 · {device_name} · {self._points.shape[0]} 点 · {smooth_text}"
        )

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"扫描失败：{message}")
        logger.error("扫描失败：%s", message)
        QMessageBox.warning(self, "扫描失败", message)

    def _on_worker_stopped(self) -> None:
        """线程结束后的界面复位。"""
        self._preview_timer.stop()
        self.progress_bar.setVisible(False)
        self.action_draw.setText("绘制")
        self.action_draw.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.action_draw.setToolTip("开始扫描并绘制 3D 空间平面图（F5）")
        self.action_draw.setEnabled(True)
        self.action_export.setEnabled(self._points.shape[0] > 0)
        self._worker = None

    # ------------------------------------------------------------------ #
    # 「导出」
    # ------------------------------------------------------------------ #

    def _on_export(self) -> None:
        """弹出保存对话框，调用 :func:`exporter.export_scan`。"""
        if self._points.shape[0] == 0:
            QMessageBox.information(self, "导出", "当前没有扫描数据，请先点击「绘制」。")
            return

        path, _selected = QFileDialog.getSaveFileName(
            self,
            "导出扫描数据",
            suggested_filename(),
            "所有文件 (*)",
        )
        if not path:
            return

        result = export_scan(self._points, path, fmt=None)
        self.status_label.setText(result.message.splitlines()[0])

        # 格式未定时只做提示（information），不作为错误处理
        if result.ok or FORMAT_PENDING_MESSAGE in result.message:
            QMessageBox.information(self, "导出", result.message)
        else:
            QMessageBox.warning(self, "导出", result.message)

    # ------------------------------------------------------------------ #
    # 「设置」
    # ------------------------------------------------------------------ #

    def _on_settings(self) -> None:
        """打开设置对话框，确定后保存并应用到后续扫描。"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.config = dialog.config()
            self._apply_config_to_status()
            logger.info("设置已更新：%s", self.config)

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #

    def _on_reset_view(self) -> None:
        self.view.reset_view(self._points if self._points.shape[0] else None)

    def _update_count_label(self) -> None:
        self.count_label.setText(f"点数：{self.view.point_count}")

    def closeEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QMainWindow
        """退出前停止扫描线程，避免残留线程导致崩溃。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self._preview_timer.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


#: 这些 Qt 平台插件无法承载 QOpenGLWidget，截图/显示窗口必然失败
_HEADLESS_PLATFORMS = {"offscreen", "minimal", "vnc"}


def _gl_available() -> bool:
    """粗略判断当前平台能否真正显示 OpenGL 窗口。

    只识别无头平台插件——它们一定会失败且能安全判断。反过来说，本函数返回
    ``True`` **并不保证** GL 可用：GLX 配置损坏时，Qt 会在 C++ 层直接 abort，
    Python 无法拦截（这也是本函数不真的去创建 GL 上下文的原因）。
    """
    app = QApplication.instance()
    return app is None or app.platformName() not in _HEADLESS_PLATFORMS


def _save_screenshot(window: MainWindow, path: str) -> bool:
    """抓取 3D 视图与整个窗口并保存为图片，成功返回 ``True``。

    ``grabFramebuffer()`` 只在 GL 上下文可用时有效；本机等没有 OpenGL 的环境下
    窗口根本无法 ``show()``，因此这里的失败是环境问题而非绘制问题。
    """
    target = Path(path).expanduser()
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        target = target.with_suffix(".png")

    saved: list[Path] = []

    # 1) 3D 视图的帧缓冲——最能说明渲染是否正常
    try:
        image = window.view.grabFramebuffer()
    except Exception as exc:  # noqa: BLE001 - 抓帧失败不应让进程崩溃
        print(f"[selftest] 抓取帧缓冲失败：{exc}")
        image = None
    if image is not None and not image.isNull():
        view_path = target.with_name(f"{target.stem}_3d{target.suffix}")
        if image.save(str(view_path)):
            saved.append(view_path)
            print(f"[selftest] 3D 视图截图：{view_path}  ({image.width()}x{image.height()})")
        else:
            print(f"[selftest] 3D 视图截图写入失败：{view_path}")
    else:
        print("[selftest] 3D 视图帧缓冲为空——通常是没有可用的 OpenGL 环境。")

    # 2) 整个窗口（工具栏 + 状态栏 + 3D 视图）
    try:
        pixmap = window.grab()
        if not pixmap.isNull() and pixmap.save(str(target)):
            saved.append(target)
            print(f"[selftest] 窗口截图：{target}  ({pixmap.width()}x{pixmap.height()})")
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] 抓取窗口失败：{exc}")

    return bool(saved)


def _run_selftest(
    window: MainWindow,
    timeout_ms: int = 60000,
    screenshot: str | None = None,
) -> int:
    """无人值守自检：用「模拟运行」跑一次扫描，完成后自动退出。

    仅用于开发期验证线程、信号与绘制链路是否正常；全程不进行真实蓝牙通信。

    传入 ``screenshot`` 时会先 ``show()`` 窗口，扫描结束后把 3D 视图与整个窗口
    各存一张图片，便于在支持 OpenGL 的机器上确认渲染效果。
    """
    app = QApplication.instance()
    result = {"code": 1}

    def _finish() -> None:
        if app is not None:
            app.quit()

    def _capture() -> None:
        if screenshot and _save_screenshot(window, screenshot):
            result["shot"] = 0
        _finish()

    def _on_finished(raw: object, smooth: object, name: str) -> None:
        raw_count = np.asarray(raw).shape[0]
        smooth_count = np.asarray(smooth).shape[0] if smooth is not None else 0
        print(f"[selftest] 设备：{name}")
        print(f"[selftest] 原始点数：{raw_count}，平滑点数：{smooth_count}")
        print(f"[selftest] 视图点数：{window.view.point_count}")
        if window.view.point_count > 0:
            points = np.asarray(window._points)
            print(
                "[selftest] 坐标范围："
                f"x [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] "
                f"y [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] "
                f"z [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
            )
            result["code"] = 0
        if screenshot:
            # 留出时间让 GL 视图用最终数据重绘一次，再抓帧
            QTimer.singleShot(800, _capture)
        else:
            _finish()

    def _start() -> None:
        window._start_scan()
        if window._worker is not None:
            window._worker.finished_ok.connect(_on_finished)
            window._worker.failed.connect(lambda msg: (print(f"[selftest] 扫描失败：{msg}"), _finish()))

    if screenshot:
        if not _gl_available():
            app_name = QApplication.instance().platformName() if app else "?"
            print(
                f"[selftest] 当前平台插件「{app_name}」无法显示 OpenGL 窗口，跳过截图。\n"
                "[selftest] 请在支持 OpenGL 的图形环境下重新运行：\n"
                f"[selftest]     python main.py --screenshot {screenshot}"
            )
            return 1
        window.show()

    QTimer.singleShot(300, _start)
    QTimer.singleShot(timeout_ms, lambda: (print("[selftest] 超时"), _finish()))
    return int(app.exec()) if app is not None else 1


def main(argv: list[str] | None = None) -> int:
    """程序入口。

    支持以下参数：

    * ``--selftest``：以「模拟运行」自动跑一次扫描后退出，便于验证环境。
    * ``--screenshot PATH``：配合 ``--selftest``，显示窗口并把 3D 视图与整个窗口
      截图保存到 ``PATH``（另存一张 ``*_3d.png``），用于确认渲染效果。
    * ``--debug``：输出调试日志。
    """
    args = list(sys.argv if argv is None else argv)
    selftest = "--selftest" in args

    screenshot: str | None = None
    for flag in ("--screenshot", "--shot"):
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args) and not args[index + 1].startswith("-"):
                screenshot = args[index + 1]
            else:
                screenshot = "screenshot.png"
            selftest = True  # 截图必然伴随一次自检扫描
            break

    logging.basicConfig(
        level=logging.DEBUG if "--debug" in args else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(args)
    app.setApplicationName("MeasureApp")
    app.setOrganizationName("EnvironmentMeasurement")

    window = MainWindow()

    if selftest:
        # 自检使用「模拟运行」+ 较快的参数：不碰任何真实设备，缩短等待时间
        window.config = ScanConfig(mode=MODE_MOCK, turns=1, smooth=True, speed_rpm=60.0,
                                   sample_interval=0.02, simulate=True)
        logger.info("自检模式（模拟运行）：%s", window.config)
        return _run_selftest(window, screenshot=screenshot)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
