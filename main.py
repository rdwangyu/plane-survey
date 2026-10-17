"""程序入口与主窗口。

界面结构::

    ┌──────────────────────────────────────────────┐
    │ 连接 │ 绘制 │ 编辑 │ 导出 │ 设置 │ 关于        │ ← 顶部菜单栏
    ├──────────────────────────────────────────────┤
    │                                              │
    │         ScanView 2D 俯视平面图                │
    │                                              │
    ├──────────────────────────────────────────────┤
    │ 状态提示       [进度条]   点数   交互提示      │
    └──────────────────────────────────────────────┘

采集在 :class:`ScanWorker` 线程中执行，通过信号把每一帧 ``{点序号, 距离}``
换算出的 ``(角度, 距离)`` 实时回传主线程绘制，因此界面不会卡死。距离→角度的
换算是 ``角度 = 点序号 / 一圈点数 * 360``，随后生成 2D 俯视坐标并连线。

平滑算法（分段最小二乘直线拟合等）也放在工作线程内完成，避免大点集阻塞界面。
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
)

from device import (
    DeviceError,
    BaseDevice,
    BleDevice,
    SimulatedDevice,
    frame_angle_deg,
)
from dialogs import AboutDialog, ConnectDialog
from exporter import export_scan, suggested_filename
from plot2d import ScanView
from settings_dialog import ScanConfig, SettingsDialog, load_config, save_config
from version import APP_TITLE, APP_VERSION

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
    """把极坐标采样转换为 2D 俯视坐标。

    ``x = d * cos(θ)``，``y = d * sin(θ)``。

    :param angles_deg: 角度序列（度）。
    :param distances: 距离序列（米）。
    :returns: 形状 ``(N, 2)`` 的坐标数组。
    """
    angle_arr = np.asarray(angles_deg, dtype=np.float64).ravel()
    dist_arr = np.asarray(distances, dtype=np.float64).ravel()
    if angle_arr.size != dist_arr.size:
        raise ValueError(f"角度与距离长度不一致：{angle_arr.size} != {dist_arr.size}")
    if angle_arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    rad = np.radians(angle_arr)
    return np.column_stack((dist_arr * np.cos(rad), dist_arr * np.sin(rad)))


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

    :param points: 形状 ``(N, 2)`` 的点集。
    :param angle_threshold_deg: 合并相邻段时的方向夹角阈值（度）。
    :param min_segment: 点数少于该值时不拟合，直接原样返回。
    :param residual_tolerance: 分段阈值（米）；``None`` 时由数据自动估计。
    :returns: 拟合后的点集，形状与输入一致；点数不足时原样返回。
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2 * max(2, min_segment):
        return arr.copy()

    xy = arr
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

    return result


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

    :returns: 形状 ``(N, 2)`` 的平滑后点集。
    """
    angle_arr = np.asarray(angles_deg, dtype=np.float64).ravel()
    dist_arr = np.asarray(distances, dtype=np.float64).ravel()
    if dist_arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # 由原始距离序列估计噪声，作为分段阈值
    sigma = _estimate_distance_noise(dist_arr)
    tolerance = float(np.clip(2.5 * sigma, 0.015, 0.30)) if sigma > 0 else None

    raw_points = polar_to_points(angle_arr, dist_arr)
    segments = _segment_runs(raw_points, tolerance, corner_threshold_deg)

    # 段内滤波：窗口被限制在同一面墙内，且不触碰段端点。
    # 中值在距离空间（离群点是径向尖刺），均值在 xy 空间（墙在那里才是直线）。
    filtered = _filter_runs(dist_arr, segments, median_window)

    points = polar_to_points(angle_arr, filtered)
    points = _mean_filter_xy(points, segments, mean_window)

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
# 采集线程
# --------------------------------------------------------------------------- #


class ScanWorker(QThread):
    """在后台线程执行一次整圈采集。

    每收到一帧 ``{点序号, 距离}``，按 ``角度 = 点序号 / 一圈点数 * 360`` 换算后
    发出 :attr:`sample_ready`。

    信号：
        * ``progress(int)``              —— 进度百分比 0~100。
        * ``sample_ready(float, float)`` —— 单个采样点（角度°, 距离 m）。
        * ``finished_ok(object, object, str)`` —— 原始点集、平滑点集（未开启平滑
          时为 ``None``）、设备名称。
        * ``failed(str)``                —— 采集失败原因。
    """

    progress = pyqtSignal(int)
    sample_ready = pyqtSignal(float, float)
    finished_ok = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        device: BaseDevice,
        points_per_turn: int,
        *,
        smooth: bool = True,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self.device = device
        self.points_per_turn = max(1, int(points_per_turn))
        self.smooth = bool(smooth)

        self._cancelled = False
        #: 置位 = 正在接收；清除 = 暂停接收
        self._receiving = threading.Event()
        self._receiving.set()

    # -- 控制 -------------------------------------------------------------- #

    def pause(self) -> None:
        """暂停接收：不再从设备取帧，但保持连接。"""
        self._receiving.clear()

    def resume(self) -> None:
        """恢复接收。"""
        self._receiving.set()

    def cancel(self) -> None:
        """请求停止采集；已采集的数据仍会正常返回。"""
        self._cancelled = True
        self._receiving.set()  # 暂停中也能退出

    # -- 入口 -------------------------------------------------------------- #

    def run(self) -> None:  # noqa: D102 - QThread 入口
        try:
            self.device.start_scan(self.points_per_turn)
        except DeviceError as exc:
            self.failed.emit(f"启动采集失败：{exc}")
            return
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程异常导致界面无响应
            logger.exception("启动采集失败")
            self.failed.emit(f"启动采集失败：{exc}")
            return

        try:
            self._scan_loop()
        finally:
            try:
                self.device.stop_scan()
            except Exception:  # noqa: BLE001 - 停止失败不影响结果
                logger.warning("停止采集时出现异常", exc_info=True)

    # -- 主循环 ------------------------------------------------------------ #

    def _scan_loop(self) -> None:
        total = self.points_per_turn
        angles: list[float] = []
        distances: list[float] = []

        for count in range(total):
            self._receiving.wait()  # 暂停时停在这里
            if self._cancelled:
                logger.info("采集被用户停止，已收到 %d 帧", len(distances))
                break

            try:
                frame = self.device.read_frame()
            except DeviceError as exc:
                self.failed.emit(f"接收中断：{exc}")
                return
            except Exception as exc:  # noqa: BLE001 - 单帧异常不应终止整圈
                logger.warning("第 %d 帧读取失败：%s", count, exc)
                continue

            angle = frame_angle_deg(frame.index, total)
            angles.append(angle)
            distances.append(float(frame.distance))
            self.sample_ready.emit(angle, float(frame.distance))
            self.progress.emit(int((count + 1) * 100 / total))

        if not distances:
            self.failed.emit("未采集到任何数据，请检查设备连接。")
            return

        try:
            raw_points = polar_to_points(angles, distances)
            smooth_points = smooth_scan(angles, distances) if self.smooth else None
        except Exception as exc:  # noqa: BLE001 - 平滑失败时退化为原始点
            logger.exception("平滑处理失败，改用原始点")
            raw_points = polar_to_points(angles, distances)
            smooth_points = None
            self.failed.emit(f"平滑处理失败，已显示原始点：{exc}")

        self.progress.emit(100)
        self.finished_ok.emit(raw_points, smooth_points, self.device.name)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    """应用主窗口：菜单栏 + 2D 俯视视图 + 状态栏。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} {APP_VERSION}")
        self.resize(1080, 720)

        self.config: ScanConfig = load_config()

        # -- 中央 2D 视图 -------------------------------------------------- #
        self.view = ScanView(self)  # 「点距」默认关闭
        self.setCentralWidget(self.view)

        # -- 设备与采集状态 ------------------------------------------------ #
        self._device: BleDevice | None = None  # 已连接的蓝牙测距仪
        self._worker: ScanWorker | None = None
        self._angles: list[float] = []
        self._distances: list[float] = []
        self._points: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._discard: bool = False  # 复位后丢弃线程回传的结果
        self._paused: bool = False

        # 定时刷新预览，避免每帧都重建图元
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_INTERVAL_MS)
        self._preview_timer.timeout.connect(self._flush_preview)

        self._build_menus()
        self._build_statusbar()
        self._update_actions()
        self._apply_config_to_status()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #

    def _build_menus(self) -> None:
        """构建顶部菜单栏：连接 | 绘制 | 编辑 | 导出 | 设置 | 关于。"""
        bar = self.menuBar()

        # -- 连接 ---------------------------------------------------------- #
        self.action_connect = QAction("连接蓝牙测距仪…", self)
        self.action_connect.setShortcut(QKeySequence("Ctrl+K"))
        self.action_connect.setStatusTip("扫描附近的蓝牙测距仪并建立连接")
        self.action_connect.triggered.connect(self._on_connect)

        self.action_disconnect = QAction("断开连接", self)
        self.action_disconnect.setStatusTip("断开当前蓝牙测距仪")
        self.action_disconnect.triggered.connect(self._on_disconnect)

        connect_menu = bar.addMenu("连接")
        connect_menu.addAction(self.action_connect)
        connect_menu.addAction(self.action_disconnect)

        # -- 绘制 ---------------------------------------------------------- #
        self.action_start = QAction("开始", self)
        self.action_start.setShortcut(QKeySequence("F5"))
        self.action_start.setStatusTip("开始采集（暂停后为「继续」）")
        self.action_start.triggered.connect(self._on_start)

        self.action_pause = QAction("暂停", self)
        self.action_pause.setShortcut(QKeySequence("F6"))
        self.action_pause.setStatusTip("暂停接收")
        self.action_pause.triggered.connect(self._on_pause)

        self.action_reset = QAction("复位", self)
        self.action_reset.setShortcut(QKeySequence("F7"))
        self.action_reset.setStatusTip("清空图形与点位")
        self.action_reset.triggered.connect(self._on_reset)

        draw_menu = bar.addMenu("绘制")
        draw_menu.addAction(self.action_start)
        draw_menu.addAction(self.action_pause)
        draw_menu.addAction(self.action_reset)

        # -- 编辑 ---------------------------------------------------------- #
        self.action_smooth = QAction("平滑", self)
        self.action_smooth.setCheckable(True)
        self.action_smooth.setChecked(self.config.smooth)
        self.action_smooth.setStatusTip("把折线边拉直（分段最小二乘直线拟合）")
        self.action_smooth.toggled.connect(self._on_toggle_smooth)

        self.action_labels = QAction("点距", self)
        self.action_labels.setCheckable(True)
        self.action_labels.setStatusTip("显示每个点到测距仪中心的距离")
        self.action_labels.toggled.connect(self._on_toggle_labels)

        edit_menu = bar.addMenu("编辑")
        edit_menu.addAction(self.action_smooth)
        edit_menu.addAction(self.action_labels)

        # -- 导出 ---------------------------------------------------------- #
        self.action_export = QAction("导出 DXF…", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.setStatusTip("导出当前 2D 图形为 DXF，供 CAD 使用")
        self.action_export.triggered.connect(self._on_export)

        export_menu = bar.addMenu("导出")
        export_menu.addAction(self.action_export)

        # -- 设置 ---------------------------------------------------------- #
        self.action_settings = QAction("设置…", self)
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.action_settings.setStatusTip("运行方式与采集参数")
        self.action_settings.triggered.connect(self._on_settings)

        settings_menu = bar.addMenu("设置")
        settings_menu.addAction(self.action_settings)

        # -- 关于 ---------------------------------------------------------- #
        self.action_about = QAction("关于…", self)
        self.action_about.setStatusTip("软件版本与测距仪信息")
        self.action_about.triggered.connect(self._on_about)

        about_menu = bar.addMenu("关于")
        about_menu.addAction(self.action_about)

        # 复位视角不占菜单位置，仅提供快捷键
        self.action_reset_view = QAction("复位视角", self)
        self.action_reset_view.setShortcut(QKeySequence("Ctrl+R"))
        self.action_reset_view.triggered.connect(
            lambda: self.view.reset_view(self._points if self._points.shape[0] else None)
        )
        self.addAction(self.action_reset_view)

    def _build_statusbar(self) -> None:
        """构建状态栏：提示文本 + 进度条 + 点数 + 操作说明。"""
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

        hint = QLabel("滚轮缩放 · 左键拖动平移 · 双击复位", self)
        hint.setEnabled(False)
        bar.addPermanentWidget(hint)

    def _apply_config_to_status(self) -> None:
        """把当前配置摘要显示在状态栏。"""
        cfg = self.config
        mode_text = "模拟运行（不通信）" if cfg.simulate else self._device_text()
        self.status_label.setText(
            f"就绪 · {mode_text} · 一圈 {cfg.points_per_turn} 点 "
            f"（{cfg.degrees_per_point():.4g}°/点）"
        )

    def _device_text(self) -> str:
        """当前蓝牙设备描述。"""
        if self._device is not None and self._device.connected:
            return self._device.name
        return "未连接蓝牙测距仪"

    def _update_actions(self) -> None:
        """按采集状态刷新菜单项的可用性与文案。"""
        running = self._worker is not None and self._worker.isRunning()
        paused = running and self._paused

        self.action_start.setText("继续" if paused else "开始")
        self.action_start.setEnabled(not running or paused)
        self.action_pause.setEnabled(running and not paused)
        self.action_reset.setEnabled(running or self._points.shape[0] > 0)
        self.action_export.setEnabled(self._points.shape[0] > 0)
        self.action_settings.setEnabled(not running)
        self.action_connect.setEnabled(not running)
        self.action_disconnect.setEnabled(self._device is not None and not running)

    # ------------------------------------------------------------------ #
    # 「连接」
    # ------------------------------------------------------------------ #

    def _on_connect(self) -> None:
        """扫描并连接蓝牙测距仪。"""
        dialog = ConnectDialog(self, simulate=self.config.simulate)
        if dialog.exec() != ConnectDialog.DialogCode.Accepted:
            self._update_actions()
            return

        device = dialog.device()
        if device is None:
            return

        if self._device is not None:
            self._disconnect_device()
        self._device = device

        self.config.address = device.address
        try:
            save_config(self.config)
        except Exception:  # noqa: BLE001 - 地址保存失败不影响连接
            logger.warning("保存设备地址失败", exc_info=True)

        self.status_label.setText(f"已连接：{device.name}")
        self._update_actions()

    def _on_disconnect(self) -> None:
        self._disconnect_device()
        self.status_label.setText("已断开蓝牙测距仪")
        self._update_actions()

    def _disconnect_device(self) -> None:
        if self._device is None:
            return
        try:
            self._device.disconnect()
        except Exception:  # noqa: BLE001 - 断开失败不影响退出
            logger.warning("断开蓝牙设备时出现异常", exc_info=True)
        self._device = None

    # ------------------------------------------------------------------ #
    # 「绘制」
    # ------------------------------------------------------------------ #

    def _on_start(self) -> None:
        """开始采集；暂停中则继续。"""
        if self._worker is not None and self._worker.isRunning():
            if self._paused:
                self._paused = False
                self._worker.resume()
                self.status_label.setText("采集中…")
                self._update_actions()
            return
        self._start_scan()

    def _on_pause(self) -> None:
        """暂停接收。"""
        if self._worker is None or not self._worker.isRunning() or self._paused:
            return
        self._paused = True
        self._worker.pause()
        self.status_label.setText(f"已暂停接收 · 已收到 {len(self._distances)} 帧")
        self._update_actions()

    def _on_reset(self) -> None:
        """复位：停止采集并清空图形与点位。

        ``_discard`` 保持为 ``True``，直到下一次「开始」才清掉：线程被取消后仍会
        把已经采到的部分数据回传，若不屏蔽，复位后的图形会被这些数据重新画上。
        """
        self._discard = True
        self._preview_timer.stop()

        worker = self._worker
        if worker is not None:
            if worker.isRunning():
                worker.cancel()
                worker.wait(3000)  # 超时也不强求，finished 信号随后会收尾
            if not worker.isRunning():
                self._worker = None

        self._paused = False
        self._angles.clear()
        self._distances.clear()
        self._points = np.zeros((0, 2), dtype=np.float64)
        self.view.clear_scan()

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self._update_count_label()
        self.status_label.setText("已复位")
        self._apply_config_to_status()
        self._update_actions()

    def _start_scan(self) -> None:
        """清空视图并启动采集线程。"""
        device = self._acquire_device()
        if device is None:
            return

        self._discard = False
        self._paused = False
        self._angles.clear()
        self._distances.clear()
        self._points = np.zeros((0, 2), dtype=np.float64)
        self.view.clear_scan()
        self.view.reset_view()
        self._update_count_label()

        worker = ScanWorker(
            device, self.config.points_per_turn, smooth=self.action_smooth.isChecked(), parent=self
        )
        worker.sample_ready.connect(self._on_sample)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_stopped)
        self._worker = worker

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"采集中…（一圈 {self.config.points_per_turn} 个点）")

        self._preview_timer.start()
        worker.start()
        self._update_actions()

    def _acquire_device(self) -> BaseDevice | None:
        """取得本次采集要用的设备。"""
        if self.config.simulate:
            # 模拟运行：每次采集用一个全新的模拟设备，完全不碰蓝牙
            device = SimulatedDevice()
            device.connect()
            return device

        if self._device is not None and self._device.connected:
            return self._device

        if self._device is not None:  # 连接已失效
            self._disconnect_device()
        self.status_label.setText("未连接蓝牙测距仪")
        QMessageBox.warning(
            self,
            "绘制",
            "尚未连接蓝牙测距仪。\n\n"
            "请先点击「连接」菜单扫描并连接测距仪；\n"
            "或在「设置」中勾选「模拟运行」后离线演示。",
        )
        return None

    def _on_sample(self, angle: float, distance: float) -> None:
        """接收单个采样点（主线程槽函数）。

        暂停时直接丢弃：暂停的瞬间可能还有一帧已经发出、正排在主线程的事件队列
        里，若照收会让「暂停」看上去漏了一拍。
        """
        if self._paused:
            return
        self._angles.append(angle)
        self._distances.append(distance)

    def _flush_preview(self) -> None:
        """把缓存中的采样点绘制到 2D 视图。"""
        if not self._distances or self._discard:
            return
        try:
            distances = np.asarray(self._distances, dtype=np.float64)
            if self.action_smooth.isChecked():
                distances = causal_smooth(distances, window=5)
            points = polar_to_points(self._angles, distances)
            self.view.set_scan(points)
            self._points = points
            self._update_count_label()
        except Exception:  # noqa: BLE001 - 预览异常不应中断采集
            logger.exception("实时预览刷新失败")

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(int(value))

    def _on_finished(self, raw_points: object, smooth_points: object, device_name: str) -> None:
        """采集完成：保留最终图形。"""
        self._preview_timer.stop()
        if self._discard:
            return

        points = smooth_points if smooth_points is not None else raw_points
        try:
            arr = np.asarray(points, dtype=np.float64)
            self.view.set_scan(arr)
            self.view.reset_view(arr)  # 按数据范围自动取景
            self._points = arr
        except Exception as exc:  # noqa: BLE001
            logger.exception("绘制最终结果失败")
            self.status_label.setText(f"绘制失败：{exc}")
            return

        self._update_count_label()
        smooth_text = "已平滑" if smooth_points is not None else "未平滑"
        self.status_label.setText(
            f"采集完成 · {device_name} · {self._points.shape[0]} 点 · {smooth_text}"
        )
        self._update_actions()

    def _on_failed(self, message: str) -> None:
        if self._discard:
            return
        self.status_label.setText(f"采集失败：{message}")
        logger.error("采集失败：%s", message)
        QMessageBox.warning(self, "采集失败", message)

    def _on_worker_stopped(self) -> None:
        """线程结束后的界面复位。"""
        self._preview_timer.stop()
        self.progress_bar.setVisible(False)
        self._worker = None
        self._paused = False
        self._update_actions()

    # ------------------------------------------------------------------ #
    # 「编辑」
    # ------------------------------------------------------------------ #

    def _on_toggle_smooth(self, checked: bool) -> None:
        """平滑开关：把当前折线重新拉直 / 还原为原始点。"""
        self.config.smooth = bool(checked)
        try:
            save_config(self.config)
        except Exception:  # noqa: BLE001 - 保存失败不影响本次操作
            logger.warning("保存平滑开关失败", exc_info=True)
        self._render()

    def _on_toggle_labels(self, checked: bool) -> None:
        """点距开关：显示 / 隐藏每个点到测距仪中心的距离。"""
        self.view.set_show_labels(bool(checked))

    def _render(self) -> None:
        """按当前「平滑」开关重新计算并显示图形。"""
        if not self._distances:
            return
        try:
            points = (
                smooth_scan(self._angles, self._distances)
                if self.action_smooth.isChecked()
                else polar_to_points(self._angles, self._distances)
            )
        except Exception as exc:  # noqa: BLE001 - 平滑失败时退回原始点
            logger.exception("平滑处理失败")
            self.status_label.setText(f"平滑失败，已显示原始点：{exc}")
            points = polar_to_points(self._angles, self._distances)

        self._points = np.asarray(points, dtype=np.float64)
        self.view.set_scan(self._points)
        self._update_count_label()
        self.status_label.setText(
            f"{'已平滑' if self.action_smooth.isChecked() else '未平滑'} · "
            f"{self._points.shape[0]} 点"
        )
        self._update_actions()

    # ------------------------------------------------------------------ #
    # 「导出」
    # ------------------------------------------------------------------ #

    def _on_export(self) -> None:
        """导出当前 2D 图形为 DXF。"""
        if self._points.shape[0] < 2:
            QMessageBox.information(self, "导出", "当前没有可导出的图形，请先「绘制 → 开始」。")
            return

        path, _selected = QFileDialog.getSaveFileName(
            self,
            "导出 DXF",
            suggested_filename("dxf"),
            "DXF 图形交换文件 (*.dxf);;所有文件 (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".dxf"):
            path += ".dxf"

        result = export_scan(self._points, path, fmt="dxf")
        self.status_label.setText(result.message.splitlines()[0])
        if result.ok:
            QMessageBox.information(self, "导出", result.message)
        else:
            QMessageBox.warning(self, "导出", result.message)

    # ------------------------------------------------------------------ #
    # 「设置」/「关于」
    # ------------------------------------------------------------------ #

    def _on_settings(self) -> None:
        """打开设置对话框，确定后保存并应用到后续采集。"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.config = dialog.config()
            self.action_smooth.setChecked(self.config.smooth)
            self._apply_config_to_status()
            self._update_actions()
            logger.info("设置已更新：%s", self.config)

    def _on_about(self) -> None:
        """显示软件版本与测距仪信息。"""
        device: BaseDevice | None = self._device
        temporary: SimulatedDevice | None = None
        if device is None and self.config.simulate:
            # 模拟运行：不发送任何蓝牙指令，直接给出模拟设备信息
            temporary = SimulatedDevice()
            temporary.connect()
            temporary.start_scan(self.config.points_per_turn)
            device = temporary

        AboutDialog(self, device=device, simulate=self.config.simulate).exec()

        if temporary is not None:
            temporary.disconnect()

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #

    def _update_count_label(self) -> None:
        self.count_label.setText(f"点数：{self.view.point_count}")

    def closeEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QMainWindow
        """退出前停止采集线程并断开设备，避免残留线程导致崩溃。"""
        self._discard = True
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self._preview_timer.stop()
        self._disconnect_device()
        super().closeEvent(event)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def _save_screenshot(window: MainWindow, path: str) -> bool:
    """抓取整个窗口（含菜单栏、平面图、状态栏）并保存为图片。"""
    target = Path(path).expanduser()
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        target = target.with_suffix(".png")

    try:
        pixmap = window.grab()
    except Exception as exc:  # noqa: BLE001 - 抓图失败不应让进程崩溃
        print(f"[selftest] 抓取窗口失败：{exc}")
        return False

    if pixmap.isNull() or not pixmap.save(str(target)):
        print(f"[selftest] 窗口截图写入失败：{target}")
        return False

    print(f"[selftest] 窗口截图：{target}  ({pixmap.width()}x{pixmap.height()})")
    return True


def _run_selftest(
    window: MainWindow,
    timeout_ms: int = 60000,
    screenshot: str | None = None,
) -> int:
    """无人值守自检：用「模拟运行」跑一次采集，完成后自动退出。

    仅用于开发期验证线程、信号与绘制链路是否正常；全程不进行真实蓝牙通信。
    """
    app = QApplication.instance()
    result = {"code": 1}

    def _finish() -> None:
        if app is not None:
            app.quit()

    def _capture() -> None:
        if screenshot:
            _save_screenshot(window, screenshot)
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
                f"y [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]"
            )
            result["code"] = 0
        if screenshot:
            QTimer.singleShot(800, _capture)  # 留出重绘时间再抓图
        else:
            _finish()

    def _start() -> None:
        window._start_scan()
        if window._worker is not None:
            window._worker.finished_ok.connect(_on_finished)
            window._worker.failed.connect(
                lambda msg: (print(f"[selftest] 采集失败：{msg}"), _finish())
            )

    if screenshot:
        window.show()

    QTimer.singleShot(300, _start)
    QTimer.singleShot(timeout_ms, lambda: (print("[selftest] 超时"), _finish()))
    return int(app.exec()) if app is not None else 1


def main(argv: list[str] | None = None) -> int:
    """程序入口。

    支持以下参数：

    * ``--selftest``：以「模拟运行」自动跑一次采集后退出，便于验证环境。
    * ``--screenshot PATH``：配合 ``--selftest``，显示窗口并截图保存到 ``PATH``。
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
            selftest = True  # 截图必然伴随一次自检采集
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
        # 自检使用「模拟运行」+ 较少的点数：不碰任何真实设备，缩短等待时间
        window.config = ScanConfig(simulate=True, points_per_turn=360, smooth=True)
        logger.info("自检模式（模拟运行）：%s", window.config)
        return _run_selftest(window, screenshot=screenshot)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
