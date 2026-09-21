"""基于 ``pyqtgraph.opengl`` 的 3D 扫描视图。

:class:`ScanView` 继承自 :class:`pyqtgraph.opengl.GLViewWidget`，自带鼠标交互
（左键拖动旋转、滚轮缩放、中键平移），并在其上叠加：

* 地面网格与坐标轴；
* 扫描点云（按距离着色）；
* 扫描轮廓折线；
* 设备原点标记。

对外只暴露 :meth:`ScanView.set_scan` / :meth:`ScanView.clear_scan` /
:meth:`ScanView.reset_view` 三个方法，便于主窗口调用。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor

__all__ = ["ScanView", "distance_colors"]

#: 距离色带控制点（由近到远），采用类 turbo 的渐变
_RAMP: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.00, (0.19, 0.07, 0.36)),
    (0.25, (0.10, 0.50, 0.85)),
    (0.50, (0.15, 0.83, 0.55)),
    (0.75, (0.95, 0.75, 0.15)),
    (1.00, (0.90, 0.17, 0.13)),
)


def distance_colors(
    distances: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """按距离生成 RGBA 颜色数组，用于给点云着色。

    :param distances: 形状 ``(N,)`` 的距离序列（米）。
    :param alpha: 透明度，取值 0~1。
    :returns: 形状 ``(N, 4)`` 的 ``float32`` 数组，分量范围 0~1。
    """
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    # 所有距离相同时统一取色带中段，避免除零
    norm = np.full_like(values, 0.5) if span < 1e-9 else (values - lo) / span

    stops = np.array([stop for stop, _ in _RAMP], dtype=np.float64)
    colors = np.array([rgb for _, rgb in _RAMP], dtype=np.float64)

    out = np.empty((values.size, 4), dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(norm, stops, colors[:, channel])
    out[:, 3] = float(np.clip(alpha, 0.0, 1.0))
    return out


class ScanView(gl.GLViewWidget):
    """扫描结果 3D 视图。"""

    #: 点云中每个点的像素大小
    POINT_SIZE: float = 7.0
    #: 轮廓线宽
    LINE_WIDTH: float = 2.0

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent=parent)

        self.setBackgroundColor(QColor(22, 25, 32))
        self.opts["distance"] = 16.0
        self.setCameraPosition(distance=16.0, elevation=32.0, azimuth=-55.0)

        # -- 地面网格 ------------------------------------------------------ #
        self._grid = gl.GLGridItem()
        self._grid.setSize(x=40.0, y=40.0, z=1.0)
        self._grid.setSpacing(x=1.0, y=1.0, z=1.0)
        self._grid.setColor((120, 130, 150, 90))
        self.addItem(self._grid)

        # -- 坐标轴 -------------------------------------------------------- #
        self._axes = gl.GLAxisItem()
        self._axes.setSize(2.5, 2.5, 2.5)
        self.addItem(self._axes)

        # -- 扫描点云 ------------------------------------------------------ #
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=self.POINT_SIZE,
            pxMode=True,
        )
        self._scatter.setVisible(False)
        self.addItem(self._scatter)

        # -- 扫描轮廓折线 -------------------------------------------------- #
        self._line = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0.35, 0.85, 1.0, 0.9),
            width=self.LINE_WIDTH,
            antialias=True,
            mode="line_strip",
        )
        self._line.setVisible(False)
        self.addItem(self._line)

        # -- 设备原点标记 -------------------------------------------------- #
        self._origin = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 0.85, 0.25, 1.0]], dtype=np.float32),
            size=13.0,
            pxMode=True,
        )
        self.addItem(self._origin)

        self._point_count: int = 0

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    @property
    def point_count(self) -> int:
        """当前显示的点数。"""
        return self._point_count

    def set_scan(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        show_line: bool = True,
        auto_scale: bool = False,
    ) -> None:
        """显示一组扫描点。

        :param points: 形状 ``(N, 3)`` 的坐标数组。
        :param show_line: 是否同时绘制轮廓折线。
        :param auto_scale: 是否按数据范围自动调整网格与相机距离。
        """
        arr = self._as_points(points)
        self._point_count = int(arr.shape[0])

        if self._point_count == 0:
            self.clear_scan()
            return

        distances = np.hypot(arr[:, 0], arr[:, 1])

        # 点云
        self._scatter.setData(
            pos=arr.astype(np.float32),
            color=distance_colors(distances, alpha=0.95),
            size=self.POINT_SIZE,
        )
        self._scatter.setVisible(True)

        # 折线：至少两个点才画线
        if show_line and self._point_count >= 2:
            self._line.setData(pos=arr.astype(np.float32))
            self._line.setVisible(True)
        else:
            self._line.setVisible(False)

        if auto_scale:
            self._fit_to(arr)

    def clear_scan(self) -> None:
        """清空点云与折线（保留网格与坐标轴）。"""
        self._point_count = 0
        self._scatter.setVisible(False)
        self._line.setVisible(False)
        self._scatter.setData(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
        )
        self._line.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def reset_view(self, points: Sequence[Sequence[float]] | np.ndarray | None = None) -> None:
        """复位相机（可传入当前点集，按数据范围自动取景）。"""
        arr = self._as_points(points) if points is not None else self._current_points()
        if arr.shape[0] >= 3:
            self._fit_to(arr)
        else:
            self.opts["distance"] = 16.0
            self.setCameraPosition(distance=16.0, elevation=32.0, azimuth=-55.0)
        self.update()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """把任意输入规范化为 ``(N, 3)`` 的 float64 数组。"""
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            if arr.size % 3 != 0:
                raise ValueError(f"点集形状非法，无法解析为 (N, 3)：{arr.shape}")
            arr = arr.reshape(-1, 3)
        # 过滤 NaN / Inf，避免 OpenGL 绘制异常
        mask = np.isfinite(arr).all(axis=1)
        return arr[mask]

    def _current_points(self) -> np.ndarray:
        pos = getattr(self._scatter, "pos", None)
        if pos is None or len(pos) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(pos, dtype=np.float64)

    def _fit_to(self, arr: np.ndarray) -> None:
        """按数据包围盒调整网格尺寸与相机距离。"""
        radius = float(np.max(np.hypot(arr[:, 0], arr[:, 1]))) if arr.size else 5.0
        radius = max(radius, 1.0)

        # 网格取整到 5 米的整数倍，保证足够覆盖
        span = max(10.0, np.ceil(radius * 2.4 / 5.0) * 5.0)
        self._grid.setSize(x=span, y=span, z=1.0)

        self.opts["distance"] = radius * 2.6
        self.setCameraPosition(distance=radius * 2.6, elevation=32.0, azimuth=-55.0)
        self.update()
