"""2D 俯视平面图视图。

:class:`ScanView` 用 ``QPainter`` 自绘，不依赖 OpenGL / pyqtgraph：原点 (0, 0)
即测距仪中心，x 向右、y 向上，单位为米。自带鼠标交互——滚轮缩放、左键拖动
平移、双击复位。

对外只暴露 :meth:`ScanView.set_scan` / :meth:`ScanView.clear_scan` /
:meth:`ScanView.reset_view` / :meth:`ScanView.set_show_labels`。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

__all__ = ["ScanView"]

#: 取景时数据半径占窗口短边的比例
_FIT_RATIO = 0.42
#: 缩放下限 / 上限（像素/米）
_SCALE_RANGE = (0.5, 20000.0)
#: 网格步进候选值（米），取第一个在屏幕上不小于 ``_MIN_GRID_PX`` 的
_GRID_STEPS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
_MIN_GRID_PX = 38.0

#: 配色
_BG = QColor(24, 27, 34)
_GRID = QColor(58, 64, 78)
_AXIS = QColor(96, 104, 122)
_LINE = QColor(90, 216, 255)
_POINT = QColor(190, 235, 255, 210)
_ORIGIN = QColor(255, 214, 64)
_TEXT = QColor(170, 180, 196)


class ScanView(QWidget):
    """扫描结果 2D 俯视视图。"""

    #: 数据点半径（屏幕像素）
    POINT_RADIUS = 2.6
    #: 点数超过该值时不再逐个画点，只画折线（避免重绘过慢）
    MAX_POINT_MARKERS = 4000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._points: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._closed: bool = True
        self._show_labels: bool = False

        #: 视图变换：屏幕 = 窗口中心 + (世界坐标 - _center) * _scale（y 轴取反）
        self._scale: float = 40.0
        self._center: tuple[float, float] = (0.0, 0.0)
        self._auto_fit: bool = True

        self._panning: bool = False
        self._last_pos: QPointF | None = None

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    @property
    def point_count(self) -> int:
        """当前显示的点数。"""
        return int(self._points.shape[0])

    @property
    def show_labels(self) -> bool:
        """是否显示每个点到测距仪中心的距离。"""
        return self._show_labels

    def set_show_labels(self, enabled: bool) -> None:
        """开关「点距」标注。"""
        enabled = bool(enabled)
        if enabled != self._show_labels:
            self._show_labels = enabled
            self.update()

    def set_scan(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        closed: bool = True,
    ) -> None:
        """显示一组扫描点。

        :param points: 形状 ``(N, 2)`` 的坐标数组（米）。
        :param closed: 是否把最后一个点连回第一个点（整圈采集时为 ``True``）。
        """
        self._points = self._as_points(points)
        self._closed = bool(closed)
        if self._auto_fit:
            self._fit()
        self.update()

    def clear_scan(self) -> None:
        """清空图形与点位。"""
        self._points = np.zeros((0, 2), dtype=np.float64)
        self.update()

    def reset_view(self, points: Sequence[Sequence[float]] | np.ndarray | None = None) -> None:
        """复位视角：重新按数据范围取景，并恢复自动取景。"""
        if points is not None:
            self._points = self._as_points(points)
        self._auto_fit = True
        self._fit()
        self.update()

    # ------------------------------------------------------------------ #
    # 坐标变换
    # ------------------------------------------------------------------ #

    def _to_screen(self, x: float, y: float) -> QPointF:
        cx, cy = self._center
        return QPointF(
            self.width() / 2.0 + (x - cx) * self._scale,
            self.height() / 2.0 - (y - cy) * self._scale,
        )

    def _to_world(self, pos: QPointF) -> tuple[float, float]:
        cx, cy = self._center
        return (
            cx + (pos.x() - self.width() / 2.0) / self._scale,
            cy - (pos.y() - self.height() / 2.0) / self._scale,
        )

    def _fit(self) -> None:
        """按数据范围调整缩放，始终以测距仪（原点）为中心。"""
        self._center = (0.0, 0.0)
        if self._points.shape[0]:
            radius = float(np.max(np.hypot(self._points[:, 0], self._points[:, 1])))
        else:
            radius = 5.0
        radius = max(radius, 0.5)
        span = max(1.0, min(self.width(), self.height()))
        self._scale = float(
            np.clip(span * _FIT_RATIO / radius, *_SCALE_RANGE)
        )

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #

    def paintEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), _BG)

        step = self._draw_grid(painter)
        self._draw_axes(painter)
        self._draw_scan(painter)
        self._draw_origin(painter)
        if self._show_labels:
            self._draw_labels(painter)
        self._draw_hud(painter, step)
        painter.end()

    def _grid_step(self) -> float:
        for step in _GRID_STEPS:
            if step * self._scale >= _MIN_GRID_PX:
                return step
        return _GRID_STEPS[-1]

    def _draw_grid(self, painter: QPainter) -> float:
        """画网格，返回本次使用的网格步进（米）。"""
        step = self._grid_step()
        left, bottom = self._to_world(QPointF(0.0, float(self.height())))
        right, top = self._to_world(QPointF(float(self.width()), 0.0))

        pen = QPen(_GRID)
        pen.setWidth(1)
        painter.setPen(pen)

        x = math.floor(left / step) * step
        while x <= right:
            screen_x = self._to_screen(x, 0.0).x()
            painter.drawLine(QPointF(screen_x, 0.0), QPointF(screen_x, float(self.height())))
            x += step

        y = math.floor(bottom / step) * step
        while y <= top:
            screen_y = self._to_screen(0.0, y).y()
            painter.drawLine(QPointF(0.0, screen_y), QPointF(float(self.width()), screen_y))
            y += step

        return step

    def _draw_axes(self, painter: QPainter) -> None:
        pen = QPen(_AXIS)
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        origin = self._to_screen(0.0, 0.0)
        painter.drawLine(QPointF(0.0, origin.y()), QPointF(float(self.width()), origin.y()))
        painter.drawLine(QPointF(origin.x(), 0.0), QPointF(origin.x(), float(self.height())))

    def _draw_scan(self, painter: QPainter) -> None:
        if self._points.shape[0] == 0:
            return

        screen = [self._to_screen(float(x), float(y)) for x, y in self._points]

        if len(screen) >= 2:
            polygon = QPolygonF(list(screen))
            if self._closed and len(screen) >= 3:
                polygon.append(screen[0])
            pen = QPen(_LINE)
            pen.setWidthF(1.6)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(polygon)

        if len(screen) <= self.MAX_POINT_MARKERS:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_POINT)
            radius = self.POINT_RADIUS
            for point in screen:
                painter.drawEllipse(point, radius, radius)

    def _draw_origin(self, painter: QPainter) -> None:
        """测距仪中心标记。"""
        origin = self._to_screen(0.0, 0.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_ORIGIN)
        painter.drawEllipse(origin, 4.0, 4.0)

    def _draw_labels(self, painter: QPainter) -> None:
        """标注每个点到测距仪中心的距离。

        标注密集时按屏幕位置去重：同一个约 30×13 像素的格子里只画一个标注，
        避免上千个点糊成一片；放大后格子变大，标注自然全部显现。
        """
        if self._points.shape[0] == 0:
            return

        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        painter.setPen(QPen(_TEXT))

        distances = np.hypot(self._points[:, 0], self._points[:, 1])
        occupied: set[tuple[int, int]] = set()
        label_width = max(30, painter.fontMetrics().horizontalAdvance("0.00 m") + 4)

        for (x, y), distance in zip(self._points, distances):
            point = self._to_screen(float(x), float(y))
            cell = (int(point.x() // label_width), int(point.y() // 13))
            if cell in occupied:
                continue
            occupied.add(cell)
            painter.drawText(point + QPointF(4.0, -4.0), f"{distance:.2f} m")

    def _draw_hud(self, painter: QPainter, step: float) -> None:
        """左下角提示：网格步进与当前点数。"""
        painter.setPen(QPen(_TEXT))
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        painter.drawText(
            QPointF(8.0, self.height() - 8.0),
            f"网格 {step:g} m · 点数 {self.point_count}",
        )

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #

    def wheelEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        delta = event.angleDelta().y()  # type: ignore[attr-defined]
        if not delta:
            return
        pos = event.position()  # type: ignore[attr-defined]

        world_x, world_y = self._to_world(pos)
        factor = 1.15 ** (delta / 120.0)
        self._scale = float(np.clip(self._scale * factor, *_SCALE_RANGE))
        # 让光标下的世界坐标保持不动
        self._center = (
            world_x + (self.width() / 2.0 - pos.x()) / self._scale,
            world_y - (self.height() / 2.0 - pos.y()) / self._scale,
        )
        self._auto_fit = False
        self.update()

    def mousePressEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        if event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            self._panning = True
            self._last_pos = event.position()  # type: ignore[attr-defined]
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        if not self._panning or self._last_pos is None:
            return
        pos = event.position()  # type: ignore[attr-defined]
        delta_x = pos.x() - self._last_pos.x()
        delta_y = pos.y() - self._last_pos.y()
        cx, cy = self._center
        self._center = (cx - delta_x / self._scale, cy + delta_y / self._scale)
        self._last_pos = pos
        self._auto_fit = False
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        if event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            self._panning = False
            self._last_pos = None
            self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseDoubleClickEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QWidget
        self.reset_view()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """把任意输入规范化为 ``(N, 2)`` 的 float64 数组。"""
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            if arr.size % 2 != 0:
                raise ValueError(f"点集形状非法，无法解析为 (N, 2)：{arr.shape}")
            arr = arr.reshape(-1, 2)
        # 过滤 NaN / Inf，避免绘制异常
        return arr[np.isfinite(arr).all(axis=1)]
