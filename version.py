"""应用版本信息。

单独成模块是为了让主窗口与「关于」对话框都能引用，而不产生循环导入。
"""

from __future__ import annotations

__all__ = ["APP_TITLE", "APP_VERSION", "APP_SUBTITLE"]

#: 窗口标题与「关于」页显示的软件名
APP_TITLE = "激光测距平面绘图"

#: 软件版本
APP_VERSION = "2.0.0"

#: 一句话说明
APP_SUBTITLE = "蓝牙激光测距仪 · 2D 俯视平面图"
