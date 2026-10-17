"""设置对话框与配置持久化。

界面只保留两项：**运行方式**与**参数**（测距仪转一圈共返回多少个点）。连接方式
固定为蓝牙，设备由「连接」菜单扫描并选定，不再在设置里配置。

配置通过 :class:`PyQt6.QtCore.QSettings` 保存到系统标准位置（Linux 下通常是
``~/.config/EnvironmentMeasurement/MeasureApp.conf``），主窗口与设置对话框共用
:func:`load_config` / :func:`save_config` 两个函数。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from device import DEFAULT_POINTS_PER_TURN

__all__ = ["ScanConfig", "SettingsDialog", "load_config", "save_config"]

#: QSettings 组织名 / 应用名
ORG_NAME = "EnvironmentMeasurement"
APP_NAME = "MeasureApp"

# -- QSettings 键名 --------------------------------------------------------- #
_KEY_ADDRESS = "bluetooth/address"
_KEY_POINTS_PER_TURN = "scan/points_per_turn"
_KEY_SIMULATE = "device/simulate"
_KEY_SMOOTH = "edit/smooth"

#: 一圈点数的取值范围
POINTS_RANGE = (12, 20000)


@dataclass
class ScanConfig:
    """一次采集所需的全部参数。"""

    simulate: bool = False
    """运行方式：勾选时走模拟逻辑，不进行任何真实通信。"""

    address: str = ""
    """上一次连接的蓝牙测距仪地址，由「连接」菜单写入。"""

    points_per_turn: int = DEFAULT_POINTS_PER_TURN
    """测距仪转一圈共返回多少个点；角度 = 点序号 / 该值 * 360。"""

    smooth: bool = True
    """「编辑 → 平滑」的开关状态（不在设置页显示，仅随配置持久化）。"""

    def degrees_per_point(self) -> float:
        """由一圈点数推算的角度步进（度）。"""
        return 360.0 / max(1, int(self.points_per_turn))


def load_config() -> ScanConfig:
    """从 QSettings 读取配置；缺失项使用默认值。"""
    settings = QSettings(ORG_NAME, APP_NAME)
    defaults = ScanConfig()
    low, high = POINTS_RANGE

    return ScanConfig(
        simulate=_to_bool(settings.value(_KEY_SIMULATE, defaults.simulate), defaults.simulate),
        address=str(settings.value(_KEY_ADDRESS, defaults.address) or ""),
        points_per_turn=_to_int(
            settings.value(_KEY_POINTS_PER_TURN, defaults.points_per_turn),
            defaults.points_per_turn,
            low,
            high,
        ),
        smooth=_to_bool(settings.value(_KEY_SMOOTH, defaults.smooth), defaults.smooth),
    )


def save_config(config: ScanConfig) -> None:
    """把配置写入 QSettings 并立即落盘。"""
    settings = QSettings(ORG_NAME, APP_NAME)
    settings.setValue(_KEY_SIMULATE, bool(config.simulate))
    settings.setValue(_KEY_ADDRESS, config.address)
    settings.setValue(_KEY_POINTS_PER_TURN, int(config.points_per_turn))
    settings.setValue(_KEY_SMOOTH, bool(config.smooth))
    settings.sync()


# --------------------------------------------------------------------------- #
# 类型转换辅助：QSettings 在部分平台会把数值读成字符串
# --------------------------------------------------------------------------- #


def _to_int(value: object, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(str(value)))))
    except (TypeError, ValueError):
        return default


def _to_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default


# --------------------------------------------------------------------------- #
# 设置对话框
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    """「设置」对话框：运行方式 + 采集参数（单页）。"""

    def __init__(self, config: ScanConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(460)

        self._config = replace(config)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_run_box())
        layout.addWidget(self._build_params_box())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("恢复默认")
        layout.addWidget(buttons)

        self._load_into_widgets()
        self._update_estimate()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #

    def _build_run_box(self) -> QWidget:
        box = QGroupBox("运行方式", self)
        form = QFormLayout(box)

        self.simulate_check = QCheckBox("模拟运行", box)
        self.simulate_check.setToolTip(
            "勾选：不连接蓝牙、不向设备下发任何指令，由内置房间轮廓产生数据。\n"
            "不勾选：点击「绘制 → 开始」后真实通过蓝牙下发启动指令并接收每帧结果，\n"
            "此时需要先在「连接」菜单里连接测距仪。"
        )
        self.simulate_check.toggled.connect(self._on_simulate_toggled)
        form.addRow(self.simulate_check)

        self.run_hint = QLabel("", box)
        self.run_hint.setWordWrap(True)
        form.addRow("说明：", self.run_hint)
        return box

    def _build_params_box(self) -> QWidget:
        box = QGroupBox("参数", self)
        form = QFormLayout(box)

        self.points_spin = QSpinBox(box)
        self.points_spin.setRange(*POINTS_RANGE)
        self.points_spin.setSingleStep(60)
        self.points_spin.setSuffix(" 个")
        self.points_spin.setToolTip(
            "测距仪转一圈共返回多少个点（每转一个点返回一帧：点序号 + 距离）。\n"
            "角度 = 点序号 / 该值 × 360。"
        )
        self.points_spin.valueChanged.connect(self._update_estimate)
        form.addRow("测距仪转一圈共返回多少个点：", self.points_spin)

        self.estimate_label = QLabel("", box)
        self.estimate_label.setWordWrap(True)
        form.addRow("角度步进：", self.estimate_label)
        return box

    # ------------------------------------------------------------------ #
    # 数据装载 / 收集
    # ------------------------------------------------------------------ #

    def _load_into_widgets(self) -> None:
        """把当前配置写入各控件。"""
        self.simulate_check.setChecked(self._config.simulate)
        self.points_spin.setValue(self._config.points_per_turn)
        self._on_simulate_toggled(self._config.simulate)

    def config(self) -> ScanConfig:
        """由控件当前值生成配置对象。"""
        return replace(
            self._config,
            simulate=bool(self.simulate_check.isChecked()),
            points_per_turn=int(self.points_spin.value()),
        )

    def _on_simulate_toggled(self, checked: bool) -> None:
        if checked:
            self.run_hint.setText("不进行真实通信，直接使用内置房间轮廓产生的数据。")
        else:
            self.run_hint.setText("真实通过蓝牙下发启动指令并接收结果，需先在「连接」菜单连接测距仪。")

    def accept(self) -> None:  # noqa: D102 - 覆写 QDialog
        try:
            cfg = self.config()
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001 - 配置写入失败不应阻断界面
            QMessageBox.warning(self, "设置", f"配置保存失败：{exc}")
            return
        self._config = cfg
        super().accept()

    def _restore_defaults(self) -> None:
        """恢复出厂默认值（不立即写盘，点「确定」后才保存）。"""
        self._config = replace(ScanConfig(), address=self._config.address)
        self._load_into_widgets()
        self._update_estimate()

    # ------------------------------------------------------------------ #
    # 预计信息
    # ------------------------------------------------------------------ #

    def _update_estimate(self) -> None:
        points = max(1, int(self.points_spin.value()))
        self.estimate_label.setText(f"{360.0 / points:.4f}° / 点，共 {points} 帧。")
