"""设置对话框与配置持久化。

配置通过 :class:`PyQt6.QtCore.QSettings` 保存到系统标准位置（Linux 下通常是
``~/.config/EnvironmentMeasurement/MeasureApp.conf``），主窗口与设置对话框共用
:func:`load_config` / :func:`save_config` 两个函数。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from device import MODE_AUTO, MODE_BLE, MODE_MOCK, BleDevice, BleScanner, DeviceError

__all__ = ["ScanConfig", "SettingsDialog", "load_config", "save_config"]

#: QSettings 组织名 / 应用名
ORG_NAME = "EnvironmentMeasurement"
APP_NAME = "MeasureApp"

# -- QSettings 键名 --------------------------------------------------------- #
_KEY_ADDRESS = "bluetooth/address"
_KEY_MODE = "device/mode"
_KEY_TURNS = "scan/turns"
_KEY_SMOOTH = "scan/smooth"
_KEY_SPEED = "scan/speed_rpm"
_KEY_INTERVAL = "scan/sample_interval"
_KEY_SCAN_TIMEOUT = "bluetooth/scan_timeout"
_KEY_SIMULATE = "device/simulate"

#: 设备模式取值 -> 界面显示文本
MODE_LABELS: tuple[tuple[str, str], ...] = (
    (MODE_AUTO, "自动（优先蓝牙，失败回退模拟）"),
    (MODE_MOCK, "仅模拟设备（无需硬件）"),
    (MODE_BLE, "仅蓝牙设备"),
)


@dataclass
class ScanConfig:
    """一次扫描所需的全部参数。"""

    mode: str = MODE_AUTO
    """设备模式，取值见 :data:`MODE_LABELS`。"""

    address: str = ""
    """蓝牙设备地址。"""

    turns: int = 1
    """测量圈数，角度范围为 ``0 ~ 360 * turns``。"""

    smooth: bool = True
    """是否对距离序列做平滑与分段直线拟合。"""

    speed_rpm: float = 12.0
    """电机转速（转/分钟）。"""

    sample_interval: float = 0.05
    """采样间隔（秒）。"""

    scan_timeout: float = 5.0
    """蓝牙扫描时长（秒）。"""

    simulate: bool = False
    """模拟运行：不进行任何真实通信，仅返回模拟值（用于测试与演示）。"""

    def degrees_per_sample(self) -> float:
        """由转速与采样间隔推算的角度步进（度）。"""
        return max(0.1, self.speed_rpm * 6.0 * self.sample_interval)

    def sample_count(self) -> int:
        """预计采样点数（不含终点，避免首尾重复）。"""
        return max(1, int(round(360.0 * self.turns / self.degrees_per_sample())))

    def estimated_seconds(self) -> float:
        """理论旋转耗时（秒），不含蓝牙通讯开销。"""
        return self.turns * 60.0 / max(0.1, self.speed_rpm)


def load_config() -> ScanConfig:
    """从 QSettings 读取配置；缺失项使用默认值。"""
    settings = QSettings(ORG_NAME, APP_NAME)
    defaults = ScanConfig()

    mode = str(settings.value(_KEY_MODE, defaults.mode))
    if mode not in {value for value, _ in MODE_LABELS}:
        mode = defaults.mode

    return ScanConfig(
        mode=mode,
        address=str(settings.value(_KEY_ADDRESS, defaults.address) or ""),
        turns=_to_int(settings.value(_KEY_TURNS, defaults.turns), defaults.turns, 1, 100),
        smooth=_to_bool(settings.value(_KEY_SMOOTH, defaults.smooth), defaults.smooth),
        speed_rpm=_to_float(
            settings.value(_KEY_SPEED, defaults.speed_rpm), defaults.speed_rpm, 0.5, 120.0
        ),
        sample_interval=_to_float(
            settings.value(_KEY_INTERVAL, defaults.sample_interval),
            defaults.sample_interval,
            0.01,
            5.0,
        ),
        scan_timeout=_to_float(
            settings.value(_KEY_SCAN_TIMEOUT, defaults.scan_timeout), defaults.scan_timeout, 1.0, 30.0
        ),
        simulate=_to_bool(settings.value(_KEY_SIMULATE, defaults.simulate), defaults.simulate),
    )


def save_config(config: ScanConfig) -> None:
    """把配置写入 QSettings 并立即落盘。"""
    settings = QSettings(ORG_NAME, APP_NAME)
    settings.setValue(_KEY_MODE, config.mode)
    settings.setValue(_KEY_ADDRESS, config.address)
    settings.setValue(_KEY_TURNS, int(config.turns))
    settings.setValue(_KEY_SMOOTH, bool(config.smooth))
    settings.setValue(_KEY_SPEED, float(config.speed_rpm))
    settings.setValue(_KEY_INTERVAL, float(config.sample_interval))
    settings.setValue(_KEY_SCAN_TIMEOUT, float(config.scan_timeout))
    settings.setValue(_KEY_SIMULATE, bool(config.simulate))
    settings.sync()


# --------------------------------------------------------------------------- #
# 类型转换辅助：QSettings 在部分平台会把数值读成字符串
# --------------------------------------------------------------------------- #


def _to_int(value: object, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(str(value)))))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(str(value))))
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
# 连接测试线程
# --------------------------------------------------------------------------- #


class _ConnectTestThread(QThread):
    """在后台线程尝试连接蓝牙设备，避免界面卡死。"""

    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, address: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.address = address

    def run(self) -> None:  # noqa: D102 - QThread 入口
        device: BleDevice | None = None
        try:
            device = BleDevice(self.address, connect_timeout=10.0)
            device.connect()
            self.succeeded.emit(f"连接成功：{self.address}")
        except DeviceError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程内异常导致崩溃
            self.failed.emit(f"连接失败：{exc}")
        finally:
            if device is not None:
                try:
                    device.disconnect()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- #
# 设置对话框
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    """「设置」对话框：蓝牙连接与扫描参数。"""

    def __init__(self, config: ScanConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(460)

        self._config = replace(config)
        self._scanner: BleScanner | None = None
        self._connect_test: _ConnectTestThread | None = None

        tabs = QTabWidget(self)
        tabs.addTab(self._build_bluetooth_tab(), "蓝牙")
        tabs.addTab(self._build_params_tab(), "参数")

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

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        self._load_into_widgets()
        self._update_estimate()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #

    def _build_bluetooth_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        # -- 设备模式 ------------------------------------------------------ #
        mode_box = QGroupBox("设备模式", page)
        mode_form = QFormLayout(mode_box)
        self.mode_combo = QComboBox(mode_box)
        for value, label in MODE_LABELS:
            self.mode_combo.addItem(label, value)
        mode_form.addRow("连接方式：", self.mode_combo)

        self.simulate_check = QCheckBox("模拟运行（不进行真实通信，仅返回模拟数据）", mode_box)
        self.simulate_check.setToolTip(
            "勾选后点击「绘制」，不会连接蓝牙、也不会向设备下发任何指令，\n"
            "电机与测距全部由模拟数据代替，用于代码测试与效果展示。\n"
            "勾选期间「连接方式」与蓝牙相关设置不生效。"
        )
        self.simulate_check.toggled.connect(self._on_simulate_toggled)
        mode_form.addRow("运行方式：", self.simulate_check)
        layout.addWidget(mode_box)

        # -- 扫描与选择 ---------------------------------------------------- #
        scan_box = QGroupBox("蓝牙设备", page)
        scan_form = QFormLayout(scan_box)

        scan_row = QHBoxLayout()
        self.scan_button = QPushButton("扫描设备", scan_box)
        self.scan_button.clicked.connect(self._start_scan)
        self.scan_timeout_spin = QDoubleSpinBox(scan_box)
        self.scan_timeout_spin.setRange(1.0, 30.0)
        self.scan_timeout_spin.setSingleStep(1.0)
        self.scan_timeout_spin.setSuffix(" 秒")
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(QLabel("扫描时长：", scan_box))
        scan_row.addWidget(self.scan_timeout_spin)
        scan_row.addStretch(1)
        scan_form.addRow(scan_row)

        self.device_combo = QComboBox(scan_box)
        self.device_combo.setEditable(False)
        self.device_combo.addItem("（尚未扫描）", "")
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        scan_form.addRow("扫描结果：", self.device_combo)

        self.address_edit = QLineEdit(scan_box)
        self.address_edit.setPlaceholderText("例如 AA:BB:CC:DD:EE:FF（可手动填写）")
        scan_form.addRow("设备地址：", self.address_edit)

        test_row = QHBoxLayout()
        self.test_button = QPushButton("连接测试", scan_box)
        self.test_button.clicked.connect(self._start_connect_test)
        test_row.addWidget(self.test_button)
        test_row.addStretch(1)
        scan_form.addRow(test_row)

        self.ble_status = QLabel("提示：协议未确定时，请使用「自动」或「仅模拟设备」模式。", scan_box)
        self.ble_status.setWordWrap(True)
        scan_form.addRow(self.ble_status)

        layout.addWidget(scan_box)
        self._ble_group = scan_box  # 模拟运行时整组禁用
        layout.addStretch(1)
        return page

    def _build_params_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        box = QGroupBox("扫描参数", page)
        form = QFormLayout(box)

        self.turns_spin = QSpinBox(box)
        self.turns_spin.setRange(1, 100)
        self.turns_spin.setSuffix(" 圈")
        self.turns_spin.valueChanged.connect(self._update_estimate)
        form.addRow("测几圈：", self.turns_spin)

        self.smooth_check = QCheckBox("开启（中值/均值滤波 + 分段直线拟合）", box)
        self.smooth_check.setChecked(True)
        self.smooth_check.toggled.connect(self._update_estimate)
        form.addRow("数据平滑：", self.smooth_check)

        self.speed_spin = QDoubleSpinBox(box)
        self.speed_spin.setRange(0.5, 120.0)
        self.speed_spin.setSingleStep(1.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSuffix(" RPM")
        self.speed_spin.valueChanged.connect(self._update_estimate)
        form.addRow("电机速度：", self.speed_spin)

        self.interval_spin = QDoubleSpinBox(box)
        self.interval_spin.setRange(0.01, 5.0)
        self.interval_spin.setSingleStep(0.01)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setSuffix(" 秒")
        self.interval_spin.valueChanged.connect(self._update_estimate)
        form.addRow("采样间隔：", self.interval_spin)

        self.estimate_label = QLabel("", box)
        self.estimate_label.setWordWrap(True)
        form.addRow("预计：", self.estimate_label)

        layout.addWidget(box)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    # 数据装载 / 收集
    # ------------------------------------------------------------------ #

    def _load_into_widgets(self) -> None:
        """把当前配置写入各控件。"""
        cfg = self._config

        index = self.mode_combo.findData(cfg.mode)
        self.mode_combo.setCurrentIndex(max(0, index))

        self.address_edit.setText(cfg.address)
        self.scan_timeout_spin.setValue(cfg.scan_timeout)
        if cfg.address:
            self.device_combo.addItem(f"已保存：{cfg.address}", cfg.address)
            self.device_combo.setCurrentIndex(self.device_combo.count() - 1)

        self.simulate_check.setChecked(cfg.simulate)
        self._on_simulate_toggled(cfg.simulate)

        self.turns_spin.setValue(cfg.turns)
        self.smooth_check.setChecked(cfg.smooth)
        self.speed_spin.setValue(cfg.speed_rpm)
        self.interval_spin.setValue(cfg.sample_interval)

    def config(self) -> ScanConfig:
        """由控件当前值生成配置对象。"""
        return ScanConfig(
            mode=str(self.mode_combo.currentData() or MODE_AUTO),
            address=self.address_edit.text().strip(),
            turns=int(self.turns_spin.value()),
            smooth=bool(self.smooth_check.isChecked()),
            speed_rpm=float(self.speed_spin.value()),
            sample_interval=float(self.interval_spin.value()),
            scan_timeout=float(self.scan_timeout_spin.value()),
            simulate=bool(self.simulate_check.isChecked()),
        )

    def _on_simulate_toggled(self, checked: bool) -> None:
        """模拟运行时禁用蓝牙相关控件——它们此时不生效，留着可点会让人误解。"""
        self.mode_combo.setEnabled(not checked)
        self._ble_group.setEnabled(not checked)
        if checked:
            self.ble_status.setText("模拟运行已开启：不会连接任何真实设备，直接返回模拟数据。")

    def accept(self) -> None:  # noqa: D102 - 覆写 QDialog
        cfg = self.config()
        # 模拟运行不需要真实地址，此时不做地址校验
        if cfg.mode == MODE_BLE and not cfg.address and not cfg.simulate:
            QMessageBox.warning(self, "设置", "「仅蓝牙设备」模式需要先填写或选择设备地址。")
            return
        try:
            save_config(cfg)
        except Exception as exc:  # noqa: BLE001 - 配置写入失败不应阻断界面
            QMessageBox.warning(self, "设置", f"配置保存失败：{exc}")
        self._config = cfg
        super().accept()

    def _restore_defaults(self) -> None:
        """恢复出厂默认值（不立即写盘，点「确定」后才保存）。"""
        defaults = ScanConfig()
        self._config = defaults
        self._load_into_widgets()
        self._update_estimate()

    # ------------------------------------------------------------------ #
    # 蓝牙扫描
    # ------------------------------------------------------------------ #

    def _start_scan(self) -> None:
        if self._scanner is not None and self._scanner.isRunning():
            self._scanner.cancel()
            return

        self.device_combo.clear()
        self.device_combo.addItem("扫描中…", "")
        self.scan_button.setText("停止扫描")
        self.ble_status.setText("正在扫描附近的蓝牙设备…")

        scanner = BleScanner(timeout=float(self.scan_timeout_spin.value()), parent=self)
        scanner.device_found.connect(self._on_device_found)
        scanner.scan_finished.connect(self._on_scan_finished)
        scanner.scan_failed.connect(self._on_scan_failed)
        scanner.finished.connect(self._on_scanner_stopped)
        self._scanner = scanner
        scanner.start()

    def _on_device_found(self, address: str, name: str) -> None:
        if self.device_combo.count() == 1 and self.device_combo.itemData(0) == "":
            self.device_combo.clear()  # 清掉「扫描中…」占位项
        self.device_combo.addItem(f"{name}  [{address}]", address)

    def _on_scan_finished(self, count: int) -> None:
        if count == 0:
            self.device_combo.clear()
            self.device_combo.addItem("（未发现设备）", "")
        self.ble_status.setText(f"扫描完成，共发现 {count} 个设备。")

    def _on_scan_failed(self, message: str) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("（扫描失败）", "")
        self.ble_status.setText(message)

    def _on_scanner_stopped(self) -> None:
        self.scan_button.setText("扫描设备")
        self._scanner = None

    def _on_device_selected(self, index: int) -> None:
        address = self.device_combo.itemData(index) if index >= 0 else None
        if address:
            self.address_edit.setText(str(address))

    # ------------------------------------------------------------------ #
    # 连接测试
    # ------------------------------------------------------------------ #

    def _start_connect_test(self) -> None:
        if self._connect_test is not None and self._connect_test.isRunning():
            return

        address = self.address_edit.text().strip()
        if not address:
            QMessageBox.information(self, "连接测试", "请先扫描或手动填写设备地址。")
            return

        self.test_button.setEnabled(False)
        self.ble_status.setText(f"正在连接 {address} …")

        worker = _ConnectTestThread(address, self)
        worker.succeeded.connect(self._on_connect_ok)
        worker.failed.connect(self._on_connect_failed)
        worker.finished.connect(self._on_connect_done)
        self._connect_test = worker
        worker.start()

    def _on_connect_ok(self, message: str) -> None:
        self.ble_status.setText(message)

    def _on_connect_failed(self, message: str) -> None:
        self.ble_status.setText(f"连接失败：{message}")

    def _on_connect_done(self) -> None:
        self.test_button.setEnabled(True)
        self._connect_test = None

    # ------------------------------------------------------------------ #
    # 预计信息
    # ------------------------------------------------------------------ #

    def _update_estimate(self) -> None:
        cfg = self.config()
        self.estimate_label.setText(
            f"角度步进约 {cfg.degrees_per_sample():.2f}°，"
            f"预计 {cfg.sample_count()} 个采样点，"
            f"理论旋转耗时约 {cfg.estimated_seconds():.1f} 秒。"
        )

    # ------------------------------------------------------------------ #
    # 关闭清理
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QDialog
        if self._scanner is not None and self._scanner.isRunning():
            self._scanner.cancel()
            self._scanner.wait(2000)
        if self._connect_test is not None and self._connect_test.isRunning():
            self._connect_test.wait(2000)
        super().closeEvent(event)  # type: ignore[arg-type]
