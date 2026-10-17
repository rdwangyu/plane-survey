"""「连接」与「关于」两个对话框。

* :class:`ConnectDialog` —— 扫描蓝牙测距仪并建立连接，成功后可取走设备对象。
* :class:`AboutDialog`   —— 显示软件版本，并通过蓝牙查询测距仪信息。

两个对话框把耗时的蓝牙操作（扫描、连接、查询）都放在后台线程里，界面不会卡死。
"""

from __future__ import annotations

import platform

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from device import BaseDevice, BleDevice, BleScanner, DeviceError
from version import APP_SUBTITLE, APP_TITLE, APP_VERSION

__all__ = ["ConnectDialog", "AboutDialog"]

#: 扫描蓝牙设备的默认时长（秒）
SCAN_TIMEOUT = 6.0


# --------------------------------------------------------------------------- #
# 后台连接线程
# --------------------------------------------------------------------------- #


class _ConnectThread(QThread):
    """在后台线程连接蓝牙测距仪，避免界面卡死。"""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, address: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.address = address

    def run(self) -> None:  # noqa: D102 - QThread 入口
        device: BleDevice | None = None
        try:
            device = BleDevice(self.address)
            device.connect()
        except DeviceError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程内异常导致崩溃
            self.failed.emit(f"连接失败：{exc}")
        else:
            self.succeeded.emit(device)
            return

        if device is not None:
            try:
                device.disconnect()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 「连接」
# --------------------------------------------------------------------------- #


class ConnectDialog(QDialog):
    """扫描并连接蓝牙测距仪。

    打开即开始扫描；选中设备后点「连接」（或双击设备）建立连接。连接成功后
    :meth:`device` 返回一个已连接的 :class:`BleDevice`，由调用方接管生命周期。
    """

    def __init__(self, parent: QWidget | None = None, *, simulate: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("连接蓝牙测距仪")
        self.setMinimumWidth(460)

        self._scanner: BleScanner | None = None
        self._connect_thread: _ConnectThread | None = None
        self._device: BleDevice | None = None

        layout = QVBoxLayout(self)

        self.hint = QLabel("正在扫描附近的蓝牙设备…", self)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.device_list = QListWidget(self)
        self.device_list.itemDoubleClicked.connect(lambda _item: self._start_connect())
        self.device_list.currentItemChanged.connect(lambda *_: self._update_buttons())
        layout.addWidget(self.device_list)

        row = QHBoxLayout()
        self.rescan_button = QPushButton("重新扫描", self)
        self.rescan_button.clicked.connect(self._start_scan)
        self.connect_button = QPushButton("连接", self)
        self.connect_button.setDefault(True)
        self.connect_button.clicked.connect(self._start_connect)
        row.addWidget(self.rescan_button)
        row.addWidget(self.connect_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        if simulate:
            self.hint.setText(
                "提示：当前「运行方式」为模拟运行，连接真实设备后不会被使用。\n"
                "正在扫描附近的蓝牙设备…"
            )

        self._update_buttons()
        self._start_scan()

    # ------------------------------------------------------------------ #
    # 结果
    # ------------------------------------------------------------------ #

    def device(self) -> BleDevice | None:
        """已连接的蓝牙设备；未连接时为 ``None``。"""
        return self._device

    # ------------------------------------------------------------------ #
    # 扫描
    # ------------------------------------------------------------------ #

    def _start_scan(self) -> None:
        if self._scanner is not None and self._scanner.isRunning():
            return

        self.device_list.clear()
        self.hint.setText("正在扫描附近的蓝牙设备…")
        self.rescan_button.setEnabled(False)
        self._update_buttons()

        scanner = BleScanner(timeout=SCAN_TIMEOUT, parent=self)
        scanner.device_found.connect(self._on_device_found)
        scanner.scan_finished.connect(self._on_scan_finished)
        scanner.scan_failed.connect(self._on_scan_failed)
        scanner.finished.connect(self._on_scanner_stopped)
        self._scanner = scanner
        scanner.start()

    def _on_device_found(self, address: str, name: str) -> None:
        item = QListWidgetItem(f"{name}    [{address}]")
        item.setData(Qt.ItemDataRole.UserRole, address)
        self.device_list.addItem(item)
        if self.device_list.count() == 1:
            self.device_list.setCurrentRow(0)

    def _on_scan_finished(self, count: int) -> None:
        if count == 0:
            self.hint.setText(
                "未发现蓝牙设备。请确认测距仪已开机、在有效范围内，"
                "且系统的蓝牙适配器可用，然后点「重新扫描」。"
            )
        else:
            self.hint.setText(f"发现 {count} 个设备，选中后点「连接」。")

    def _on_scan_failed(self, message: str) -> None:
        self.hint.setText(message)

    def _on_scanner_stopped(self) -> None:
        self.rescan_button.setEnabled(True)
        self._scanner = None
        self._update_buttons()

    # ------------------------------------------------------------------ #
    # 连接
    # ------------------------------------------------------------------ #

    def _selected_address(self) -> str:
        item = self.device_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _start_connect(self) -> None:
        if self._connect_thread is not None and self._connect_thread.isRunning():
            return

        address = self._selected_address()
        if not address:
            self.hint.setText("请先在列表中选择一台测距仪。")
            return

        self.hint.setText(f"正在连接 {address} …（最长等待 15 秒）")
        self.rescan_button.setEnabled(False)
        self._update_buttons()

        thread = _ConnectThread(address, self)
        thread.succeeded.connect(self._on_connected)
        thread.failed.connect(self._on_connect_failed)
        thread.finished.connect(self._on_connect_finished)
        self._connect_thread = thread
        thread.start()

    def _on_connected(self, device: object) -> None:
        self._device = device  # type: ignore[assignment]
        self.hint.setText(f"连接成功：{self._device.address if self._device else ''}")
        self.accept()

    def _on_connect_failed(self, message: str) -> None:
        self.hint.setText(f"连接失败：{message}")

    def _on_connect_finished(self) -> None:
        self._connect_thread = None
        self.rescan_button.setEnabled(True)
        self._update_buttons()

    def _update_buttons(self) -> None:
        connecting = self._connect_thread is not None and self._connect_thread.isRunning()
        self.connect_button.setEnabled(bool(self._selected_address()) and not connecting)

    # ------------------------------------------------------------------ #
    # 关闭清理
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QDialog
        if self._connect_thread is not None and self._connect_thread.isRunning():
            # 连接线程无法强制终止，此时关闭会让信号发往已销毁的对象
            self.hint.setText("正在连接，请稍候…")
            event.ignore()  # type: ignore[attr-defined]
            return

        if self._scanner is not None and self._scanner.isRunning():
            self._scanner.cancel()
            self._scanner.wait(3000)
        super().closeEvent(event)  # type: ignore[arg-type]

    def done(self, result: int) -> None:  # noqa: D102 - 覆写 QDialog
        """对话框未成功连接就关闭时，释放半途建立的连接。"""
        if self._device is not None and result != QDialog.DialogCode.Accepted:
            try:
                self._device.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._device = None
        super().done(result)


# --------------------------------------------------------------------------- #
# 后台查询线程
# --------------------------------------------------------------------------- #


class _InfoThread(QThread):
    """在后台线程查询测距仪信息。"""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, device: BaseDevice, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device = device

    def run(self) -> None:  # noqa: D102 - QThread 入口
        try:
            info = self.device.query_info()
        except DeviceError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 兜底
            self.failed.emit(f"查询失败：{exc}")
            return
        if not info:
            self.failed.emit("测距仪未返回任何信息。")
            return
        self.succeeded.emit(dict(info))


# --------------------------------------------------------------------------- #
# 「关于」
# --------------------------------------------------------------------------- #


class AboutDialog(QDialog):
    """显示软件版本与测距仪信息。

    打开时自动向已连接的测距仪发送查询指令并展示响应；也可以点「重新查询」。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        device: BaseDevice | None = None,
        simulate: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setMinimumWidth(460)

        self._device = device
        self._info_thread: _InfoThread | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_software_box())
        layout.addWidget(self._build_device_box())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_buttons()
        self._query()  # 打开即查询

    # ------------------------------------------------------------------ #
    # 界面
    # ------------------------------------------------------------------ #

    def _build_software_box(self) -> QWidget:
        box = QGroupBox("软件信息", self)
        form = QFormLayout(box)
        form.addRow("名称：", QLabel(APP_TITLE, box))
        form.addRow("版本：", QLabel(APP_VERSION, box))
        form.addRow("说明：", QLabel(APP_SUBTITLE, box))
        form.addRow(
            "运行环境：",
            QLabel(
                f"Python {platform.python_version()} · "
                f"PyQt {_qt_version()} · {platform.system()}",
                box,
            ),
        )
        return box

    def _build_device_box(self) -> QWidget:
        box = QGroupBox("测距仪信息", self)
        layout = QVBoxLayout(box)

        if self._device is None:
            self.device_label = QLabel("未连接测距仪（可在「连接」菜单中扫描并连接）", box)
        else:
            self.device_label = QLabel(f"已连接：{self._device.name}", box)
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)

        self.info_view = QPlainTextEdit(box)
        self.info_view.setReadOnly(True)
        self.info_view.setMinimumHeight(140)
        self.info_view.setPlaceholderText("设备信息将在查询后显示…")
        layout.addWidget(self.info_view)

        row = QHBoxLayout()
        self.query_button = QPushButton("重新查询", box)
        self.query_button.clicked.connect(self._query)
        row.addWidget(self.query_button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def _query(self) -> None:
        if self._device is None:
            self.info_view.setPlainText("未连接测距仪，无法通过蓝牙查询设备信息。")
            return
        if self._info_thread is not None and self._info_thread.isRunning():
            return

        self.info_view.setPlainText("正在通过蓝牙发送查询指令…")
        self._update_buttons()

        thread = _InfoThread(self._device, self)
        thread.succeeded.connect(self._on_info_ready)
        thread.failed.connect(self._on_info_failed)
        thread.finished.connect(self._on_query_finished)
        self._info_thread = thread
        thread.start()

    def _on_info_ready(self, info: dict) -> None:
        self.info_view.setPlainText(
            "\n".join(f"{key}：{value}" for key, value in info.items())
        )

    def _on_info_failed(self, message: str) -> None:
        self.info_view.setPlainText(f"查询失败：{message}")

    def _on_query_finished(self) -> None:
        self._info_thread = None
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self._info_thread is not None and self._info_thread.isRunning()
        self.query_button.setEnabled(self._device is not None and not running)

    def closeEvent(self, event: object) -> None:  # noqa: D102 - 覆写 QDialog
        if self._info_thread is not None and self._info_thread.isRunning():
            self._info_thread.wait(3000)
        super().closeEvent(event)  # type: ignore[arg-type]


def _qt_version() -> str:
    """当前 PyQt6 / Qt 版本。"""
    try:
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

        return f"{PYQT_VERSION_STR} (Qt {QT_VERSION_STR})"
    except Exception:  # noqa: BLE001 - 仅用于展示
        return "未知"
