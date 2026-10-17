"""导出模块：把 2D 平面图写成 DXF，供 CAD 打开。

当前只支持 DXF（AutoCAD R12 / AC1009 的 ASCII 形式），这是兼容性最好的一档：
所有主流 CAD（AutoCAD、中望、浩辰、LibreCAD、QCAD 等）都能直接打开。

写法上不追求花哨——整圈轮廓拆成一条条 ``LINE`` 实体，坐标单位为米，放在
``SCAN`` 图层上。R12 没有单位字段，因此导出后在 CAD 里请按「米」理解。

扩展方式：按 :func:`register_writer` 注册新的写入函数即可，主窗口无需改动。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "ExportResult",
    "export_scan",
    "register_writer",
    "supported_formats",
    "suggested_filename",
    "write_dxf",
]

logger = logging.getLogger(__name__)

#: 写入函数签名：``(points, path, **options) -> ExportResult``
WriterFunc = Callable[..., "ExportResult"]

#: 默认图层名
DEFAULT_LAYER = "SCAN"


@dataclass
class ExportResult:
    """导出结果。

    :param ok: 是否真正写出了文件。
    :param message: 给用户看的提示文案。
    :param path: 写出的文件路径，未写出时为 ``None``。
    :param point_count: 参与导出的点数。
    :param extra: 附加信息（例如直线段数）。
    """

    ok: bool
    message: str
    path: Path | None = None
    point_count: int = 0
    extra: dict[str, object] = field(default_factory=dict)


#: 已注册的格式：后缀 -> (显示名, 写入函数)
_WRITERS: dict[str, tuple[str, WriterFunc]] = {}


def register_writer(suffix: str, label: str, writer: WriterFunc) -> None:
    """注册一种导出格式。

    :param suffix: 文件后缀（不含点），例如 ``"dxf"``。
    :param label: 界面显示的格式名称。
    :param writer: 写入函数，需返回 :class:`ExportResult`。
    """
    key = suffix.lower().lstrip(".")
    _WRITERS[key] = (label, writer)
    logger.debug("已注册导出格式：%s", key)


def supported_formats() -> dict[str, str]:
    """返回当前支持的格式（后缀 -> 显示名）。"""
    return {suffix: label for suffix, (label, _) in _WRITERS.items()}


def suggested_filename(fmt: str = "dxf") -> str:
    """返回默认的导出文件名。"""
    base = "plane_scan"
    return f"{base}.{fmt.lower().lstrip('.')}" if fmt else base


# --------------------------------------------------------------------------- #
# DXF 写入
# --------------------------------------------------------------------------- #


def _num(value: float) -> str:
    """DXF 数值：定点小数，避免出现科学计数法。"""
    return f"{float(value):.6f}"


def _dxf_pairs(vertices: list[tuple[float, float]], layer: str) -> list[tuple[object, str]]:
    """按 DXF 的「组码 / 值」交替格式生成全部内容。"""
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    pairs: list[tuple[object, str]] = []

    def add(code: object, value: str) -> None:
        pairs.append((code, value))

    # -- HEADER：版本与图形范围 ---------------------------------------- #
    add(0, "SECTION")
    add(2, "HEADER")
    add(9, "$ACADVER")
    add(1, "AC1009")  # R12
    for name, x, y in (("$EXTMIN", min(xs), min(ys)), ("$EXTMAX", max(xs), max(ys))):
        add(9, name)
        add(10, _num(x))
        add(20, _num(y))
        add(30, "0.0")
    add(0, "ENDSEC")

    # -- TABLES：图层 --------------------------------------------------- #
    add(0, "SECTION")
    add(2, "TABLES")
    add(0, "TABLE")
    add(2, "LAYER")
    add(70, "1")
    add(0, "LAYER")
    add(2, layer)
    add(70, "0")
    add(62, "7")
    add(6, "CONTINUOUS")
    add(0, "ENDTAB")
    add(0, "ENDSEC")

    # -- ENTITIES：逐段直线 --------------------------------------------- #
    add(0, "SECTION")
    add(2, "ENTITIES")
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:]):
        add(0, "LINE")
        add(8, layer)
        add(10, _num(x1))
        add(20, _num(y1))
        add(30, "0.0")
        add(11, _num(x2))
        add(21, _num(y2))
        add(31, "0.0")
    add(0, "ENDSEC")

    add(0, "EOF")
    return pairs


def write_dxf(
    points: Sequence[Sequence[float]] | np.ndarray,
    path: str | Path,
    *,
    closed: bool = True,
    layer: str = DEFAULT_LAYER,
) -> ExportResult:
    """把 2D 点集写成 DXF 直线段。

    :param points: 形状 ``(N, 2)`` 的点集（米）。
    :param path: 目标文件路径。
    :param closed: 是否补一段「末点 → 首点」，把整圈轮廓闭合。
    :param layer: 图层名。
    """
    target = Path(path).expanduser()
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return ExportResult(ok=False, message=f"DXF 导出失败：点集数据非法（{exc}）")

    if arr.ndim != 2 or arr.shape[1] != 2:
        return ExportResult(ok=False, message=f"DXF 导出失败：点集形状应为 (N, 2)，实际为 {arr.shape}。")
    if arr.shape[0] < 2:
        return ExportResult(ok=False, message="DXF 导出失败：至少需要 2 个点才能连成图形。")

    arr = arr[np.isfinite(arr).all(axis=1)]
    vertices = [(float(x), float(y)) for x, y in arr]
    if closed and len(vertices) >= 3:
        vertices.append(vertices[0])

    pairs = _dxf_pairs(vertices, layer)
    try:
        # DXF R12 为纯 ASCII；行尾统一用 \n，多数 CAD 均可接受
        target.write_text("".join(f"{code}\n{value}\n" for code, value in pairs), encoding="ascii")
    except OSError as exc:
        logger.exception("DXF 写入失败：%s", exc)
        return ExportResult(ok=False, message=f"DXF 写入失败：{exc}", path=target)
    except UnicodeEncodeError as exc:
        return ExportResult(ok=False, message=f"DXF 写入失败：内容无法编码为 ASCII（{exc}）", path=target)

    segments = max(0, len(vertices) - 1)
    logger.info("已导出 DXF：%s（%d 点 / %d 段）", target, arr.shape[0], segments)
    return ExportResult(
        ok=True,
        message=(
            f"已导出 DXF：{target}\n"
            f"共 {arr.shape[0]} 个点、{segments} 条直线段（单位：米，图层 {layer}）。"
        ),
        path=target,
        point_count=int(arr.shape[0]),
        extra={"segments": segments, "layer": layer},
    )


register_writer("dxf", "DXF 图形交换文件（供 CAD 使用）", write_dxf)


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #


def export_scan(
    points: Sequence[Sequence[float]] | np.ndarray,
    path: str | Path,
    fmt: str | None = None,
    **options: object,
) -> ExportResult:
    """导出当前 2D 图形。

    :param points: 形状 ``(N, 2)`` 的点集（米）。
    :param path: 用户选择的保存路径。
    :param fmt: 目标格式后缀；为 ``None`` 时按 ``path`` 的后缀推断。
    :param options: 传给具体写入函数的选项（例如 ``closed=False``）。
    """
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        logger.warning("导出失败：点集无法解析（%s）", exc)
        return ExportResult(ok=False, message=f"导出失败：点集数据非法（{exc}）")

    if arr.size == 0:
        return ExportResult(ok=False, message="当前没有扫描数据，无法导出。")

    target = Path(path).expanduser() if str(path).strip() else None
    suffix = (fmt or (target.suffix.lstrip(".") if target else "")).lower()

    if suffix not in _WRITERS:
        supported = "、".join(sorted(_WRITERS)) or "（无）"
        return ExportResult(
            ok=False,
            message=f"导出失败：不支持的格式「{suffix}」。当前支持：{supported}。",
        )
    if target is None:
        return ExportResult(ok=False, message="导出失败：未指定保存路径。")

    label, writer = _WRITERS[suffix]
    try:
        result = writer(arr, target, **options)
    except Exception as exc:  # noqa: BLE001 - 写入失败不应让界面崩溃
        logger.exception("导出写入失败：%s", exc)
        return ExportResult(ok=False, message=f"导出失败：{exc}", path=target)

    if result.ok:
        logger.info("已导出 %d 个点到 %s（%s）", arr.shape[0], target, label)
    return result
