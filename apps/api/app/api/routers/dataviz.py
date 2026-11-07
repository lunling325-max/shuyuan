"""数据可视化路由：CSV -> 统计表 / SVG 图 / 源码导出 / 抗幻觉图注。

为什么 CSV 不经过 `deps.registry`：registry 是"论文库"，CSV 是一次性输入，
两者生命周期完全不同。把 CSV 塞进论文库会让列表接口多出一堆非 PDF 条目。
这里也**不落盘**：请求内解析、请求结束即释放。无状态实现更省事，也没有
临时文件清理和越权读取的问题；代价是同一份 CSV 每次都要重新解析，对大文件
（几 MB）来说这点 CPU 完全可以接受。

图注的抗幻觉设计（本模块最核心的部分）：

1. 先用 `csv_table.describe` 对**实际绘制的清洗后数据**算统计卡；
2. 模型只准基于统计卡写 caption，并被要求返回 key_numbers；
3. 服务端用正则抽出 caption 里的每个数字，逐个与统计卡比对
   （支持 12.3 vs 12.30、0.5 vs 50% 等等价写法）；
4. 少量编造数字 -> 原位替换成 `[未核实]` 并记入 notes；
   大量编造 -> 整段作废，退化成纯模板图注；
5. 最后服务端追加一行"数据区间：n=…"，保证图注里至少有确定的数字。

错误码：非 CSV / 解码失败 / 结构错误 -> 400；空表、无数值列、图表参数
不成立 -> 422；超过 5 MB -> 413。模型不可用绝不 500，而是 200 + degraded。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.analysis.chart import ChartData, ChartError, build_chart_data, render_chart
from app.analysis.csv_table import (
    CsvDecodeError,
    CsvEmptyError,
    CsvError,
    CsvFormatError,
    NoNumericColumnError,
    SeriesStats,
    Table,
    describe,
    fmt_number,
    parse_csv,
    pearson,
)
from app.analysis.export import to_matplotlib_script, to_mermaid, to_tikz

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_PREVIEW_ROWS = 100
MAX_RESPONSE_ROWS = 200

# CSV 没有魔数，但常见"其实不是 CSV"的文件都有稳定文件头。命中就 400，
# 比让 pymupdf/zip 的二进制内容一路冒到 CSV 解析器里更容易解释。
_BINARY_MAGIC = (
    b"%PDF-",
    b"PK\x03\x04",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"RIFF",
    b"\x1f\x8b",
)

# 图注里的数字 token。前面的 lookbehind 防止把日期 "2023-01" 里的 "-01"
# 当成负一号；日期本身会被拆成 2023 和 01，而统计卡的 labels 里同样会
# 抽出这两个数，所以正常的日期描述不会被误杀。
_NUM_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?"
)


router = APIRouter()


# ================================================================ 响应模型


class ColumnOut(BaseModel):
    name: str
    kind: str = Field(description="number | date | category")
    unit: str = ""
    missing: int = 0


class SeriesStatsOut(BaseModel):
    name: str
    unit: str = ""
    n: int
    missing: int = 0
    min: float
    max: float
    mean: float
    median: float
    std: float
    slope: float
    first: float
    last: float
    min_index: int
    max_index: int
    min_label: str
    max_label: str


class TableOut(BaseModel):
    columns: list[ColumnOut]
    rows: list[dict[str, Any]]
    stats: dict[str, SeriesStatsOut]
    total_rows: int
    truncated: bool = False


class ChartOut(BaseModel):
    kind: str
    svg: str
    width: int
    height: int
    legend: list[str] = Field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    note: str = ""


class KeyNumberOut(BaseModel):
    value: float
    meaning: str = ""


class CaptionOut(BaseModel):
    text: str
    key_numbers: list[KeyNumberOut] = Field(default_factory=list)
    trusted: bool = True
    notes: list[str] = Field(default_factory=list)


class ExportsOut(BaseModel):
    mermaid: str
    tikz: str
    matplotlib: str


class DatavizOut(BaseModel):
    table: TableOut
    chart: ChartOut
    caption: CaptionOut | None = None
    exports: ExportsOut
    degraded: bool = False
    notes: list[str] = Field(default_factory=list)


class PreviewOut(BaseModel):
    table: TableOut
    notes: list[str] = Field(default_factory=list)


# ================================================================ 上传读取


def _read_csv_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_CSV_BYTES + 1)
    if len(data) > MAX_CSV_BYTES:
        raise HTTPException(413, "CSV 文件超过 5 MB，请先抽样或拆分后再上传。")
    # 400 而不是 422：这一条的兄弟（二进制文件头、非 UTF-8、列数不齐）都返回 400，
    # 只有它是 422，前端按状态码分类提示时会被拆成两种"同一种错"。
    if not data:
        raise HTTPException(400, "上传的 CSV 是空的。")
    for magic in _BINARY_MAGIC:
        if data.startswith(magic):
            raise HTTPException(400, "上传的文件不是 CSV（检测到二进制文件头）。")
    if b"\x00" in data[:4096]:
        raise HTTPException(400, "上传的文件不是 CSV（包含 NUL 字节）。")
    return data


def _parse_upload(data: bytes) -> Table:
    try:
        return parse_csv(data)
    except (CsvDecodeError, CsvFormatError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except CsvEmptyError as exc:
        raise HTTPException(422, str(exc)) from exc
    except NoNumericColumnError as exc:
        raise HTTPException(422, str(exc)) from exc
    except CsvError as exc:  # 兜底：任何解析问题都不该变成 500
        raise HTTPException(400, str(exc)) from exc


def _build_chart(
    table: Table, kind: str | None, x: str | None, y: str | None
) -> ChartData:
    try:
        return build_chart_data(table, kind=kind, x=x, y=y)
    except ChartError as exc:
        raise HTTPException(422, str(exc)) from exc


def _table_out(table: Table, *, max_rows: int | None = None) -> TableOut:
    limit = len(table.rows) if max_rows is None else min(max_rows, len(table.rows))
    rows: list[dict[str, Any]] = []
    for row_index in range(limit):
        record: dict[str, Any] = {}
        for column in table.columns:
            value = column.values[row_index]
            if isinstance(value, float):
                value = round(value, 6)
            record[column.name] = value
        rows.append(record)
    return TableOut(
        columns=[
            ColumnOut(
                name=column.name,
                kind=column.kind,
                unit=column.unit,
                missing=column.missing,
            )
            for column in table.columns
        ],
        rows=rows,
        stats={
            name: SeriesStatsOut(**stats.to_dict())
            for name, stats in table.stats.items()
        },
        total_rows=len(table.rows),
        truncated=limit < len(table.rows),
    )


def _dedupe(notes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for note in notes:
        text = str(note).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


# ============================================================ 图注：统计卡


def _card_entries(chart: ChartData) -> list[SeriesStats]:
    """按"实际画出来的数据"生成统计卡。

    雷达图的 `series` 是类别、`labels` 是指标，所以按轴重新聚合；其余图
    直接对每个序列做统计。这样图注里的数字与 SVG 保持一致，不会出现
    "统计的是原始 CSV、画的是剔除缺值后的行"这种错位。
    """
    if chart.kind == "radar":
        entries: list[SeriesStats] = []
        for axis_index, axis_name in enumerate(chart.labels):
            values = [item.values[axis_index] for item in chart.series]
            labels = [item.name for item in chart.series]
            entries.append(
                describe(values, name=axis_name, unit="", labels=labels)
            )
        return entries
    return [
        describe(item.values, name=item.name, unit=item.unit, labels=chart.labels)
        for item in chart.series
    ]


def _stats_card(chart: ChartData, entries: Sequence[SeriesStats]) -> dict[str, Any]:
    series_payload: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry.to_dict()
        payload["range"] = round(entry.max - entry.min, 6)
        payload["change_pct"] = (
            round((entry.last - entry.first) / abs(entry.first) * 100, 6)
            if entry.first
            else 0.0
        )
        payload["max_min_ratio"] = (
            round(entry.max / entry.min, 6) if entry.min > 0 else 0.0
        )
        series_payload.append(payload)

    correlations: list[dict[str, Any]] = []
    if chart.kind == "radar":
        axis_count = len(chart.labels)
        for left in range(axis_count):
            for right in range(left + 1, axis_count):
                correlations.append(
                    {
                        "a": chart.labels[left],
                        "b": chart.labels[right],
                        "r": pearson(
                            [item.values[left] for item in chart.series],
                            [item.values[right] for item in chart.series],
                        ),
                    }
                )
    else:
        for left in range(len(chart.series)):
            for right in range(left + 1, len(chart.series)):
                correlations.append(
                    {
                        "a": chart.series[left].name,
                        "b": chart.series[right].name,
                        "r": pearson(
                            chart.series[left].values,
                            chart.series[right].values,
                        ),
                    }
                )
        if chart.kind == "scatter" and chart.x_values:
            for item in chart.series:
                correlations.append(
                    {
                        "a": chart.x_label,
                        "b": item.name,
                        "r": pearson(chart.x_values, item.values),
                    }
                )

    return {
        "figure": 1,
        "kind": chart.kind,
        "n": len(chart.labels),
        "x": chart.x_label,
        "y": chart.y_label,
        "series": series_payload,
        "correlations": correlations,
        "labels": chart.labels[:50],
        "small_multiples": chart.small_multiples,
    }


def _token_value(token: str) -> tuple[float, bool, str] | None:
    has_percent = token.endswith("%")
    raw = token[:-1] if has_percent else token
    raw = raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value, has_percent, raw


def _add_text_numbers(allowed: set[float], text: object) -> None:
    for match in _NUM_RE.finditer(str(text or "")):
        parsed = _token_value(match.group(0))
        if parsed is None:
            continue
        value, has_percent, _raw = parsed
        allowed.add(value)
        if has_percent:
            allowed.add(value / 100)


def _collect_allowed(card: dict[str, Any], chart: ChartData) -> set[float]:
    """把统计卡里所有可引用的数字摊平成集合，供 caption 校验。"""
    allowed: set[float] = set()

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            value = float(node)
            if math.isfinite(value):
                allowed.add(value)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value)
                _add_text_numbers(allowed, key)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
            return
        if isinstance(node, str):
            _add_text_numbers(allowed, node)

    walk(card)

    # 带 % 的序列额外允许"小数形式"：12.5% 与 0.125 等价
    for entry in card.get("series", []):
        if isinstance(entry, dict) and entry.get("unit") == "%":
            for value in entry.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    allowed.add(float(value) / 100)

    for label in chart.labels:
        _add_text_numbers(allowed, label)
    for item in chart.series:
        _add_text_numbers(allowed, item.name)
    if chart.x_values:
        for value in chart.x_values:
            allowed.add(float(value))

    allowed.add(1.0)  # "图 1"
    return allowed


def _matches_raw(value: float, decimals: int, allowed: set[float]) -> bool:
    tolerance = 0.5 * (10**-decimals) if decimals > 0 else 0.0
    return any(
        math.isclose(
            value,
            reference,
            rel_tol=1e-6,
            abs_tol=max(1e-9, tolerance),
        )
        for reference in allowed
    )


def _matches_exact(value: float, allowed: set[float]) -> bool:
    return any(
        math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-9)
        for reference in allowed
    )


def _number_matches(
    value: float,
    has_percent: bool,
    decimals: int,
    allowed: set[float],
) -> bool:
    """相对误差 1e-6，或按模型写出的位数做四舍五入匹配。

    例如统计卡里是 12.345，模型写 12.3，按 1 位小数应视为同一数字。
    百分比只做两向等价中的一个方向：`50%` 可以匹配统计卡里的 `0.5`，
    但**不给比例换算叠加四舍五入容差** —— 否则 `42.7%` 会被 0.427
    与 0.42 的容差误判为合法，正好放过了我们要拦的编造数字。
    """
    if _matches_raw(value, decimals, allowed):
        return True
    if has_percent and _matches_exact(value / 100, allowed):
        return True
    return False


def validate_caption(
    text: str, allowed: set[float]
) -> tuple[str, list[str], int]:
    """抽出 caption 里的数字并逐个核对。

    返回 (清洗后的文本, 编造的数字列表, 数字总数)。编造的数字原位替换成
    `[未核实]`，而不是整句删除 —— 这样读者能看到"模型在这里说了个数，
    但我们无法核实"，比悄悄删掉更透明。
    """
    invented: list[str] = []
    total = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total
        token = match.group(0)
        parsed = _token_value(token)
        if parsed is None:
            return token
        value, has_percent, raw = parsed
        total += 1
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if _number_matches(value, has_percent, decimals, allowed):
            return token
        invented.append(token)
        return "[未核实]"

    scrubbed = _NUM_RE.sub(replace, text)
    return scrubbed, invented, total


def _validate_key_numbers(
    raw: object, allowed: set[float]
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw, list):
        return [], []
    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        raw_value = item.get("value")
        if isinstance(raw_value, str):
            parsed = _token_value(raw_value.strip())
            if parsed is None:
                rejected.append(raw_value)
                continue
            value, has_percent, _raw = parsed
        else:
            try:
                value = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rejected.append(str(raw_value))
                continue
            has_percent = False
        if not math.isfinite(value):
            continue
        if not _number_matches(value, has_percent, 0, allowed):
            rejected.append(fmt_number(value))
            continue
        kept.append(
            {
                "value": round(value, 6),
                "meaning": str(item.get("meaning", "")).strip(),
            }
        )
    return kept, rejected


def _with_unit(value: float, unit: str) -> str:
    text = fmt_number(value)
    if unit in ("%", "‰"):
        text += unit
    return text


def _template_caption(
    chart: ChartData, entries: Sequence[SeriesStats], card: dict[str, Any]
) -> str:
    """模型不可用或不可信时的纯统计卡图注。每个数字都直接来自统计卡。"""
    if not entries:
        return "图 1. 数据为空，无法生成图注。"

    if chart.kind == "radar":
        parts = []
        for entry in entries:
            parts.append(
                f"{entry.name} 的范围为 "
                f"{_with_unit(entry.min, entry.unit)}–"
                f"{_with_unit(entry.max, entry.unit)}，"
                f"均值 {_with_unit(entry.mean, entry.unit)}"
            )
        return (
            f"图 1. {len(chart.series)} 个类别在 {len(chart.labels)} 个指标上的对比。"
            + "；".join(parts)
            + "。各轴按自身范围归一化。"
        )

    if chart.kind == "scatter" and chart.x_values:
        parts = []
        for entry in entries:
            correlation = next(
                (
                    item["r"]
                    for item in card.get("correlations", [])
                    if item.get("a") == chart.x_label and item.get("b") == entry.name
                ),
                0.0,
            )
            parts.append(
                f"{entry.name} 的范围为 "
                f"{_with_unit(entry.min, entry.unit)}–"
                f"{_with_unit(entry.max, entry.unit)}，"
                f"均值 {_with_unit(entry.mean, entry.unit)}，"
                f"与 {chart.x_label} 的相关系数 r={fmt_number(correlation)}"
            )
        return f"图 1. {'; '.join(parts)}。散点仅呈现观测值，不含拟合线。"

    if chart.kind == "bar":
        parts = []
        for entry in entries:
            parts.append(
                f"{entry.name} 在「{entry.max_label}」取得最大值 "
                f"{_with_unit(entry.max, entry.unit)}，在「{entry.min_label}」"
                f"取得最小值 {_with_unit(entry.min, entry.unit)}，"
                f"均值 {_with_unit(entry.mean, entry.unit)}"
            )
        return f"图 1. {len(chart.labels)} 个类别上的观测值。" + "；".join(parts) + "。"

    parts = []
    for entry in entries:
        parts.append(
            f"{entry.name} 从 {_with_unit(entry.first, entry.unit)} 变化到 "
            f"{_with_unit(entry.last, entry.unit)}，范围 "
            f"{_with_unit(entry.min, entry.unit)}–"
            f"{_with_unit(entry.max, entry.unit)}，"
            f"均值 {_with_unit(entry.mean, entry.unit)}，"
            f"中位数 {_with_unit(entry.median, entry.unit)}，"
            f"逐行趋势斜率 {_with_unit(entry.slope, entry.unit)}/行"
        )
    return f"图 1. {chart.x_label} 上的序列变化。" + "；".join(parts) + "。"


def _data_range_line(chart: ChartData) -> str:
    return (
        f"数据区间：n={len(chart.labels)}"
        f"（{chart.x_label or '行'}，{len(chart.series)} 个序列）。"
    )


def _append_data_range(caption: str, chart: ChartData) -> str:
    # 去掉模型可能自己写的"数据区间"行，保证这一行只有一个、且由服务端生成
    base = re.sub(r"\n?\s*数据区间：[^\n]*$", "", caption.strip()).strip()
    return f"{base}\n{_data_range_line(chart)}".strip()


SYSTEM_PROMPT = """你是严谨的科研图表图注助手。铁律：

1. 只允许使用统计卡里已经出现的数字；不要自己计算、不要外推、不要编造。
2. 不要写"证明""表明""导致"这类因果断言，只描述图中可观察到的量。
3. caption 用中文，1–3 句：先说明图展示什么，再给 1–2 个关键数值。
4. key_numbers 里每个 value 必须原样来自统计卡，meaning 用一句中文说明它代表什么。
5. 只输出 JSON，不要解释文字：
{"caption": "图注文字", "key_numbers": [{"value": 1.23, "meaning": "该数字的含义"}]}"""


@dataclass(slots=True)
class CaptionResult:
    text: str
    key_numbers: list[dict[str, Any]] = field(default_factory=list)
    trusted: bool = True
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


def build_caption(chart: ChartData, *, llm_call=None) -> CaptionResult:
    """生成图注并做服务端数字校验。

    `llm_call` 允许测试注入假模型；不传时才在函数内部 import 真实
    `chat_json`，这样 monkeypatch `app.llm.client.chat_json` 也能生效。
    """
    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    entries = _card_entries(chart)
    card = _stats_card(chart, entries)
    allowed = _collect_allowed(card, chart)
    template = _append_data_range(_template_caption(chart, entries, card), chart)

    user = (
        f"图表类型：{chart.kind}\n"
        f"统计卡（唯一允许的数字来源）：\n"
        f"{json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "请为这张图写一段学术图注。"
    )

    try:
        raw = llm_call(user, system=SYSTEM_PROMPT, temperature=0.2)
    except Exception:  # noqa: BLE001 —— 模型不可用不能 500
        return CaptionResult(
            text=template,
            key_numbers=[],
            trusted=False,
            degraded=True,
            notes=["大模型不可用，已使用统计卡生成的模板图注。"],
        )

    if not isinstance(raw, dict) or not str(raw.get("caption", "")).strip():
        return CaptionResult(
            text=template,
            key_numbers=[],
            trusted=False,
            degraded=True,
            notes=["模型输出结构不可用（缺少 caption），已使用模板图注。"],
        )

    caption = str(raw["caption"]).strip()
    scrubbed, invented, total = validate_caption(caption, allowed)
    key_numbers, rejected_keys = _validate_key_numbers(
        raw.get("key_numbers"), allowed
    )
    notes: list[str] = []

    if rejected_keys:
        notes.append(
            "已剔除模型编造的关键数字："
            + "、".join(rejected_keys[:5])
            + "。"
        )

    # 没有数字，或编造比例超过 1/3：整段图注不可信，退化成模板
    trusted = total > 0 and len(invented) <= max(1, total // 3)
    if not trusted:
        reason = (
            "模型图注没有任何可核验的数字"
            if total == 0
            else f"模型图注中 {len(invented)}/{total} 个数字无法在统计卡中核实"
        )
        notes.append(reason + "，已整体改用模板图注。")
        return CaptionResult(
            text=template,
            key_numbers=key_numbers,
            trusted=False,
            degraded=False,
            notes=notes,
        )

    if invented:
        unique = list(dict.fromkeys(invented))
        notes.append(
            f"剔除了 {len(invented)} 个统计卡中不存在的数字"
            f"（{'、'.join(unique[:5])}），已在原文替换为 [未核实]。"
        )

    return CaptionResult(
        text=_append_data_range(scrubbed, chart),
        key_numbers=key_numbers,
        trusted=True,
        degraded=False,
        notes=notes,
    )


# ===================================================================== 路由


@router.post("/api/dataviz/chart", response_model=DatavizOut)
def dataviz_chart(
    file: UploadFile = File(..., description="CSV 文件（UTF-8）"),
    kind: str | None = Form(None, description="line | bar | scatter | radar"),
    x: str | None = Form(None, description="横轴列名"),
    y: str | None = Form(None, description="纵轴列名，多个用逗号分隔"),
    dark: bool = Form(False),
    caption: bool = Form(True),
) -> DatavizOut:
    """上传 CSV，返回统计表 + SVG 图 + Mermaid/TikZ/Matplotlib + 学术图注。

    CSV 在请求内解析，不落盘、不进入论文 registry。这样做的取舍：
    无状态、无需清理、不会把用户的实验数据留在服务器上；代价是无法对
    同一个文件做增量查询，每次都要重新上传。
    """
    data = _read_csv_upload(file)
    table = _parse_upload(data)
    # 前端把未填写的可选字段作为空字符串提交时，按"未指定"处理，
    # 否则空 kind 会被当成非法图表类型直接 422。
    kind = (kind or "").strip() or None
    x = (x or "").strip() or None
    y = (y or "").strip() or None
    chart_data = _build_chart(table, kind, x, y)
    rendered = render_chart(chart_data, dark=dark)

    notes = list(table.notes)
    if chart_data.note:
        notes.append(chart_data.note)
    if len(table.rows) > MAX_RESPONSE_ROWS:
        notes.append(
            f"响应中的 table.rows 仅保留前 {MAX_RESPONSE_ROWS} 行，"
            f"完整数据共 {len(table.rows)} 行。"
        )

    caption_out: CaptionOut | None = None
    degraded = False
    if caption:
        result = build_caption(chart_data)
        caption_out = CaptionOut(
            text=result.text,
            key_numbers=[KeyNumberOut(**item) for item in result.key_numbers],
            trusted=result.trusted,
            notes=list(result.notes),
        )
        notes.extend(result.notes)
        degraded = result.degraded

    return DatavizOut(
        table=_table_out(table, max_rows=MAX_RESPONSE_ROWS),
        chart=ChartOut(**rendered),
        caption=caption_out,
        exports=ExportsOut(
            mermaid=to_mermaid(chart_data),
            tikz=to_tikz(chart_data),
            matplotlib=to_matplotlib_script(chart_data),
        ),
        degraded=degraded,
        notes=_dedupe(notes),
    )


@router.post("/api/dataviz/preview", response_model=PreviewOut)
def dataviz_preview(
    file: UploadFile = File(..., description="CSV 文件（UTF-8）"),
) -> PreviewOut:
    """只解析不画图：大 CSV 可以先看列类型、统计量和前 100 行。"""
    data = _read_csv_upload(file)
    table = _parse_upload(data)
    notes = list(table.notes)
    if len(table.rows) > MAX_PREVIEW_ROWS:
        notes.append(
            f"预览仅返回前 {MAX_PREVIEW_ROWS} 行，完整数据共 {len(table.rows)} 行。"
        )
    return PreviewOut(
        table=_table_out(table, max_rows=MAX_PREVIEW_ROWS),
        notes=_dedupe(notes),
    )
