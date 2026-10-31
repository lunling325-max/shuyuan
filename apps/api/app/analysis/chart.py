"""服务端 SVG 图表：line / bar / scatter / radar。

为什么手写 SVG，而不是 matplotlib / plotly：

1. 部署依赖已经很克制，matplotlib 会额外带进几十 MB 的字体与绘图后端；
   比赛环境离线安装，能少一个依赖就少一个失败点。
2. 我们只画四种图，且要"像论文里的图"——细轴线、弱网格、固定色板、
   选择性标注。这些细节手写反而比调库更可控，输出还能做到完全确定性
   （没有时间戳、没有随机 id），便于测试逐字比较。

设计上的硬约束：

- **绝不使用双 Y 轴**。两个量纲不同的序列会拆成并排小图（small multiples），
  并在 `note` 里说明。双 Y 轴会制造"看起来相关"的假象，学术图表里应避免。
- 分类色板固定顺序、最多 8 个系列，第 9 个起合并成「其他」。色板不循环，
  避免同一张图里两种含义共用一种颜色。
- 单系列柱状图的颜色按数值强弱在同一色相内由浅到深，而不是按行号换色。
- 深色模式从同一套色相里取更亮的一档，不是把浅色简单反色。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as _xml_escape

from app.analysis.csv_table import Column, Table, fmt_number

KINDS = ("line", "bar", "scatter", "radar")
MAX_SERIES = 8
MAX_LINE_POINTS = 400
MAX_BAR_POINTS = 40
MAX_RADAR_AXES = 12
# 超过这个基数的类别轴不再适合柱状图：每根柱子会细到看不清，
# 这时按行序退化为折线，比硬塞 40 根柱子更诚实。
LOW_CARDINALITY_MAX = 24

# 固定的分类色板：顺序写死，不循环。深色模式是同一组色相在深背景下的
# 亮度步进（更亮、略降饱和），而不是 RGB 简单取反。
PALETTE_LIGHT = (
    "#2f6db3",
    "#d9772f",
    "#2f8f6b",
    "#a0446f",
    "#7a5aa6",
    "#8a6d1f",
    "#3f7f8f",
    "#6b6b6b",
)
PALETTE_DARK = (
    "#7fb0e6",
    "#f0a06a",
    "#6cc9a3",
    "#e08cb0",
    "#b39adb",
    "#d1b45a",
    "#7ec2cf",
    "#b0b0b0",
)

THEME_LIGHT = {
    "bg": "#ffffff",
    "panel": "#f7f9fb",
    "ink": "#1f2933",
    "ink2": "#52606d",
    "ink3": "#7b8794",
    "axis": "#cbd2d9",
    "grid": "#e4e7eb",
    "zero": "#9aa5b1",
}
THEME_DARK = {
    "bg": "#11161d",
    "panel": "#161d26",
    "ink": "#e6edf3",
    "ink2": "#b6c2cf",
    "ink3": "#8a97a5",
    "axis": "#3b4652",
    "grid": "#2a3340",
    "zero": "#596677",
}


class ChartError(ValueError):
    """请求无法映射成一张合理的图。"""


@dataclass(slots=True)
class Series:
    name: str
    values: list[float]
    unit: str = ""


@dataclass(slots=True)
class ChartData:
    kind: str
    title: str
    x_label: str
    y_label: str
    labels: list[str]
    series: list[Series]
    x_values: list[float] | None = None
    note: str = ""
    small_multiples: bool = False


def _parse_y(y: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if y is None:
        return None
    if isinstance(y, (list, tuple)):
        parts = [str(item).strip() for item in y]
    else:
        parts = [part.strip() for part in str(y).replace("，", ",").split(",")]
    parts = [part for part in parts if part]
    return parts or None


def _is_monotonic(values: list[object]) -> bool:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 3 or len(clean) < 0.8 * len(values):
        return False
    return all(right > left for left, right in zip(clean, clean[1:]))


def _cardinality(column: Column) -> int:
    """类别列的基数（去掉缺失值后的不同取值数）。"""
    return len({value for value in column.values if value is not None})


def _downsample(indices: list[int], limit: int) -> tuple[list[int], int]:
    """均匀抽稀，但保留首尾。返回 (保留索引, 被丢弃数量)。"""
    if len(indices) <= limit or limit < 2:
        return indices, 0
    picks = sorted(
        {
            round(position * (len(indices) - 1) / (limit - 1))
            for position in range(limit)
        }
    )
    return [indices[pick] for pick in picks], len(indices) - len(picks)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cap_series(series: list[Series]) -> tuple[list[Series], str]:
    if len(series) <= MAX_SERIES:
        return series, ""
    head = series[: MAX_SERIES - 1]
    rest = series[MAX_SERIES - 1 :]
    values = [_mean([item.values[i] for item in rest]) for i in range(len(rest[0].values))]
    # 「其他」的单位取合并项的共同单位；否则会被误判成"混合量纲"而触发分面
    rest_units = {item.unit for item in rest}
    other_unit = rest[0].unit if len(rest_units) == 1 else ""
    head = list(head) + [Series(name="其他", values=values, unit=other_unit)]
    return head, "序列超过 8 个，第 9 个起已合并为「其他」（逐行取均值）。"


def _cap_radar_rows(
    labels: list[str], rows: list[list[float]]
) -> tuple[list[str], list[list[float]], str]:
    if len(rows) <= MAX_SERIES:
        return labels, rows, ""
    head_labels = labels[: MAX_SERIES - 1] + ["其他"]
    head_rows = rows[: MAX_SERIES - 1]
    rest = rows[MAX_SERIES - 1 :]
    other = [
        _mean([row[axis] for row in rest]) for axis in range(len(rows[0]))
    ]
    return head_labels, head_rows + [other], (
        "雷达图类别超过 8 个，第 8 个起合并为「其他」（逐轴取均值）。"
    )


def build_chart_data(
    table: Table,
    *,
    kind: str | None = None,
    x: str | None = None,
    y: str | list[str] | None = None,
) -> ChartData:
    """把 `Table` 变成一份"所有渲染器共用"的清洗数据。

    关键取舍：**先按选定列剔除缺值行，再画图**。这样 SVG、TikZ、Mermaid
    和 Matplotlib 拿到的是同一批点，不会出现"图里 7 个点、导出 6 个点"。
    被剔除的行数写进 `note`，读者仍然知道数据被裁剪过。
    """
    numeric_cols = [column for column in table.columns if column.kind == "number"]
    cat_cols = [column for column in table.columns if column.kind == "category"]
    date_cols = [column for column in table.columns if column.kind == "date"]
    if not numeric_cols:
        raise ChartError("没有数值列，无法绘图。")

    numeric_names = [column.name for column in numeric_cols]
    y_names = _parse_y(y)
    if y_names:
        for name in y_names:
            column = table.column(name)
            if column is None:
                raise ChartError(f"y 列「{name}」不存在。")
            if column.kind != "number":
                raise ChartError(f"y 列「{name}」不是数值列。")
        # 去重但保持用户给出的顺序
        y_names = list(dict.fromkeys(y_names))

    if kind is not None:
        kind = kind.strip().lower()
        if kind not in KINDS:
            raise ChartError(
                f"不支持的图表类型「{kind}」，可选：{', '.join(KINDS)}。"
            )

    xcol: Column | None = None
    if x:
        xcol = table.column(x)
        if xcol is None:
            raise ChartError(f"x 列「{x}」不存在。")

    notes: list[str] = []

    # ---------------------------------------------------------------- 推断

    def pick_for_category(column: Column, names: list[str]) -> str:
        """按类别基数挑图：低基数才适合柱状/雷达，高基数按行序画折线。"""
        cardinality = _cardinality(column)
        if len(names) >= 3 and cardinality <= MAX_SERIES:
            return "radar"
        if cardinality <= LOW_CARDINALITY_MAX:
            return "bar"
        return "line"

    if kind is None:
        if xcol is not None:
            if xcol.kind == "date":
                # 日期是天然的有序横轴，画柱状图会把时间连续性格丢掉
                if y_names is None:
                    y_names = list(numeric_names)
                kind = "line"
            elif xcol.kind == "number":
                if y_names is None:
                    y_names = [name for name in numeric_names if name != xcol.name]
                kind = (
                    "line"
                    if (len(y_names or []) > 1 or _is_monotonic(xcol.values))
                    else "scatter"
                )
            else:
                if y_names is None:
                    y_names = list(numeric_names)
                kind = pick_for_category(xcol, y_names)
        elif date_cols:
            # 有日期列时优先把它当横轴，比用递增的行号/id 更有语义
            xcol = date_cols[0]
            x = xcol.name
            y_names = y_names or list(numeric_names)
            kind = "line"
        else:
            monotonic = next(
                (column for column in numeric_cols if _is_monotonic(column.values)),
                None,
            )
            if monotonic is not None and len(numeric_names) >= 2:
                xcol = monotonic
                x = xcol.name
                y_names = y_names or [
                    name for name in numeric_names if name != xcol.name
                ]
                kind = "line"
            elif cat_cols:
                xcol = cat_cols[0]
                x = xcol.name
                y_names = y_names or list(numeric_names)
                kind = pick_for_category(xcol, y_names)
            elif len(numeric_names) >= 2:
                xcol = numeric_cols[0]
                x = xcol.name
                y_names = y_names or [numeric_names[1]]
                kind = "scatter"
            else:
                xcol = None
                x = None
                y_names = y_names or [numeric_names[0]]
                kind = "line"

    # ---------------------------------------------------------------- 补全显式参数
    if kind == "bar":
        if xcol is None:
            candidates = date_cols or cat_cols
            if candidates:
                xcol = candidates[0]
                x = xcol.name
        if y_names is None:
            y_names = [numeric_names[0]]
    elif kind == "line":
        # 显式指定 kind 但没给 x 时，仍然按语义补一个横轴，
        # 否则城市名/日期会退化成"行1、行2"这种无信息标签。
        if xcol is None:
            if date_cols:
                xcol = date_cols[0]
                x = xcol.name
            elif cat_cols:
                xcol = cat_cols[0]
                x = xcol.name
            elif len(numeric_cols) >= 2:
                monotonic = next(
                    (column for column in numeric_cols if _is_monotonic(column.values)),
                    None,
                )
                if monotonic is not None:
                    xcol = monotonic
                    x = xcol.name
        if y_names is None:
            if xcol is not None and xcol.kind == "number":
                y_names = [name for name in numeric_names if name != xcol.name]
            if not y_names:
                y_names = list(numeric_names)
                if xcol is not None and xcol.kind == "number":
                    xcol = None
                    x = None
        else:
            if xcol is not None and xcol.name in y_names:
                y_names = [name for name in y_names if name != xcol.name]
                if not y_names:
                    raise ChartError("折线图的 y 列不能全部与横轴列相同。")
    elif kind == "scatter":
        if xcol is None:
            xcol = numeric_cols[0]
            x = xcol.name
        if xcol.kind != "number":
            raise ChartError("散点图的 x 列必须是数值列。")
        if y_names is None:
            y_names = [name for name in numeric_names if name != xcol.name]
        else:
            y_names = [name for name in y_names if name != xcol.name]
        if not y_names:
            raise ChartError("散点图至少需要一个与 x 不同的数值 y 列。")
    elif kind == "radar":
        if xcol is None:
            candidates = cat_cols or date_cols
            if candidates:
                xcol = candidates[0]
                x = xcol.name
        if xcol is None:
            raise ChartError("雷达图需要一个类别列作为分组依据。")
        if xcol.kind not in ("category", "date"):
            raise ChartError("雷达图的分组列必须是类别或日期列。")
        if y_names is None:
            y_names = list(numeric_names)
    else:  # pragma: no cover - KINDS 已保证不会走到这里
        raise ChartError(f"不支持的图表类型「{kind}」。")

    if not y_names:
        raise ChartError("没有可绘制的数值序列。")

    if kind == "radar":
        if len(y_names) < 3:
            raise ChartError("雷达图至少需要 3 个数值指标列。")
        if len(y_names) > MAX_RADAR_AXES:
            y_names = y_names[:MAX_RADAR_AXES]
            notes.append(f"雷达图指标超过 {MAX_RADAR_AXES} 个，仅保留前 {MAX_RADAR_AXES} 个。")

    y_cols = [table.column(name) for name in y_names if table.column(name) is not None]
    if len(y_cols) != len(y_names):  # pragma: no cover - 前面已校验
        raise ChartError("存在无法识别的 y 列。")

    # ---------------------------------------------------------------- 清洗行
    keep: list[int] = []
    for row_index in range(len(table.rows)):
        if xcol is not None and xcol.values[row_index] is None:
            continue
        if any(column.values[row_index] is None for column in y_cols):
            continue
        keep.append(row_index)

    total_rows = len(table.rows)
    dropped_missing = total_rows - len(keep)
    if not keep:
        raise ChartError("剔除缺失值后没有可绘制的数据。")

    limit = MAX_BAR_POINTS if kind == "bar" else MAX_LINE_POINTS
    keep, downsampled = _downsample(keep, limit)
    if downsampled:
        notes.append(
            f"数据点过多，已均匀抽稀到 {len(keep)} 行（首尾保留）；"
            "SVG 与所有导出使用同一批点。"
        )
    if dropped_missing:
        notes.append(
            f"有 {dropped_missing} 行因所选列缺值未绘制；"
            f"实际绘图 {len(keep)} 行，导出点数与此一致。"
        )

    def label_for(row_index: int) -> str:
        if xcol is None:
            return f"行{row_index + 1}"
        value = xcol.values[row_index]
        # 数值横轴用紧凑格式（1.0 -> 1），避免刻度标签出现无意义的小数位
        return fmt_number(value) if xcol.kind == "number" else str(value)

    # ---------------------------------------------------------------- 组装
    if kind == "radar":
        axes = list(y_names)
        row_labels = [label_for(index) for index in keep]
        rows = [[float(column.values[index]) for column in y_cols] for index in keep]
        row_labels, rows, cap_note = _cap_radar_rows(row_labels, rows)
        if cap_note:
            notes.append(cap_note)
        series = [
            Series(name=row_labels[i], values=rows[i], unit="")
            for i in range(len(rows))
        ]
        title_names = "、".join(row_labels[:3])
        if len(row_labels) > 3:
            title_names += " 等"
        notes.append("雷达图各轴按自身范围归一化到 0–1；原始数值未被修改。")
        return ChartData(
            kind="radar",
            title=f"多指标对比：{title_names}",
            x_label=xcol.name if xcol is not None else "",
            y_label="指标（归一化）",
            labels=axes,
            series=series,
            x_values=None,
            note=" ".join(notes),
            small_multiples=False,
        )

    labels = [label_for(index) for index in keep]
    x_values: list[float] | None = None
    if kind in ("line", "scatter") and xcol is not None and xcol.kind == "number":
        x_values = [float(xcol.values[index]) for index in keep]

    series = [
        Series(
            name=column.name,
            values=[float(column.values[index]) for index in keep],
            unit=column.unit,
        )
        for column in y_cols
    ]
    series, cap_note = _cap_series(series)
    if cap_note:
        notes.append(cap_note)

    # ---------------------------------------------------------------- 双 Y 轴
    small_multiples = False
    if len(series) >= 2:
        # 注意这里不能过滤掉空单位：`%` 与"无单位"混在一张图里，
        # 本身就是两种量纲，必须分面。过滤掉空串会漏掉这种情况。
        units = {item.unit for item in series}
        if len(units) > 1:
            small_multiples = True
            shown_units = sorted(unit or "无单位" for unit in units)
            notes.append(
                "检测到多个量纲（"
                + "、".join(shown_units)
                + "）；不使用双 Y 轴，改为并排小图（small multiples）。"
            )
        else:
            magnitudes = [max((abs(value) for value in item.values), default=0.0) for item in series]
            positives = [value for value in magnitudes if value > 0]
            if len(positives) >= 2 and max(positives) / min(positives) >= 1000:
                small_multiples = True
                notes.append(
                    f"序列量级相差约 {max(positives) / min(positives):.0f} 倍；"
                    "不使用双 Y 轴，改为并排小图（small multiples）。"
                )

    x_label = xcol.name if xcol is not None else "行"
    if len(series) == 1:
        y_label = series[0].name + (f"（{series[0].unit}）" if series[0].unit else "")
    else:
        units = {item.unit for item in series if item.unit}
        y_label = next(iter(units)) if len(units) == 1 else "数值"

    joined = "、".join(item.name for item in series[:4])
    if len(series) > 4:
        joined += " 等"
    if kind == "line":
        title = f"{joined} 随 {x_label} 的变化"
    elif kind == "bar":
        title = f"{joined} 按 {x_label} 分组"
    else:
        title = f"{joined} 与 {x_label} 的关系"
    if small_multiples:
        title = f"{joined} 的分面对比（small multiples）"

    return ChartData(
        kind=kind,
        title=title,
        x_label=x_label,
        y_label=y_label,
        labels=labels,
        series=series,
        x_values=x_values,
        note=" ".join(notes),
        small_multiples=small_multiples,
    )


# ====================================================================== SVG


def _esc(text: object) -> str:
    """XML 转义。标签里出现 `<b>` 时必须变成实体，否则整段 SVG 会坏掉。"""
    return _xml_escape(str(text), {'"': "&quot;", "'": "&apos;"})


def _p(value: float) -> str:
    """坐标保留 1 位小数，既够精确又保证输出稳定。"""
    return f"{value:.1f}"


def _clean_float(value: float) -> float:
    return round(float(value), 10)


def _trunc(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _text_width(text: str, size: float = 11.0) -> float:
    width = 0.0
    for char in str(text):
        width += size * (0.62 if ord(char) < 128 else 1.0)
    return width


def _theme(dark: bool) -> dict[str, str]:
    return THEME_DARK if dark else THEME_LIGHT


def _palette(dark: bool) -> tuple[str, ...]:
    return PALETTE_DARK if dark else PALETTE_LIGHT


def _colors(count: int, dark: bool) -> list[str]:
    palette = _palette(dark)
    return [palette[index] for index in range(min(count, len(palette)))]


def _mix(color_a: str, color_b: str, t: float) -> str:
    """两个 #RRGGBB 之间线性插值；t=0 取 a，t=1 取 b。"""
    t = max(0.0, min(1.0, t))
    a = color_a.lstrip("#")
    b = color_b.lstrip("#")
    channels = []
    for offset in (0, 2, 4):
        left = int(a[offset : offset + 2], 16)
        right = int(b[offset : offset + 2], 16)
        channels.append(round(left + (right - left) * t))
    return "#" + "".join(f"{value:02x}" for value in channels)


def _style(theme: dict[str, str]) -> str:
    # 内联 SVG 的 <style> 是文档级的，不加作用域会污染页面里的 text/.title
    # 等通用选择器。所有规则都挂在根节点的 .dataviz-svg 下，互不越界。
    scope = ".dataviz-svg"
    return (
        f"{scope} text{{font-family:system-ui,-apple-system,'Segoe UI',"
        "'Noto Sans CJK SC','Microsoft YaHei',sans-serif;"
        f"fill:{theme['ink']};}}"
        f"{scope} .title{{font-size:15px;font-weight:600;fill:{theme['ink']};}}"
        f"{scope} .subtitle{{font-size:11px;fill:{theme['ink3']};}}"
        f"{scope} .tick{{font-size:11px;fill:{theme['ink3']};}}"
        f"{scope} .axis-label{{font-size:12px;fill:{theme['ink2']};}}"
        f"{scope} .legend{{font-size:11px;fill:{theme['ink2']};}}"
        f"{scope} .value-label{{font-size:10px;fill:{theme['ink2']};}}"
        f"{scope} .grid{{stroke:{theme['grid']};stroke-width:1;stroke-dasharray:3 3;}}"
        f"{scope} .axis{{stroke:{theme['axis']};stroke-width:1;}}"
        f"{scope} .baseline{{stroke:{theme['zero']};stroke-width:1;stroke-dasharray:4 4;}}"
        f"{scope} .tick-mark{{stroke:{theme['axis']};stroke-width:1;}}"
    )


def _svg_head(
    width: int, height: int, title: str, theme: dict[str, str]
) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="dataviz-svg" '
            f'width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{_esc(_trunc(title, 80))}">'
        ),
        f"<title>{_esc(_trunc(title, 80))}</title>",
        f"<style>{_style(theme)}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{theme["bg"]}"/>',
    ]


def _legend(
    parts: list[str],
    series: list[Series],
    colors: list[str],
    theme: dict[str, str],
    start_x: float,
    y: float,
    max_x: float,
) -> float:
    """横排图例，放不下就换行。返回最后一行的 y。"""
    cursor_x = start_x
    cursor_y = y
    for index, item in enumerate(series):
        name = _trunc(item.name, 20)
        width = _text_width(name) + 24
        if cursor_x + width > max_x and cursor_x > start_x:
            cursor_x = start_x
            cursor_y += 16
        color = colors[index] if index < len(colors) else theme["ink2"]
        parts.append(
            f'<rect x="{_p(cursor_x)}" y="{_p(cursor_y - 8)}" width="12" '
            f'height="9" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text class="legend" x="{_p(cursor_x + 17)}" '
            f'y="{_p(cursor_y)}">{_esc(name)}</text>'
        )
        cursor_x += width
    return cursor_y


def _nice_axis(
    vmin: float, vmax: float, *, include_zero: bool = False, target: int = 5
) -> tuple[float, float, float, list[float]]:
    """1/2/5 × 10^k 的刻度：网格线落在人眼好读的数上。"""
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax == vmin:
        pad = abs(vmax) * 0.1 or 1.0
        vmin -= pad
        vmax += pad
    if not include_zero:
        pad = (vmax - vmin) * 0.06
        vmin -= pad
        vmax += pad

    span = vmax - vmin
    raw_step = span / max(1, target)
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        step = 1 * 10**exponent
    elif fraction <= 2:
        step = 2 * 10**exponent
    elif fraction <= 5:
        step = 5 * 10**exponent
    else:
        step = 10 * 10**exponent

    low = math.floor(vmin / step) * step
    high = math.ceil(vmax / step) * step
    ticks: list[float] = []
    value = low
    guard = 0
    while value <= high + step * 1e-9 and guard < 100:
        ticks.append(_clean_float(value))
        value += step
        guard += 1
    return _clean_float(low), _clean_float(high), step, ticks


def _bar_path(
    x: float, width: float, top: float, bottom: float, *, round_top: bool, radius: float = 4.0
) -> str:
    """柱子路径：只圆化远离基线的那一端，柱基始终贴住基线。"""
    if width <= 0 or abs(bottom - top) < 0.5:
        return ""
    r = min(radius, width / 2, abs(bottom - top) / 2)
    if round_top:
        return (
            f"M {_p(x)} {_p(bottom)} "
            f"L {_p(x)} {_p(top + r)} "
            f"Q {_p(x)} {_p(top)} {_p(x + r)} {_p(top)} "
            f"L {_p(x + width - r)} {_p(top)} "
            f"Q {_p(x + width)} {_p(top)} {_p(x + width)} {_p(top + r)} "
            f"L {_p(x + width)} {_p(bottom)} Z"
        )
    return (
        f"M {_p(x)} {_p(top)} "
        f"L {_p(x + width)} {_p(top)} "
        f"L {_p(x + width)} {_p(bottom - r)} "
        f"Q {_p(x + width)} {_p(bottom)} {_p(x + width - r)} {_p(bottom)} "
        f"L {_p(x + r)} {_p(bottom)} "
        f"Q {_p(x)} {_p(bottom)} {_p(x)} {_p(bottom - r)} Z"
    )


def _value_label(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    *,
    above: bool,
    top_limit: float,
) -> None:
    if above and y - 8 < top_limit:
        above = False
    dy = -8 if above else 15
    parts.append(
        f'<text class="value-label" x="{_p(x)}" y="{_p(y + dy)}" '
        f'text-anchor="middle">{_esc(text)}</text>'
    )


def _unit_suffix(unit: str) -> str:
    return unit if unit in ("%", "‰") else ""


def _draw_line_or_scatter(
    parts: list[str],
    data: ChartData,
    colors: list[str],
    theme: dict[str, str],
    x_index,
    y,
    plot_top: float,
    label_points: bool = True,
) -> None:
    for series_index, item in enumerate(data.series):
        color = colors[series_index]
        points = [(x_index(index), y(value)) for index, value in enumerate(item.values)]
        if data.kind == "line" and len(points) > 1:
            path = "M " + " L ".join(f"{_p(px)} {_p(py)}" for px, py in points)
            parts.append(
                f'<path class="series" data-series="{_esc(item.name)}" d="{path}" '
                f'fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        if data.kind == "line" or data.kind == "scatter":
            for px, py in points:
                parts.append(
                    f'<circle class="point" data-series="{_esc(item.name)}" '
                    f'cx="{_p(px)}" cy="{_p(py)}" r="4" fill="{color}" '
                    f'stroke="{theme["bg"]}" stroke-width="1"/>'
                )
        if not label_points or not item.values:
            continue

        # 只标首末点与极值点；每个点都标数字会让图变成表格
        min_index = item.values.index(min(item.values))
        max_index = item.values.index(max(item.values))
        chosen = sorted({0, len(item.values) - 1, min_index, max_index})
        suffix = _unit_suffix(item.unit)
        for index in chosen:
            value = item.values[index]
            px, py = points[index]
            _value_label(
                parts,
                px,
                py,
                fmt_number(value) + suffix,
                above=True,
                top_limit=plot_top,
            )


def _draw_bars(
    parts: list[str],
    data: ChartData,
    colors: list[str],
    dark: bool,
    theme: dict[str, str],
    x_index,
    y,
    plot_width: float,
    plot_top: float,
) -> None:
    count = len(data.labels)
    series_count = len(data.series)
    group_width = plot_width / max(1, count)
    inner_width = group_width * 0.72
    gap = 2.0  # 相邻填充之间留 2px，柱子不会糊成一块
    bar_width = max(2.0, (inner_width - gap * (series_count - 1)) / series_count)
    baseline = y(0)

    for row_index in range(count):
        center = x_index(row_index)
        group_left = center - inner_width / 2
        for series_index, item in enumerate(data.series):
            value = item.values[row_index]
            bar_x = group_left + series_index * (bar_width + gap)
            top = y(value)
            bottom = baseline
            if abs(bottom - top) < 1:
                top = bottom - 1 if value >= 0 else bottom + 1
            if series_count > 1:
                color = colors[series_index]
            else:
                peak = max((abs(number) for number in item.values), default=0.0)
                if dark:
                    color = _mix("#24384d", "#7fb0e6", abs(value) / peak if peak else 0.0)
                else:
                    color = _mix("#dbe9f7", "#2f6db3", abs(value) / peak if peak else 0.0)
            path = _bar_path(
                bar_x,
                bar_width,
                min(top, bottom),
                max(top, bottom),
                round_top=value >= 0,
            )
            if path:
                parts.append(
                    f'<path class="series" data-series="{_esc(item.name)}" '
                    f'd="{path}" fill="{color}"/>'
                )

    # 选择性直接标注：单系列且柱子不多时逐根标数值；类别多或系列多时
    # 只标极值，以免标签互相压住。
    for series_index, item in enumerate(data.series):
        if series_count == 1:
            if count <= 8:
                chosen = set(range(count))
            else:
                chosen = {
                    item.values.index(min(item.values)),
                    item.values.index(max(item.values)),
                }
        else:
            chosen = {item.values.index(max(item.values))}
        suffix = _unit_suffix(item.unit)
        for row_index in chosen:
            value = item.values[row_index]
            bar_x = (
                x_index(row_index)
                - inner_width / 2
                + series_index * (bar_width + gap)
                + bar_width / 2
            )
            top = y(value)
            # top 是柱子远离基线的一端：正值时在上方，负值时在下方
            _value_label(
                parts,
                bar_x,
                top,
                fmt_number(value) + suffix,
                above=value >= 0,
                top_limit=plot_top,
            )


def _render_xy(data: ChartData, dark: bool) -> tuple[str, int, int]:
    theme = _theme(dark)
    width, height = 720, 420
    margin_left, margin_right = 68, 28
    legend_needed = len(data.series) >= 2
    margin_top = 80 if legend_needed else 52
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    parts = _svg_head(width, height, data.title, theme)
    parts.append(
        f'<text class="title" x="{margin_left}" y="24">'
        f"{_esc(_trunc(data.title, 48))}</text>"
    )
    colors = _colors(len(data.series), dark)
    if legend_needed:
        _legend(parts, data.series, colors, theme, margin_left, 46, width - margin_right)

    all_values = [value for item in data.series for value in item.values]
    raw_min = min(all_values)
    raw_max = max(all_values)
    include_zero = data.kind == "bar" or (
        raw_min >= 0
        and raw_max > 0
        and raw_min <= (raw_max - raw_min) * 0.25
    )
    low, high, _step, y_ticks = _nice_axis(
        raw_min, raw_max, include_zero=include_zero
    )

    def y(value: float) -> float:
        return (
            margin_top
            + plot_height
            - (value - low) / (high - low) * plot_height
        )

    numeric_x = data.x_values is not None
    x_ticks: list[float] = []
    if numeric_x:
        x_values = data.x_values or []
        x_low, x_high, _x_step, x_ticks = _nice_axis(
            min(x_values), max(x_values), include_zero=False
        )

        def x_numeric(value: float) -> float:
            return margin_left + (value - x_low) / (x_high - x_low) * plot_width

        def x_index(index: int) -> float:
            return x_numeric(x_values[index])

    else:
        count = len(data.labels)

        def x_index(index: int) -> float:
            return margin_left + plot_width * (index + 0.5) / count

    # 网格：只画横线，浅、发丝级
    for tick in y_ticks:
        tick_y = y(tick)
        parts.append(
            f'<line class="grid" x1="{_p(margin_left)}" y1="{_p(tick_y)}" '
            f'x2="{_p(margin_left + plot_width)}" y2="{_p(tick_y)}"/>'
        )
        parts.append(
            f'<text class="tick" x="{_p(margin_left - 8)}" y="{_p(tick_y + 4)}" '
            f'text-anchor="end">{_esc(fmt_number(tick))}</text>'
        )

    baseline = y(0) if (data.kind == "bar" and low <= 0 <= high) else (
        margin_top + plot_height
    )
    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + plot_height}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{_p(baseline)}" '
        f'x2="{_p(margin_left + plot_width)}" y2="{_p(baseline)}"/>'
    )
    if data.kind != "bar" and low < 0 < high:
        parts.append(
            f'<line class="baseline" x1="{margin_left}" y1="{_p(y(0))}" '
            f'x2="{_p(margin_left + plot_width)}" y2="{_p(y(0))}"/>'
        )

    if numeric_x:
        for tick in x_ticks:
            tick_x = x_numeric(tick)
            parts.append(
                f'<line class="tick-mark" x1="{_p(tick_x)}" y1="{_p(baseline)}" '
                f'x2="{_p(tick_x)}" y2="{_p(baseline + 4)}"/>'
            )
            parts.append(
                f'<text class="tick" x="{_p(tick_x)}" y="{_p(baseline + 17)}" '
                f'text-anchor="middle">{_esc(fmt_number(tick))}</text>'
            )
    else:
        count = len(data.labels)
        every = max(1, math.ceil(count / 12))
        for index, label in enumerate(data.labels):
            if index % every != 0 and index != count - 1:
                continue
            parts.append(
                f'<text class="tick" x="{_p(x_index(index))}" '
                f'y="{_p(baseline + 17)}" text-anchor="middle">'
                f"{_esc(_trunc(str(label), 10))}</text>"
            )

    parts.append(
        f'<text class="axis-label" x="{_p(margin_left + plot_width / 2)}" '
        f'y="{_p(height - 14)}" text-anchor="middle">{_esc(data.x_label)}</text>'
    )
    center_y = margin_top + plot_height / 2
    parts.append(
        f'<text class="axis-label" x="18" y="{_p(center_y)}" '
        f'text-anchor="middle" transform="rotate(-90 18 {_p(center_y)})">'
        f"{_esc(data.y_label)}</text>"
    )

    if data.kind == "bar":
        _draw_bars(
            parts,
            data,
            colors,
            dark,
            theme,
            x_index,
            y,
            plot_width,
            margin_top,
        )
    else:
        _draw_line_or_scatter(
            parts, data, colors, theme, x_index, y, margin_top
        )

    parts.append("</svg>")
    return "\n".join(parts), width, height


def _render_radar(data: ChartData, dark: bool) -> tuple[str, int, int]:
    theme = _theme(dark)
    width, height = 620, 520
    parts = _svg_head(width, height, data.title, theme)
    parts.append(
        f'<text class="title" x="24" y="24">{_esc(_trunc(data.title, 40))}</text>'
    )
    colors = _colors(len(data.series), dark)
    if len(data.series) >= 2:
        _legend(parts, data.series, colors, theme, 24, 46, width - 24)

    axes = data.labels
    axis_count = len(axes)
    if axis_count < 3:
        raise ChartError("雷达图至少需要 3 个轴。")

    center_x = width / 2
    center_y = height / 2 + 22
    radius = min(width - 150, height - 190) / 2
    angles = [
        -math.pi / 2 + 2 * math.pi * index / axis_count for index in range(axis_count)
    ]

    axis_min = [min(item.values[j] for item in data.series) for j in range(axis_count)]
    axis_max = [max(item.values[j] for item in data.series) for j in range(axis_count)]

    def normalize(value: float, axis: int) -> float:
        span = axis_max[axis] - axis_min[axis]
        # 所有类别在该轴上的值相同时，点落在中间，避免除零
        return 0.5 if span == 0 else (value - axis_min[axis]) / span

    # 网格环 + 辐条
    for fraction in (0.25, 0.5, 0.75, 1.0):
        ring = [
            (
                center_x + radius * fraction * math.cos(angle),
                center_y + radius * fraction * math.sin(angle),
            )
            for angle in angles
        ]
        path = "M " + " L ".join(f"{_p(px)} {_p(py)}" for px, py in ring) + " Z"
        parts.append(f'<path class="grid" d="{path}" fill="none"/>')
        parts.append(
            f'<text class="tick" x="{_p(center_x + 4)}" '
            f'y="{_p(center_y - radius * fraction + 4)}" text-anchor="start">'
            f"{int(fraction * 100)}%</text>"
        )
    for angle in angles:
        parts.append(
            f'<line class="grid" x1="{_p(center_x)}" y1="{_p(center_y)}" '
            f'x2="{_p(center_x + radius * math.cos(angle))}" '
            f'y2="{_p(center_y + radius * math.sin(angle))}"/>'
        )

    # 轴标签：按位置改锚点，避免压到图形
    for index, angle in enumerate(angles):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        label_x = center_x + (radius + 18) * cosine
        label_y = center_y + (radius + 18) * sine
        anchor = "start" if cosine > 0.2 else ("end" if cosine < -0.2 else "middle")
        dy = 4 if abs(sine) < 0.2 else (12 if sine > 0 else -4)
        parts.append(
            f'<text class="tick" x="{_p(label_x)}" y="{_p(label_y + dy)}" '
            f'text-anchor="{anchor}">{_esc(_trunc(axes[index], 12))}</text>'
        )

    for series_index, item in enumerate(data.series):
        color = colors[series_index]
        points = [
            (
                center_x
                + radius * normalize(item.values[index], index) * math.cos(angle),
                center_y
                + radius * normalize(item.values[index], index) * math.sin(angle),
            )
            for index, angle in enumerate(angles)
        ]
        path = "M " + " L ".join(f"{_p(px)} {_p(py)}" for px, py in points) + " Z"
        parts.append(
            f'<path class="series" data-series="{_esc(item.name)}" d="{path}" '
            f'fill="{color}" fill-opacity="0.13" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        for px, py in points:
            parts.append(
                f'<circle class="point" data-series="{_esc(item.name)}" '
                f'cx="{_p(px)}" cy="{_p(py)}" r="4" fill="{color}" '
                f'stroke="{theme["bg"]}" stroke-width="1"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts), width, height


def _render_small_multiples(data: ChartData, dark: bool) -> tuple[str, int, int]:
    """量纲/量级差异过大时的分面图：每个序列一个独立 Y 轴，并排放置。"""
    theme = _theme(dark)
    count = len(data.series)
    width = max(760, 300 * count)
    height = 430
    outer = 24
    gap = 18
    panel_width = (width - 2 * outer - gap * (count - 1)) / count
    colors = _colors(count, dark)

    parts = _svg_head(width, height, data.title, theme)
    parts.append(
        f'<text class="subtitle" x="{_p(width / 2)}" y="20" text-anchor="middle">'
        "各序列量纲或量级不同，分面展示（small multiples），不使用双 Y 轴"
        "</text>"
    )

    numeric_x = data.x_values is not None
    if numeric_x:
        x_values = data.x_values or []
        x_low, x_high, _x_step, x_ticks = _nice_axis(
            min(x_values), max(x_values), include_zero=False
        )

    for panel_index, item in enumerate(data.series):
        panel_x = outer + panel_index * (panel_width + gap)
        plot_left = panel_x + 48
        plot_top = 66
        plot_width = panel_width - 58
        plot_height = height - plot_top - 58
        color = colors[panel_index]

        parts.append(
            f'<text class="title" x="{_p(panel_x)}" y="44">'
            f"{_esc(_trunc(item.name, 18))}</text>"
        )

        low, high, _step, y_ticks = _nice_axis(
            min(item.values), max(item.values), include_zero=(data.kind == "bar")
        )

        def y(value: float, low=low, high=high) -> float:
            return plot_top + plot_height - (value - low) / (high - low) * plot_height

        if numeric_x:
            def x_index(index: int, panel=plot_left, x_values=x_values) -> float:
                return panel + (x_values[index] - x_low) / (x_high - x_low) * plot_width

            for tick in x_ticks:
                tick_x = plot_left + (tick - x_low) / (x_high - x_low) * plot_width
                parts.append(
                    f'<text class="tick" x="{_p(tick_x)}" y="{_p(plot_top + plot_height + 17)}" '
                    f'text-anchor="middle">{_esc(fmt_number(tick))}</text>'
                )
        else:
            label_count = len(data.labels)

            def x_index(index: int, panel=plot_left) -> float:
                return panel + plot_width * (index + 0.5) / label_count

            every = max(1, math.ceil(label_count / 6))
            for index, label in enumerate(data.labels):
                if index % every != 0 and index != label_count - 1:
                    continue
                parts.append(
                    f'<text class="tick" x="{_p(x_index(index))}" '
                    f'y="{_p(plot_top + plot_height + 17)}" text-anchor="middle">'
                    f"{_esc(_trunc(str(label), 8))}</text>"
                )

        for tick in y_ticks:
            tick_y = y(tick)
            parts.append(
                f'<line class="grid" x1="{_p(plot_left)}" y1="{_p(tick_y)}" '
                f'x2="{_p(plot_left + plot_width)}" y2="{_p(tick_y)}"/>'
            )
            parts.append(
                f'<text class="tick" x="{_p(plot_left - 7)}" y="{_p(tick_y + 4)}" '
                f'text-anchor="end">{_esc(fmt_number(tick))}</text>'
            )

        baseline = y(0) if data.kind == "bar" else plot_top + plot_height
        parts.append(
            f'<line class="axis" x1="{_p(plot_left)}" y1="{_p(plot_top)}" '
            f'x2="{_p(plot_left)}" y2="{_p(plot_top + plot_height)}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{_p(plot_left)}" y1="{_p(baseline)}" '
            f'x2="{_p(plot_left + plot_width)}" y2="{_p(baseline)}"/>'
        )
        if data.kind != "bar" and low < 0 < high:
            parts.append(
                f'<line class="baseline" x1="{_p(plot_left)}" y1="{_p(y(0))}" '
                f'x2="{_p(plot_left + plot_width)}" y2="{_p(y(0))}"/>'
            )

        # 单面板永远只有一个序列：按固定色板取色，不再叠加图例
        panel_series = [item]
        if data.kind == "bar":
            single = ChartData(
                kind="bar",
                title=data.title,
                x_label=data.x_label,
                y_label=data.y_label,
                labels=data.labels,
                series=panel_series,
            )
            _draw_bars(
                parts,
                single,
                [color],
                dark,
                theme,
                x_index,
                y,
                plot_width,
                plot_top,
            )
        else:
            single = ChartData(
                kind=data.kind,
                title=data.title,
                x_label=data.x_label,
                y_label=data.y_label,
                labels=data.labels,
                series=panel_series,
                x_values=data.x_values,
            )
            _draw_line_or_scatter(
                parts, single, [color], theme, x_index, y, plot_top
            )

        parts.append(
            f'<text class="axis-label" x="{_p(plot_left + plot_width / 2)}" '
            f'y="{_p(height - 12)}" text-anchor="middle">'
            f"{_esc(_trunc(data.x_label, 12))}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts), width, height


def render_chart(data: ChartData, *, dark: bool = False) -> dict[str, object]:
    """渲染成自包含 SVG，并返回前端可直接消费的元信息。

    返回的 `svg` 不引用任何外部字体或图片：`xmlns`、`viewBox`、内联
    `<style>` 都在字符串里，前端可以直接写入 `innerHTML`。
    """
    if data.kind == "radar":
        svg, width, height = _render_radar(data, dark)
    elif data.small_multiples:
        svg, width, height = _render_small_multiples(data, dark)
    else:
        svg, width, height = _render_xy(data, dark)

    legend = [item.name for item in data.series] if len(data.series) >= 2 else []
    return {
        "kind": data.kind,
        "svg": svg,
        "width": width,
        "height": height,
        "legend": legend,
        "x_label": data.x_label,
        "y_label": data.y_label,
        "note": data.note,
    }
