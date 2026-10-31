"""图表源码导出：Mermaid / TikZ(pgfplots) / Matplotlib 脚本。

三种导出的共同底线：**数据必须与 SVG 来自同一份 `ChartData`**。
因此这里不重新读 CSV、不重新做类型推断、不再剔除缺失值 —— 点数不一致
是这类功能最容易被一眼看穿的 bug（"图里 7 个点、TikZ 里 6 个点"）。

各格式的能力边界不同，处理方式是"降级但要能编译/渲染"：

- Mermaid：没有原生散点图，用 `quadrantChart` 降级；没有雷达图，按各轴
  归一化 ×100 后降级为分组柱状图。降级原因写在注释里，读者不会误以为
  原图就是柱状图。
- TikZ：雷达图用"归一化后的分组柱状图"降级，保证 `pgfplots` 能直接编译。
- Matplotlib：雷达图用极坐标子图真实还原；small multiples 用并排子图，
  同样不制造双 Y 轴。
"""

from __future__ import annotations

import math
from typing import Sequence

from app.analysis.chart import ChartData, Series, fmt_number

_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def _tex(text: object) -> str:
    return "".join(_TEX_SPECIAL.get(char, char) for char in str(text))


def _tex_num(value: float) -> str:
    """pgfplots 对科学计数法的解析依赖版本，统一写成普通小数。"""
    number = float(value)
    if not math.isfinite(number):
        return "0"
    if number != 0 and abs(number) < 1e-4:
        text = f"{number:.10f}"
    else:
        text = f"{number:.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _mm(text: object) -> str:
    """Mermaid 的带引号字符串：转义反斜杠与双引号，压掉换行。"""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


def _mm_label(text: object, limit: int = 24) -> str:
    value = str(text)
    if len(value) > limit:
        value = value[: limit - 1] + "…"
    return _mm(value)


def _range(values: Sequence[float], include_zero: bool = False) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if high == low:
        pad = abs(high) * 0.1 or 1.0
        low -= pad
        high += pad
    return low, high


def _radar_values(chart: ChartData) -> list[Series]:
    """把雷达图每个轴归一化到 0..1，供降级导出使用。"""
    if not chart.series:
        return []
    axis_count = len(chart.labels)
    lows = [min(item.values[j] for item in chart.series) for j in range(axis_count)]
    highs = [max(item.values[j] for item in chart.series) for j in range(axis_count)]
    result: list[Series] = []
    for item in chart.series:
        values = []
        for axis in range(axis_count):
            span = highs[axis] - lows[axis]
            values.append(
                0.5
                if span == 0
                else (item.values[axis] - lows[axis]) / span
            )
        result.append(Series(name=item.name, values=values, unit=""))
    return result


def _pie_suitable(chart: ChartData) -> bool:
    """只有"标题明确说占比"的单序列柱图才自动转饼图。

    普通柱状图并不等于"部分与整体"；把它们画成饼图是统计意义上常见的
    误导，所以这里刻意保守。需要饼图可以显式 `style="pie"`。
    """
    if chart.kind != "bar" or len(chart.series) != 1:
        return False
    values = chart.series[0].values
    if any(value < 0 for value in values) or sum(values) <= 0:
        return False
    return any(word in chart.title for word in ("占比", "比例", "份额"))


# ================================================================== Mermaid


def _mermaid_xy(chart: ChartData, series: Sequence[Series] | None = None) -> list[str]:
    items = list(series) if series is not None else list(chart.series)
    lines = [
        "xychart-beta",
        f'    title "{_mm(chart.title)}"',
        "    x-axis ["
        + ", ".join(f'"{_mm_label(label)}"' for label in chart.labels)
        + "]",
    ]
    all_values = [value for item in items for value in item.values]
    low, high = _range(all_values, include_zero=chart.kind == "bar")
    lines.append(
        f'    y-axis "{_mm(chart.y_label)}" '
        f"{fmt_number(low)} --> {fmt_number(high)}"
    )
    keyword = "bar" if chart.kind == "bar" else "line"
    for item in items:
        values = ", ".join(fmt_number(value) for value in item.values)
        lines.append(f"    {keyword} [{values}]")
    return lines


def _mermaid_pie(chart: ChartData) -> str:
    item = chart.series[0]
    lines = ["pie showData", f'    title "{_mm(chart.title)}"']
    for label, value in zip(chart.labels, item.values):
        lines.append(f'    "{_mm_label(label)}" : {fmt_number(value)}')
    return "\n".join(lines)


def _mermaid_scatter(chart: ChartData) -> str:
    x_values = chart.x_values or []
    low_x, high_x = _range(x_values)
    lines = [
        "%% Mermaid 没有原生散点图，这里用 quadrantChart 降级；"
        "坐标已按各自范围归一化到 0..1"
    ]
    for item in chart.series:
        low_y, high_y = _range(item.values)
        lines.append("quadrantChart")
        lines.append(f'    title "{_mm(chart.title)}"')
        lines.append(
            f'    x-axis "{_mm(chart.x_label)}" '
            f"{fmt_number(low_x)} --> {fmt_number(high_x)}"
        )
        lines.append(
            f'    y-axis "{_mm(item.name)}" '
            f"{fmt_number(low_y)} --> {fmt_number(high_y)}"
        )
        lines.append('    quadrant-1 "高"')
        lines.append('    quadrant-2 "中"')
        lines.append('    quadrant-3 "低"')
        lines.append('    quadrant-4 "中"')
        for index, value in enumerate(item.values):
            raw_x = x_values[index] if index < len(x_values) else index
            norm_x = (
                0.5
                if high_x == low_x
                else (raw_x - low_x) / (high_x - low_x)
            )
            norm_y = (
                0.5
                if high_y == low_y
                else (value - low_y) / (high_y - low_y)
            )
            lines.append(
                f'    "{_mm_label(chart.labels[index])}": '
                f"[{norm_x:.4f}, {norm_y:.4f}]"
            )
    return "\n".join(lines)


def _mermaid_radar(chart: ChartData) -> str:
    normalized = _radar_values(chart)
    lines = [
        "%% Mermaid 没有原生雷达图：按各轴归一化 ×100 后降级为分组柱状图。",
        "%% 若当前 Mermaid 版本只渲染第一条 bar，请升级或使用 TikZ 导出。",
        "xychart-beta",
        f'    title "{_mm(chart.title)}（归一化）"',
        "    x-axis ["
        + ", ".join(f'"{_mm_label(axis)}"' for axis in chart.labels)
        + "]",
        '    y-axis "归一化值 (0-100)" 0 --> 100',
    ]
    for item in normalized:
        values = ", ".join(f"{value * 100:.1f}" for value in item.values)
        lines.append(f"    bar [{values}]")
    return "\n".join(lines)


def _mermaid_unknown(chart: ChartData) -> str:
    first = chart.series[:1]
    lines = [
        f"%% 不支持的图表类型「{chart.kind}」，已降级为第一条序列的折线图。"
    ]
    preview = ChartData(
        kind="line",
        title=chart.title,
        x_label=chart.x_label,
        y_label=chart.y_label,
        labels=chart.labels,
        series=first,
    )
    lines.extend(_mermaid_xy(preview, first))
    return "\n".join(lines)


def to_mermaid(chart: ChartData, *, style: str = "auto") -> str:
    """导出 Mermaid 源码。

    `style`：`auto` 按图表类型选择；`pie` 显式要求饼图；`xychart` 强制
    柱/折线。不支持的组合会给出可用的降级写法，而不是抛异常。
    """
    style = (style or "auto").strip().lower()
    if chart.small_multiples:
        lines = [
            "%% small multiples：以下每段是一张独立图表，请分别粘贴到 Mermaid 渲染器。"
        ]
        for item in chart.series:
            lines.append("")
            lines.append(f"%% 分面：{_mm(item.name)}")
            lines.extend(_mermaid_xy(chart, [item]))
        return "\n".join(lines)

    if style == "pie":
        if chart.kind == "bar" and len(chart.series) == 1:
            return _mermaid_pie(chart)
        return (
            "%% 饼图只支持单序列、非负的柱状图；当前数据已降级为分组柱状图。\n"
            + "\n".join(_mermaid_xy(chart))
        )
    if style == "auto" and _pie_suitable(chart):
        return _mermaid_pie(chart)

    if chart.kind == "scatter":
        return _mermaid_scatter(chart)
    if chart.kind == "radar":
        return _mermaid_radar(chart)
    if chart.kind in ("line", "bar"):
        prefix = ""
        if len(chart.series) > 1:
            prefix = (
                "%% 多序列：较新的 Mermaid 版本支持重复 bar/line；"
                "若只显示第一条，请升级 Mermaid 或改用 TikZ 导出。\n"
            )
        return prefix + "\n".join(_mermaid_xy(chart))
    return _mermaid_unknown(chart)


# ===================================================================== TikZ


class _AxisContext:
    """一个 pgfplots `axis` 环境需要的全部信息。"""

    def __init__(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        labels: Sequence[str],
        series: Sequence[Series],
        kind: str,
        x_values: Sequence[float] | None = None,
    ) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.labels = list(labels)
        self.series = list(series)
        self.kind = kind
        self.x_values = list(x_values) if x_values else None


def _tikz_axis_block(
    context: _AxisContext,
    *,
    color_offset: int = 0,
    include_legend: bool = True,
) -> list[str]:
    labels = context.labels
    series = context.series
    values = [value for item in series for value in item.values]
    low, high = _range(values, include_zero=context.kind == "bar")
    count = len(labels)
    series_count = len(series)

    options = [
        "axis lines=left",
        "width=12.6cm",
        "height=7.4cm",
        f"title={{{_tex(context.title)}}}",
        f"xlabel={{{_tex(context.x_label)}}}",
        f"ylabel={{{_tex(context.y_label)}}}",
        f"ymin={_tex_num(low)}",
        f"ymax={_tex_num(high)}",
        "grid=major",
        "grid style={gray!20}",
        "axis line style={gray!60}",
        "tick label style={font=\\footnotesize}",
        "label style={font=\\small}",
        "legend style={font=\\footnotesize, draw=none}",
        "legend pos=north east",
    ]

    # 折线图如果横轴是真实数值（如 epoch），导出也用真实 x，而不是 0..n-1，
    # 否则非等距横轴会被悄悄改成等距类别轴。
    numeric_x = context.x_values is not None and context.kind in ("line", "scatter")
    if numeric_x:
        x_values = context.x_values or list(range(count))
        x_low, x_high = _range(x_values)
        options.extend([f"xmin={_tex_num(x_low)}", f"xmax={_tex_num(x_high)}"])
    else:
        options.extend(
            [
                "xmin=-0.5",
                f"xmax={_tex_num(max(0, count - 0.5))}",
                "xtick={"
                + ",".join(str(index) for index in range(count))
                + "}",
                # 每个标签单独加大括号：标签里含逗号时不会把 xticklabels 拆坏
                "xticklabels={"
                + ",".join("{" + _tex(label) + "}" for label in labels)
                + "}",
            ]
        )
        if context.kind == "bar":
            options.append("enlarge x limits=0.08")

    lines = ["\\begin{tikzpicture}", "\\begin{axis}[", "  " + ",\n  ".join(options) + ",", "]"]

    for series_index, item in enumerate(series):
        color = f"dv{color_offset + series_index + 1}"
        if numeric_x:
            x_values = context.x_values or list(range(count))
            coordinates = " ".join(
                f"({_tex_num(x_values[index])},{_tex_num(value)})"
                for index, value in enumerate(item.values)
            )
            plot_options = [
                f"color={color}",
                "mark=*",
                "mark size=1.6pt",
                "line width=0.8pt",
            ]
            if context.kind == "scatter":
                plot_options = [
                    f"color={color}",
                    "only marks",
                    "mark=*",
                    "mark size=2pt",
                ]
        elif context.kind == "bar":
            coordinates = " ".join(
                f"({index},{_tex_num(value)})"
                for index, value in enumerate(item.values)
            )
            if series_count > 1:
                step = 8.0
                shift = (series_index - (series_count - 1) / 2) * step
                bar_width = 6.0
            else:
                shift = 0.0
                bar_width = 10.0
            plot_options = [
                "ybar",
                f"bar width={bar_width}pt",
                f"bar shift={shift}pt",
                f"fill={color}",
                "draw=none",
            ]
        else:
            coordinates = " ".join(
                f"({index},{_tex_num(value)})"
                for index, value in enumerate(item.values)
            )
            plot_options = [
                f"color={color}",
                "mark=*",
                "mark size=1.6pt",
                "line width=0.8pt",
            ]

        lines.append(
            "\\addplot+[" + ", ".join(plot_options) + f"] coordinates {{{coordinates}}};"
        )
        if include_legend and series_count > 1:
            lines.append(f"\\addlegendentry{{{_tex(item.name)}}}")

    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    return lines


def _tikz_preamble(series_count: int) -> list[str]:
    lines = [
        "% 由数据可视化接口生成，可直接粘贴进 LaTeX。",
        "% 需要 \\usepackage{pgfplots}；含中文标签时请用 XeLaTeX/LuaLaTeX + ctex。",
    ]
    palette = (
        "2F6DB3",
        "D9772F",
        "2F8F6B",
        "A0446F",
        "7A5AA6",
        "8A6D1F",
        "3F7F8F",
        "6B6B6B",
    )
    for index in range(min(series_count, len(palette))):
        lines.append(
            f"\\definecolor{{dv{index + 1}}}{{HTML}}{{{palette[index]}}}"
        )
    return lines


def to_tikz(chart: ChartData) -> str:
    """导出 pgfplots 的 `axis` 环境，数据以 `coordinates` 内联。"""
    if chart.kind == "radar":
        normalized = _radar_values(chart)
        context = _AxisContext(
            title=chart.title + "（各轴归一化，雷达图降级为分组柱状图）",
            x_label="指标",
            y_label="归一化值",
            labels=chart.labels,
            series=normalized,
            kind="bar",
        )
        lines = [
            "% 雷达图在 pgfplots 里没有稳定通用的原生写法；",
            "% 这里按各轴归一化到 0..1 后降级为分组柱状图，保证可直接编译。",
        ]
        lines.extend(_tikz_preamble(len(normalized)))
        lines.extend(_tikz_axis_block(context))
        return "\n".join(lines)

    lines = _tikz_preamble(len(chart.series))
    if chart.small_multiples:
        lines.append("% 量纲/量级差异过大：每个序列一个独立 Y 轴，未使用双 Y 轴。")
        for index, item in enumerate(chart.series):
            lines.append("")
            lines.append(f"% 分面：{_tex(item.name)}")
            context = _AxisContext(
                title=item.name,
                x_label=chart.x_label,
                y_label=item.name,
                labels=chart.labels,
                series=[item],
                kind=chart.kind,
                x_values=chart.x_values,
            )
            lines.extend(
                _tikz_axis_block(
                    context, color_offset=index, include_legend=False
                )
            )
        return "\n".join(lines)

    context = _AxisContext(
        title=chart.title,
        x_label=chart.x_label,
        y_label=chart.y_label,
        labels=chart.labels,
        series=chart.series,
        kind=chart.kind,
        x_values=chart.x_values,
    )
    lines.extend(_tikz_axis_block(context))
    return "\n".join(lines)


# =============================================================== Matplotlib


def _py_literal(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_py_literal(item) for item in value) + "]"
    return repr(value)


def _py_series(series: Sequence[Series]) -> str:
    blocks = []
    for item in series:
        blocks.append(
            "{"
            f'"name": {_py_literal(item.name)}, '
            f'"values": {_py_literal(item.values)}, '
            f'"unit": {_py_literal(item.unit)}'
            "}"
        )
    return "[\n    " + ",\n    ".join(blocks) + ",\n]"


def to_matplotlib_script(chart: ChartData) -> str:
    """生成一段可独立运行的 Matplotlib 脚本。

    数据以字面量内联，脚本不会回头读原 CSV —— 只有这样导出文件才是
    可复现的：把脚本发给合作者，他不需要你本地的数据文件。
    """
    palette = (
        "#2f6db3",
        "#d9772f",
        "#2f8f6b",
        "#a0446f",
        "#7a5aa6",
        "#8a6d1f",
        "#3f7f8f",
        "#6b6b6b",
    )
    labels = chart.labels
    series = chart.series
    x_values = chart.x_values or []
    kind = chart.kind
    small = chart.small_multiples

    header = [
        '"""由数据可视化接口生成的独立绘图脚本。',
        "",
        "数据已作为字面量内联，不依赖原始 CSV；直接运行即可复现同一张图。",
        "生成后可用 `python chart.py` 运行，输出 chart.png。",
        '"""',
        "import math",
        "import matplotlib.pyplot as plt",
        "",
        f"TITLE = {_py_literal(chart.title)}",
        f"X_LABEL = {_py_literal(chart.x_label)}",
        f"Y_LABEL = {_py_literal(chart.y_label)}",
        f"LABELS = {_py_literal(labels)}",
        f"X_VALUES = {_py_literal([float(value) for value in x_values])}",
        f"SERIES = {_py_series(series)}",
        f"KIND = {_py_literal(kind)}",
        f"SMALL_MULTIPLES = {'True' if small else 'False'}",
        f"PALETTE = {_py_literal(palette)}",
        "",
        "",
        "def _style(ax):",
        '    ax.spines["top"].set_visible(False)',
        '    ax.spines["right"].set_visible(False)',
        '    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8, alpha=0.9)',
        "    ax.set_axisbelow(True)",
        "    ax.tick_params(labelsize=11)",
        "",
        "",
        "def _set_category_ticks(ax, labels):",
        "    ax.set_xticks(range(len(labels)))",
        "    ax.set_xticklabels(labels)",
        "",
        "",
        "def _xs():",
        "    return X_VALUES if X_VALUES else list(range(len(LABELS)))",
        "",
        "",
        "def _draw(ax, item, color):",
        "    xs = _xs()",
        '    values = item["values"]',
        '    if KIND == "line":',
        '        ax.plot(xs, values, color=color, linewidth=2, marker="o",',
        '                markersize=4, label=item["name"])',
        '    elif KIND == "scatter":',
        '        ax.scatter(xs, values, s=28, color=color, label=item["name"])',
        '    elif KIND == "bar":',
        '        ax.bar(xs, values, color=color, width=0.6, label=item["name"])',
        "    else:",
        '        ax.plot(xs, values, color=color, linewidth=2, label=item["name"])',
        '    if KIND in ("line", "bar") and not X_VALUES:',
        "        _set_category_ticks(ax, LABELS)",
        "    _style(ax)",
        "",
        "",
        "def _draw_grouped_bars(ax):",
        "    count = len(LABELS)",
        "    series_count = len(SERIES)",
        "    width = 0.8 / series_count",
        "    for index, item in enumerate(SERIES):",
        "        offsets = [i + (index - (series_count - 1) / 2) * width",
        "                   for i in range(count)]",
        '        ax.bar(offsets, item["values"], width=width * 0.9,',
        '               color=PALETTE[index], label=item["name"])',
        "    _set_category_ticks(ax, LABELS)",
        "    _style(ax)",
        "",
        "",
        "def _draw_radar(fig):",
        "    ax = fig.add_subplot(111, polar=True)",
        "    axes = LABELS",
        "    angles = [2 * math.pi * i / len(axes) for i in range(len(axes))]",
        "    lows = [min(item[\"values\"][j] for item in SERIES)",
        "            for j in range(len(axes))]",
        "    highs = [max(item[\"values\"][j] for item in SERIES)",
        "             for j in range(len(axes))]",
        "    for index, item in enumerate(SERIES):",
        "        values = []",
        "        for j in range(len(axes)):",
        "            span = highs[j] - lows[j]",
        "            values.append(0.5 if span == 0 else",
        "                          (item[\"values\"][j] - lows[j]) / span)",
        "        closed = values + [values[0]]",
        "        closed_angles = angles + [angles[0]]",
        "        ax.plot(closed_angles, closed, color=PALETTE[index],",
        "                linewidth=2, marker=\"o\", markersize=4,",
        "                label=item[\"name\"])",
        "        ax.fill(closed_angles, closed, color=PALETTE[index], alpha=0.13)",
        "    ax.set_xticks(angles)",
        "    ax.set_xticklabels(axes, fontsize=11)",
        "    ax.set_ylim(0, 1.05)",
        "    ax.set_yticks([0.25, 0.5, 0.75, 1.0])",
        "    ax.set_yticklabels([\"25%\", \"50%\", \"75%\", \"100%\"], fontsize=9)",
        "    if len(SERIES) > 1:",
        "        ax.legend(frameon=False, fontsize=10, loc=\"upper right\",",
        "                  bbox_to_anchor=(1.18, 1.12))",
        "",
        "",
        "def main():",
        "    plt.rcParams.update({",
        '        "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei",',
        '                           "SimHei", "DejaVu Sans"],',
        '        "axes.unicode_minus": False,',
        "    })",
        '    if KIND == "radar":',
        "        fig = plt.figure(figsize=(7.0, 6.2))",
        "        _draw_radar(fig)",
        "        fig.suptitle(TITLE, fontsize=14)",
        "    elif SMALL_MULTIPLES:",
        "        fig, axes = plt.subplots(1, len(SERIES),",
        "                                 figsize=(3.6 * len(SERIES), 4.4))",
        "        if len(SERIES) == 1:",
        "            axes = [axes]",
        "        for index, (ax, item) in enumerate(zip(axes, SERIES)):",
        "            _draw(ax, item, PALETTE[index])",
        "            ax.set_title(item[\"name\"], fontsize=11)",
        "        fig.suptitle(TITLE, fontsize=13)",
        "    else:",
        "        fig, ax = plt.subplots(figsize=(8.0, 4.6))",
        '        if KIND == "bar":',
        "            _draw_grouped_bars(ax)",
        "        else:",
        "            for index, item in enumerate(SERIES):",
        "                _draw(ax, item, PALETTE[index])",
        '        ax.set_xlabel(X_LABEL)',
        '        ax.set_ylabel(Y_LABEL)',
        "        if len(SERIES) > 1:",
        '            ax.legend(frameon=False, fontsize=10)',
        "        fig.suptitle(TITLE, fontsize=13)",
        "    fig.tight_layout()",
        '    fig.savefig("chart.png", dpi=150)',
        "    plt.show()",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ]
    return "\n".join(header)
