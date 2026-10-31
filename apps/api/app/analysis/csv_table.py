"""CSV 解析、类型推断与数值统计。

为什么不用 pandas：

1. 部署依赖刻意克制（见 pyproject.toml 的注释），pandas 会带进一个几十 MB
   的 wheel；本模块只需要"读表 + 少量统计"，标准库 `csv` + `statistics`
   足够。
2. pandas 对畸形输入太宽容 —— 它会把 `1,234` 读成字符串、把列数不一致
   的行默默补齐。比赛演示更希望得到**明确、可解释**的结果：哪一行坏了、
   哪一列被当成缺失值，都写进 `notes`，而不是让图悄悄少几个点。

这里刻意不做的事：

- 不持久化 CSV。无状态接口更省事，也没有"临时文件什么时候清理"的安全问题。
- 不支持 GBK。非 UTF-8 字节会得到明确的 `CsvDecodeError`，提示用户另存为
  UTF-8。悄悄用 gb18030 猜编码会把乱码变成看似正常的单元格，比报错更危险。
- 不排序行。CSV 的行序就是图的横轴顺序；擅自按数值排序会让时间序列失真。
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

# 单个单元格的展示值不需要无限精度；统计量统一保留 6 位小数，
# 既避免 0.30000000000000004 之类的噪声，也保证同一输入输出完全一致。
ROUND_DIGITS = 6

_MISSING_TOKENS = {
    "",
    "-",
    "--",
    "—",
    "–",
    "?",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "nil",
}

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE_RE = re.compile(
    r"^(?:"
    r"\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?(?:[ T].*)?"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
    r"|\d{4}年\d{1,2}月(?:\d{1,2}日)?"
    r"|(?:" + _MONTHS + r")[a-z]*[ .-]?\d{0,4}"
    r")$",
    re.IGNORECASE,
)


class CsvError(ValueError):
    """CSV 处理相关错误的基类。"""


class CsvDecodeError(CsvError):
    """字节无法按 UTF-8 解码，或明显是二进制内容。"""


class CsvFormatError(CsvError):
    """结构不成立：只有一行、列数不一致、引号不闭合等。"""


class CsvEmptyError(CsvError):
    """没有任何非空行。"""


class NoNumericColumnError(CsvError):
    """能解析，但找不到可绘制的数值列。"""


def parse_number(text: str) -> tuple[float, str] | None:
    """把单元格解析成 (数值, 单位)；不是数值就返回 None。

    支持的写法：
    - `1,234` / `1,234.5`（千分位，只接受规范的 3 位分组）
    - `12.5%` / `12.5％`（返回值仍是 12.5，单位记为 `%`）
    - `3‰`（单位记为 `‰`）
    - `(1,234)`（会计负数）
    - 科学计数法 `1.2e-3`

    不做的事：不把 `1,2` 当 12（那更像打错的分隔符），不把带单位的
    `3 kg` 硬拆成 3 —— 这类模糊写法一律当缺失，并记录在 notes 里。
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None
    # 全角符号与 Unicode 减号在真实 CSV 里很常见，先归一化
    s = (
        s.replace(" ", "")
        .replace("　", " ")
        .replace("−", "-")
        .replace("－", "-")
        .replace("％", "%")
        .replace("，", ",")
    )
    if s.lower() in _MISSING_TOKENS:
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    unit = ""
    if s.endswith("%"):
        unit = "%"
        s = s[:-1].strip()
    elif s.endswith("‰"):
        unit = "‰"
        s = s[:-1].strip()

    # 货币符号只影响展示，不改变量纲；统一剥掉，避免同一列被拆成两种单位
    for symbol in ("$", "¥", "￥", "€", "£"):
        if s.startswith(symbol):
            s = s[len(symbol) :].strip()
            break

    if not s:
        return None

    # 千分位必须是规范的 1-3 位 + 若干 3 位分组；否则视为非法
    if "," in s:
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
            return None
        s = s.replace(",", "")

    if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s):
        return None

    try:
        value = float(s)
    except ValueError:  # 理论上上面的正则已经挡住了
        return None
    if not math.isfinite(value):
        return None
    return (-value if negative else value), unit


def is_dateish(text: str) -> bool:
    """判断一个单元格是否"像日期"。只做形态匹配，不真正解析日历。"""
    s = str(text or "").strip()
    return bool(s) and bool(_DATE_RE.fullmatch(s))


def fmt_number(value: float | None) -> str:
    """紧凑、确定性的数值展示：12.30 -> 12.3，1.0 -> 1。"""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return ""
    if v == 0:
        return "0"
    s = f"{v:.6g}"
    if s in ("-0", "0"):
        return "0"
    return s


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


@dataclass(slots=True)
class Column:
    """一列的类型、单位与逐行取值（缺失为 None）。"""

    name: str
    kind: str  # number | date | category
    unit: str = ""
    missing: int = 0
    values: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class SeriesStats:
    """一个数值序列的统计卡。图注里允许出现的数字都来自这里或其派生量。"""

    name: str
    unit: str
    n: int
    missing: int
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "n": self.n,
            "missing": self.missing,
            "min": _round(self.min),
            "max": _round(self.max),
            "mean": _round(self.mean),
            "median": _round(self.median),
            "std": _round(self.std),
            "slope": _round(self.slope),
            "first": _round(self.first),
            "last": _round(self.last),
            "min_index": self.min_index,
            "max_index": self.max_index,
            "min_label": self.min_label,
            "max_label": self.max_label,
        }


@dataclass(slots=True)
class Table:
    """解析后的表：保留行序，数值列附带统计量。"""

    columns: list[Column]
    rows: list[list[Any]]
    stats: dict[str, SeriesStats]
    label_column: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == "number"]

    @property
    def category_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind in ("category", "date")]

    def column(self, name: str | None) -> Column | None:
        if name is None:
            return None
        for col in self.columns:
            if col.name == name:
                return col
        return None


def _strip_trailing_empty(row: list[str]) -> list[str]:
    end = len(row)
    while end > 0 and not row[end - 1].strip():
        end -= 1
    return row[:end]


def _detect_delimiter(text: str) -> str:
    """在 `, ; \\t` 里挑分隔符：优先"能切出更多列且每行列数一致"的。

    不用 `csv.Sniffer`：它对短表和小样本经常猜错，而且错误方式不稳定
    （同样的输入在不同 Python 版本上可能给出不同答案）。这里用固定的
    候选顺序和固定的打分规则，保证确定性。
    """
    lines = [line for line in text.splitlines() if line.strip()][:20]
    if not lines:
        return ","

    best: tuple[tuple[int, float, int, int], str] | None = None
    for index, delimiter in enumerate((",", ";", "\t")):
        try:
            rows = list(csv.reader(lines, delimiter=delimiter))
        except csv.Error:
            continue
        counts = [len(row) for row in rows if any(cell.strip() for cell in row)]
        if not counts:
            continue
        mode = max(set(counts), key=lambda value: (counts.count(value), value))
        consistency = counts.count(mode) / len(counts)
        # 排序键：先看是否真的分出了多列，再看一致性、列数，最后按候选顺序
        key = (1 if mode > 1 else 0, consistency, mode, -index)
        if best is None or key > best[0]:
            best = (key, delimiter)
    return best[1] if best is not None else ","


def _looks_like_header(first: list[str], body: list[list[str]]) -> bool:
    """表头识别的核心：比较第一行与其余行的"类型是否突变"。

    典型数据 `label,value / A,1 / B,2`：第二列第一行是文字、下面全是数字，
    这强烈说明第一行是表头。无表头的 `A,1 / B,2` 则没有任何一列发生这种
    突变，按无表头处理。宁可少认表头，也不要吞掉第一行真实数据。
    """
    if not body:
        return False

    signals = 0
    for j, raw_first in enumerate(first):
        first_value = raw_first.strip()
        body_values = [
            row[j].strip()
            for row in body
            if j < len(row) and row[j].strip()
        ]
        if not body_values:
            # 整列为空但表头有字：更像表头
            if first_value:
                signals += 1
            continue

        first_is_number = parse_number(first_value) is not None
        number_ratio = sum(
            1 for value in body_values if parse_number(value) is not None
        ) / len(body_values)
        first_is_date = is_dateish(first_value)
        date_ratio = sum(1 for value in body_values if is_dateish(value)) / len(
            body_values
        )

        if not first_value:
            signals += 1
        elif not first_is_number and number_ratio >= 0.6:
            signals += 1
        elif not first_is_date and date_ratio >= 0.6:
            signals += 1
        elif first_is_number and number_ratio < 0.2:
            signals += 1

    return signals >= 1


def _unique_names(names: Sequence[str]) -> list[str]:
    """空列名补成 `列N`，重名加后缀，避免 JSON 对象键互相覆盖。"""
    result: list[str] = []
    used: Counter[str] = Counter()
    for index, raw in enumerate(names):
        name = (raw or "").strip() or f"列{index + 1}"
        if name in used:
            used[name] += 1
            candidate = f"{name}_{used[name]}"
            while candidate in used:
                used[name] += 1
                candidate = f"{name}_{used[name]}"
            name = candidate
        used[name] += 1
        result.append(name)
    return result


def _build_column(name: str, raw_values: list[str], notes: list[str]) -> Column:
    nonempty = [value.strip() for value in raw_values if value.strip()]
    missing = len(raw_values) - len(nonempty)

    parsed = [(value, parse_number(value)) for value in nonempty]
    numeric_count = sum(1 for _value, number in parsed if number is not None)
    numeric_ratio = numeric_count / len(nonempty) if nonempty else 0.0
    date_ratio = (
        sum(1 for value in nonempty if is_dateish(value)) / len(nonempty)
        if nonempty
        else 0.0
    )

    if nonempty and numeric_ratio >= 0.8:
        kind = "number"
        values: list[Any] = []
        units: list[str] = []
        skipped = 0
        for value in raw_values:
            number = parse_number(value)
            if number is None:
                values.append(None)
                if value.strip():
                    skipped += 1
            else:
                values.append(number[0])
                units.append(number[1])
        unit = ""
        if units:
            unit = sorted(Counter(units).items(), key=lambda item: (-item[1], item[0]))[
                0
            ][0]
            if len(set(units)) > 1:
                notes.append(
                    f"列「{name}」存在多种单位（{'、'.join(sorted(set(units)))}），"
                    f"按最多的「{unit or '无单位'}」展示。"
                )
        if skipped:
            notes.append(f"列「{name}」有 {skipped} 个非数值单元格按缺失值处理。")
        return Column(
            name=name,
            kind=kind,
            unit=unit,
            missing=len(raw_values) - sum(1 for value in values if value is not None),
            values=values,
        )

    kind = "date" if nonempty and date_ratio >= 0.8 else "category"
    scalar_values = [value.strip() or None for value in raw_values]
    return Column(
        name=name,
        kind=kind,
        unit="",
        missing=sum(1 for value in scalar_values if value is None),
        values=scalar_values,
    )


def describe(
    values: Sequence[float],
    *,
    name: str = "",
    unit: str = "",
    labels: Sequence[str] | None = None,
    missing: int = 0,
) -> SeriesStats:
    """一个数值序列的统计量。空序列给出全 0 的卡片，而不是抛异常。"""
    vals = [float(value) for value in values]
    n = len(vals)
    if labels is None or len(labels) != n:
        labels = [str(index + 1) for index in range(n)]

    if n == 0:
        return SeriesStats(
            name=name,
            unit=unit,
            n=0,
            missing=missing,
            min=0.0,
            max=0.0,
            mean=0.0,
            median=0.0,
            std=0.0,
            slope=0.0,
            first=0.0,
            last=0.0,
            min_index=0,
            max_index=0,
            min_label="",
            max_label="",
        )

    minimum = min(vals)
    maximum = max(vals)
    min_index = vals.index(minimum)
    max_index = vals.index(maximum)
    mean = statistics.fmean(vals)
    median = statistics.median(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0

    slope = 0.0
    if n > 1:
        xs = list(range(n))
        mean_x = statistics.fmean(xs)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator:
            slope = sum(
                (x - mean_x) * (y - mean) for x, y in zip(xs, vals)
            ) / denominator

    return SeriesStats(
        name=name,
        unit=unit,
        n=n,
        missing=missing,
        min=_round(minimum),
        max=_round(maximum),
        mean=_round(mean),
        median=_round(median),
        std=_round(std),
        slope=_round(slope),
        first=_round(vals[0]),
        last=_round(vals[-1]),
        min_index=min_index + 1,
        max_index=max_index + 1,
        min_label=str(labels[min_index]),
        max_label=str(labels[max_index]),
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """皮尔逊相关系数；任一侧方差为 0 时返回 0（不产生 NaN）。"""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_vals = [float(value) for value in xs]
    y_vals = [float(value) for value in ys]
    mean_x = statistics.fmean(x_vals)
    mean_y = statistics.fmean(y_vals)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)
    )
    sum_x = sum((x - mean_x) ** 2 for x in x_vals)
    sum_y = sum((y - mean_y) ** 2 for y in y_vals)
    denominator = math.sqrt(sum_x * sum_y)
    if denominator == 0:
        return 0.0
    return _round(numerator / denominator)


def parse_csv(data: bytes) -> Table:
    """把 CSV 字节解析成 `Table`。

    失败时抛出明确异常，由路由层翻译成 400/422：
    - `CsvDecodeError`：非 UTF-8 / 含 NUL 字节
    - `CsvFormatError`：只有一行、列数不一致、引号不闭合
    - `CsvEmptyError`：全空
    - `NoNumericColumnError`：没有可用于绘图的数值列
    """
    if not data or not data.strip():
        raise CsvEmptyError("CSV 文件是空的")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvDecodeError(
            "文件不是 UTF-8 编码。请在表格软件里另存为「CSV UTF-8」后重新上传。"
        ) from exc

    if "\x00" in text:
        raise CsvDecodeError("文件包含 NUL 字节，看起来不是文本 CSV。")

    delimiter = _detect_delimiter(text)
    rows: list[list[str]] = []
    try:
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter=delimiter,
            quotechar='"',
            skipinitialspace=True,
            strict=True,
        )
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue  # 空行直接跳过，不计入行数
            rows.append(_strip_trailing_empty(row))
    except csv.Error as exc:
        raise CsvFormatError(f"CSV 语法错误（引号可能没有闭合）：{exc}") from exc

    if not rows:
        raise CsvEmptyError("CSV 里没有任何非空行")
    if len(rows) < 2:
        raise CsvFormatError(
            "CSV 至少需要表头和一行数据；当前只有一行，无法推断图表。"
        )

    header_candidate = rows[0]
    body = rows[1:]
    has_header = _looks_like_header(header_candidate, body)

    if has_header:
        names = _unique_names(header_candidate)
        body_rows = body
    else:
        names = _unique_names([f"列{index + 1}" for index in range(len(header_candidate))])
        body_rows = rows

    ncols = len(names)
    notes: list[str] = []
    padded = 0
    normalized: list[list[str]] = []
    for row_index, row in enumerate(body_rows, start=1):
        if len(row) > ncols:
            raise CsvFormatError(
                f"第 {row_index + (1 if has_header else 0)} 行有 {len(row)} 列，"
                f"表头只有 {ncols} 列。请检查分隔符或缺失的引号。"
            )
        if len(row) < ncols:
            row = row + [""] * (ncols - len(row))
            padded += 1
        normalized.append(row)

    if padded:
        notes.append(f"有 {padded} 行列数少于表头，缺失单元格按空值补齐。")
    if not has_header:
        notes.append("没有识别到表头，列名按「列1、列2…」生成。")
    if delimiter != ",":
        shown = "制表符" if delimiter == "\t" else delimiter
        notes.append(f"检测到分隔符「{shown}」。")

    columns = [
        _build_column(name, [row[index] for row in normalized], notes)
        for index, name in enumerate(names)
    ]

    label_column = next(
        (column.name for column in columns if column.kind in ("category", "date")),
        None,
    )
    label_index = (
        next(index for index, column in enumerate(columns) if column.name == label_column)
        if label_column is not None
        else None
    )

    def row_label(row_index: int) -> str:
        if label_index is None:
            return f"行{row_index + 1}"
        value = normalized[row_index][label_index].strip()
        return value or f"行{row_index + 1}"

    stats: dict[str, SeriesStats] = {}
    for column in columns:
        if column.kind != "number":
            continue
        positions = [
            index for index, value in enumerate(column.values) if value is not None
        ]
        stats[column.name] = describe(
            [column.values[index] for index in positions],
            name=column.name,
            unit=column.unit,
            labels=[row_label(index) for index in positions],
            missing=column.missing,
        )

    if not stats:
        raise NoNumericColumnError(
            "没有识别到数值列。请确认 CSV 中至少有一列是数字（"
            "支持 1,234 与 12.5% 这类写法），或检查分隔符是否正确。"
        )

    return Table(
        columns=columns,
        rows=normalized,
        stats=stats,
        label_column=label_column,
        notes=notes,
    )
