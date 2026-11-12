"""公式块识别：只认有编号的独立公式，认出的是"在哪儿、第几个"，不是 LaTeX。

两组用例：

1. **判据本身**（不起 PDF）：拿几个手工摆的 `_Line` 去撞 `_formula_at`，
   把"编号不在右边缘""窗口里混进图注""数学字体占比不够""另一栏的行被算进来"
   这四种打回逐条钉住。这些是逻辑，跟具体某篇论文无关。
2. **七份样本**（要 `pdf/` 里的文件）：Transformer 的两版都是式 (1)(2)(3)，
   扫描件与 BERT 必须是"没有"而不是"解析失败"。

第二组真正守的不是"今天认出 3 个"，而是 `note` 说的那句话得是真的：
BERT、GPT-3、GPT-4 一条编号公式都没有（全篇搜不到孤立的 `(N)` 行），
所以它们的 0 条不是检测失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.formulas import (
    MATH_FONT,
    NUMBER,
    FormulaReport,
    _formula_at,
    _Line,
    analyze,
)

PDF_DIR = Path(__file__).resolve().parents[3] / "pdf"
TRANSFORMER_V7 = PDF_DIR / "1706.03762v7.pdf"
TRANSFORMER_V1 = PDF_DIR / "1706.03762v1.pdf"
TRANSFORMER_SCAN = PDF_DIR / "1706.03762v1_scan.pdf"
BERT = PDF_DIR / "1810.04805_BERT.pdf"
UBD = PDF_DIR / "UnderstandingDeepLearning_02_09_26_C.pdf"


def _line(x0: float, x1: float, y0: float, y1: float, text: str,
          math_chars: int, chars: int | None = None) -> _Line:
    return _Line(bbox=(x0, y0, x1, y1), text=text,
                 math_chars=math_chars, chars=len(text.replace(" ", "")) if chars is None else chars)


# ------------------------------------------------------------------ 判据

def test_numbered_formula_at_the_right_edge_is_recognised():
    number = _line(493.1, 504.7, 474.0, 482.9, "(1)", 0)
    body = _line(220.0, 389.8, 465.6, 491.1, "Attention(Q, K, V ) = softmax(QKT)", 13, 13)
    # 正文行定义栏目左边缘，也顺便证明"正文不会被当成公式体"
    para = _line(108.0, 505.7, 500.0, 510.0, "The two most commonly used attention functions are", 0, 54)

    found = _formula_at(number, [number, body, para], right_edges=[505.7])
    assert found is not None
    assert found.number == "1"
    assert found.label == "式 (1)"
    assert found.bbox == body.bbox
    assert found.math_ratio == 1.0


def test_number_away_from_the_column_edge_is_not_a_formula():
    """参考文献里也有 `(1)` 这样的行；不在栏目右边缘的就不是公式编号。"""
    number = _line(300.0, 311.0, 474.0, 482.9, "(1)", 0)
    body = _line(220.0, 290.0, 465.6, 491.1, "softmax(QK)", 10)
    assert _formula_at(number, [number, body], right_edges=[505.7]) is None


def test_caption_inside_the_window_disqualifies_the_candidate():
    """表头里的 `(1)` 是脚注标记，不是公式 —— 窗口里有图注/表注就不认。"""
    number = _line(493.1, 504.7, 474.0, 482.9, "(1)", 0)
    caption = _line(220.0, 389.8, 470.0, 480.0, "Table 2.1 Sizes and architectures", 12)
    assert _formula_at(number, [number, caption], right_edges=[505.7]) is None


def test_low_math_ratio_is_rejected():
    """一整行正文正好挨着编号时，不能把它当成公式体。"""
    number = _line(493.1, 504.7, 474.0, 482.9, "(1)", 0)
    prose = _line(108.0, 480.0, 470.0, 480.0,
                  "and the choice of parameters determines the model", 1)
    assert _formula_at(number, [number, prose], right_edges=[505.7]) is None


def test_lines_from_the_other_column_are_not_swallowed():
    """两栏页面：编号在右栏，左栏同一高度的那一行不能被吞进公式体。

    UDL 的式 2.3 一开始就是被这一行害的（框从 45 一直拉到 366）。
    """
    number = _line(494.9, 506.4, 470.0, 479.0, "(2.3)", 0)
    right_column = _line(300.0, 366.6, 468.0, 480.0, "h i argmin L[ϕ]", 12)
    left_column = _line(45.1, 250.0, 470.0, 480.0, "Argmin function over the parameters", 0)
    # 两栏各自的正文行，用来定义"右栏从 300 起"
    body_right = _line(300.0, 506.4, 500.0, 510.0,
                       "the model makes a prediction for the output based on the input", 0, 63)
    body_left = _line(45.1, 250.0, 500.0, 510.0,
                      "different choices for the y-intercept and the slope change", 0, 58)

    found = _formula_at(number, [number, right_column, left_column, body_right, body_left],
                        right_edges=[506.4, 250.0])
    assert found is not None
    assert found.bbox[0] == 300.0          # 左栏那行没有进来
    assert "Argmin function" not in found.text


# ------------------------------------------------------------------ 七份样本

def test_transformer_has_three_numbered_equations():
    report = analyze(TRANSFORMER_V7)
    assert [f.number for f in report.formulas] == ["1", "2", "3"]
    assert [f.page for f in report.formulas] == [4, 5, 7]
    for formula in report.formulas:
        assert NUMBER.match(f"({formula.number})")
        assert formula.label == f"式 ({formula.number})"
        assert formula.math_ratio >= 0.3
        assert formula.text.strip()
        assert formula.bbox[0] < formula.bbox[2] and formula.bbox[1] < formula.bbox[3]
    # 式 1 的文本层是有损的：上标 T 摊平、根号被拼成普通字符 —— 这正是"不给 LaTeX"的理由
    first = report.formulas[0].text
    assert first.startswith("Attention(Q, K, V )")
    assert "√" in first


def test_both_versions_number_the_same_three_equations():
    """v1 与 v7 的版式不同（v7 顶上多一段授权声明），但公式编号必须一致。"""
    v1 = analyze(TRANSFORMER_V1)
    v7 = analyze(TRANSFORMER_V7)
    assert [f.number for f in v1.formulas] == [f.number for f in v7.formulas]
    assert [f.page for f in v1.formulas] != [f.page for f in v7.formulas]


def test_scan_without_text_layer_reports_why_there_are_none():
    report = analyze(TRANSFORMER_SCAN)
    assert report.formulas == ()
    assert report.scanned_pages                 # 整份都是扫描页
    assert "扫描页" in report.note


def test_paper_without_numbered_equations_says_none_not_failed():
    """BERT 全篇没有编号公式（措辞里写的就是"没有找到"，不是"解析失败"）。"""
    report = analyze(BERT)
    assert report.formulas == ()
    assert report.scanned_pages == ()
    assert "没有找到带编号的公式" in report.note


def test_book_numbers_in_chapters_and_skips_unparsed_pages():
    report = analyze(UBD)
    assert len(report.formulas) >= 5
    numbers = [f.number for f in report.formulas]
    assert numbers[0].startswith("2.")          # 第 2 章的公式从 (2.1) 起
    assert all(NUMBER.match(f"({n})") for n in numbers)
    assert report.analyzed_pages == 40 < report.pages
    assert "只解析了前 40 页" in report.note


def test_second_run_comes_from_the_disk_cache(tmp_path, monkeypatch):
    """缓存目录跟着 `PAPER_STORE_DIR` 走（它是在调用时读的，不是导入时）。

    换成临时目录再跑：否则这台机器上早已存在的缓存会让"第一次"也是 `cached=True`。
    """
    monkeypatch.setenv("PAPER_STORE_DIR", str(tmp_path / "papers"))
    first = analyze(TRANSFORMER_V7)
    second = analyze(TRANSFORMER_V7)
    assert first.cached is False
    assert second.cached is True
    assert second.formulas == first.formulas
    assert list((tmp_path / "formula_cache").glob("*.json"))


def test_missing_file_raises():
    with pytest.raises(Exception):
        analyze(PDF_DIR / "不存在的论文.pdf")


def test_math_font_pattern_covers_the_families_we_depend_on():
    for font in ("CMMI10", "CMSY10", "CMEX9", "CMR7", "MSBM10",
                 "STIXTwoMath-Regular", "LatinModernMath", "XITSMath"):
        assert MATH_FONT.search(font), font
    for font in ("NimbusRomNo9L-Regu", "Times-Roman", "Helvetica-Bold", "Arial"):
        assert not MATH_FONT.search(font), font


def test_report_is_a_frozen_dataclass():
    report = analyze(TRANSFORMER_V7)
    assert isinstance(report, FormulaReport)
    with pytest.raises(Exception):
        report.cached = True  # type: ignore[misc]
