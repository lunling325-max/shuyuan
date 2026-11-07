"""表格与插图提取的测试。

守的是一条底线：**表格里的数字必须还在自己那一格里**。
切块器把表格当正文处理时，`BLEU` 在表头、`41.8` 在第五行，检索到
"BLEU 多少"时模型只能猜。这里造一份带网格线、带图注和插图的小 PDF，
断言抽出来的行、列、标题与 HTML 面板要显示的东西一一对上。
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.parsing import visuals
from app.parsing.visuals import analyze, render_figure


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """缓存写到临时目录：别把测试产物混进真实资料库的 visual_cache。"""
    monkeypatch.setenv("PAPER_STORE_DIR", str(tmp_path / "papers"))
    yield
    visuals._load_cached.__wrapped__ if False else None  # 保持 fixture 对称，不做额外清理


def _table_pdf(path: Path, *, caption: str = "Table 1: BLEU scores of our models.") -> Path:
    """造一份"有网格线的真表格"的 PDF：4 行 × 3 列，外加一行标题。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), caption, fontsize=9)

    x0, y0 = 72, 110
    cell_w, cell_h = 130, 22
    rows = [
        ("Model", "BLEU EN-DE", "BLEU EN-FR"),
        ("ByteNet", "23.75", "39.2"),
        ("GNMT + RL", "24.6", "39.92"),
        ("Transformer", "28.4", "41.8"),
    ]
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x = x0 + c * cell_w
            y = y0 + r * cell_h
            page.draw_rect(pymupdf.Rect(x, y, x + cell_w, y + cell_h), color=(0, 0, 0), width=0.6)
            page.insert_text((x + 4, y + 15), text, fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def _figure_pdf(path: Path, caption: str = "Figure 1: The model architecture.") -> Path:
    """造一份带位图插图和图注的 PDF。图注在图片**下方**（期刊常见排法）。"""
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
    pix.set_rect(pix.irect, (200, 90, 60))  # 一块有颜色的方块，面积够大
    page.insert_image(pymupdf.Rect(200, 100, 400, 233), pixmap=pix)
    page.insert_text((72, 260), caption, fontsize=9)
    page.insert_text((72, 274), "Encoder and decoder stacks, drawn to scale.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def test_a_grid_table_keeps_its_numbers_in_the_right_cells(tmp_path: Path):
    """行、列、数字必须一一对上 —— 这是这张表存在的全部意义。"""
    report = analyze(_table_pdf(tmp_path / "table.pdf"))
    assert len(report.tables) == 1
    table = report.tables[0]
    assert table.n_rows == 4
    assert table.n_cols == 3
    assert table.rows[0] == ("Model", "BLEU EN-DE", "BLEU EN-FR")
    assert table.rows[3] == ("Transformer", "28.4", "41.8")


def test_the_table_caption_above_the_grid_is_picked_up(tmp_path: Path):
    """标题在表格上方（论文排版惯例），而且只取标题，不把正文一起吞进来。"""
    report = analyze(_table_pdf(tmp_path / "table.pdf"))
    table = report.tables[0]
    assert table.label == "Table 1"
    assert table.caption.startswith("Table 1: BLEU scores")


def test_markdown_rendering_has_a_header_separator(tmp_path: Path):
    """Markdown 表格没有分隔行就渲染不出来；竖线也要转义，否则列会被撑开。"""
    report = analyze(_table_pdf(tmp_path / "table.pdf"))
    md = report.tables[0].markdown().splitlines()
    assert md[0].startswith("| Model |")
    assert md[1] == "|---|---|---|"
    assert len(md) == 5


def test_a_figure_is_found_with_its_caption_below(tmp_path: Path):
    """图片标题在下方，且要把紧挨着的续行一起收进来。"""
    report = analyze(_figure_pdf(tmp_path / "fig.pdf"))
    assert len(report.figures) == 1
    figure = report.figures[0]
    assert figure.label == "Figure 1"
    assert figure.caption.startswith("Figure 1: The model architecture.")
    assert "drawn to scale" in figure.caption


def test_a_figure_can_be_rendered_to_png(tmp_path: Path):
    """缩略图得真的能画出来：前端 `<img>` 拿到的必须是 PNG 字节。"""
    path = _figure_pdf(tmp_path / "fig.pdf")
    figure = analyze(path).figures[0]
    png = render_figure(path, figure.page, figure.bbox)
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_rendering_an_out_of_range_page_returns_none_instead_of_raising(tmp_path: Path):
    """页码越界是调用方的问题，不是崩溃的理由。"""
    path = _figure_pdf(tmp_path / "fig.pdf")
    assert render_figure(path, 99, (0, 0, 10, 10)) is None


def test_a_scan_without_a_text_layer_says_so_instead_of_guessing(tmp_path: Path):
    """扫描页没有文字层，`find_tables()` 什么也找不到 —— 要如实说明，不要编。"""
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (10, 10, 10))
    page.insert_image(page.rect, pixmap=pix)  # 整页位图＝扫描件
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()

    report = analyze(path)
    assert report.tables == ()
    assert report.figures == ()
    assert "扫描页" in report.note


def test_the_report_is_cached_on_disk_and_reused(tmp_path: Path):
    """解析一次要十几秒，第二次必须直接读缓存。"""
    path = _table_pdf(tmp_path / "table.pdf")
    first = analyze(path)
    second = analyze(path)
    assert first.cached is False
    assert second.cached is True
    assert [t.rows for t in second.tables] == [t.rows for t in first.tables]


def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path: Path):
    """缓存坏了就当没有，重新解析 —— 不能让一个坏 JSON 卡住整个文献库。"""
    path = _table_pdf(tmp_path / "table.pdf")
    analyze(path)
    for item in visuals._cache_dir().glob("*.json"):
        item.write_text("{ this is not json", encoding="utf-8")
    report = analyze(path)
    assert report.cached is False
    assert len(report.tables) == 1


def _word(x0: float, y0: float, x1: float, y1: float, text: str) -> tuple:
    return (x0, y0, x1, y1, text, 0, 0, 0)


def test_a_missing_value_leaves_a_hole_instead_of_shifting_the_row():
    """第一个模型没有 EN-FR 值：那个格子必须是空的，不能把下一行的 39.92 顶上来。"""
    cells = [(0, 0, 60, 40), (60, 0, 120, 40), (120, 0, 180, 40)]
    words = [
        _word(0, 0, 40, 10, "ByteNet"), _word(60, 0, 100, 10, "23.75"),
        _word(0, 14, 40, 24, "GNMT"), _word(60, 14, 100, 24, "24.6"),
        _word(120, 14, 160, 24, "39.92"),
    ]
    assert visuals._visual_rows(words, cells) == [
        ("ByteNet", "23.75", ""),
        ("GNMT", "24.6", "39.92"),
    ]


def test_a_merged_cell_falls_back_to_the_per_cell_path():
    """有合并单元格（None）时拼不出视觉行，返回空让调用方逐格拼接，别猜。"""
    cells = [(0, 0, 60, 40), None, (120, 0, 180, 40)]
    assert visuals._visual_rows([_word(0, 0, 40, 10, "x")], cells) == []


def test_a_stacked_cell_keeps_its_line_breaks():
    """booktabs 表的整个表体会被认成一格：格内必须保留分行，否则模型名和数字对不上。

    直接测 `_cell_text`：GPT-3 的 Table 2.1 是"外框都没有、只有三条横线"的排法，
    造一份同样形态的 PDF 反而造不准（`find_tables()` 会按文字行自己切）。
    """
    words = [
        _word(0, 0, 40, 10, "GPT-3"),
        _word(44, 0, 90, 10, "Small"),
        _word(0, 14, 40, 24, "GPT-3"),
        _word(44, 14, 90, 24, "Large"),
        _word(0, 28, 40, 38, "GPT-3"),
        _word(44, 28, 90, 38, "XL"),
    ]
    assert visuals._cell_text(words, (0, 0, 100, 40)) == "GPT-3 Small\nGPT-3 Large\nGPT-3 XL"


def test_markdown_replaces_in_cell_newlines_so_the_table_is_not_cut_in_half():
    """Markdown 表格里裸换行会把表拦腰切断，格内分行必须写成 <br>。"""
    table = visuals.DocTable(
        page=1, bbox=(0, 0, 1, 1), label="Table 1", caption="",
        rows=(("Model", "Size"), ("GPT-3 Small\nGPT-3 Large", "125M\n760M")),
    )
    md = table.markdown()
    assert "GPT-3 Small<br>GPT-3 Large" in md
    assert len(md.splitlines()) == 3  # 表头 + 分隔行 + 一行数据


def test_the_caption_label_keeps_its_sub_number(tmp_path: Path):
    """`Table 2.1` 不能退化成 `Table 2` —— GPT-3 里三张不同的表会全叫同一个名字。"""
    report = analyze(_table_pdf(tmp_path / "t.pdf", caption="Table 2.1: Sizes and architectures."))
    assert report.tables[0].label == "Table 2.1"


def test_a_captionless_grid_on_the_title_page_is_not_a_table(tmp_path: Path):
    """首页的"作者 + 单位"是两列对齐排版，`find_tables()` 会把整块人名认成一张表。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Language Models are Few-Shot Learners", fontsize=14)
    names = [("Tom B. Brown", "Benjamin Mann"), ("Nick Ryder", "Melanie Subbiah"),
             ("Jared Kaplan", "Prafulla Dhariwal"), ("Amanda Askell", "Girish Sastry")]
    for r, pair in enumerate(names):
        for c, name in enumerate(pair):
            x, y = 72 + c * 200, 120 + r * 24
            page.draw_rect(pymupdf.Rect(x, y, x + 200, y + 24), color=(0, 0, 0), width=0.5)
            page.insert_text((x + 4, y + 16), name, fontsize=9)
    path = tmp_path / "title.pdf"
    doc.save(str(path))
    doc.close()

    assert analyze(path).tables == ()


def test_a_page_full_of_dots_is_treated_as_a_table_of_contents(tmp_path: Path):
    """目录页的"章节名 + 页码"对齐得像表格，GPT-3 就是这么被误认成 32×3 的表的。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "Contents", fontsize=14)
    for i in range(6):
        page.insert_text((72, 120 + i * 16), "1.%d Section title ........... %d" % (i, 10 + i), fontsize=10)
    path = tmp_path / "toc.pdf"
    doc.save(str(path))
    doc.close()

    assert analyze(path).tables == ()
