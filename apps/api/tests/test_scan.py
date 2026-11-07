"""扫描件识别与 OCR 降级的测试。

这里守的是四条底线：
1. **该认的认出来**：只有图片没有文字层的 PDF 必须被判成扫描件，
   正常的文字 PDF 不能被误判 —— 误判会让好端端的文档去跑昂贵的 OCR。
2. **混合文档要精确到页**：一页文字一页图时，只能把图那页标为 scanned_pages，
   否则上层会把整份文档都当扫描件处理。
3. **没有 OCR 也不能崩**：引擎加载不起来时，`ocr_available` 必须是 False，
   `ocr_pages` 必须返回空字典；损坏文件也必须返回 pages=0 的报告而不是抛异常。
   假装识别成功比明确失败更危险，因为它会把幻觉文本送进检索库。
4. **识别出来的东西要能直接喂给下游**：读序、词框、字号代理、磁盘缓存，
   任何一处不对，切块与引用就会建立在错乱文本上。

真正跑一次识别的用例只有一条（`test_image_page_is_actually_recognised`）：
OCR 在这台机器上约 9 秒一页，测试里只认一页，其余用例都用合成数据。

运行: PYTHONUTF8=1 uv run pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.parsing import scan
from app.parsing.scan import analyze, ocr_pages


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """OCR 缓存指到临时目录，别把测试结果写进真实资料库里。"""
    monkeypatch.setenv("PAPER_STORE_DIR", str(tmp_path / "papers"))
    scan._LINE_CACHE.clear()
    yield
    scan._LINE_CACHE.clear()


def _line(text: str, x0: float, y0: float, x1: float, y1: float) -> scan.OcrLine:
    return scan.OcrLine(
        text=text,
        bbox=(x0, y0, x1, y1),
        size=y1 - y0,
        words=[scan.OcrWord(text, (x0, y0, x1, y1))],
    )


def _write_text_pdf(path: Path, n_pages: int = 1) -> Path:
    """造一份有真实文字层的 PDF：每页塞足够多的正文。"""
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        body = f"Page {i + 1}. The quick brown fox jumps over the lazy dog. " * 4
        page.insert_text((72, 72), body, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _make_image_only_pdf(path: Path, n_pages: int = 1) -> Path:
    """造一份只有图片、没有文字层的 PDF。

    做法是先生成带文字的页，再把它渲染成 PNG 插入新文档。
    这样新文档里只剩像素，`get_text()` 必然拿不到任何字符 ——
    这正模拟扫描仪输出的扫描件。
    """
    source = pymupdf.open()
    src_page = source.new_page()
    src_page.insert_text((72, 72), "This text must survive only as pixels. " * 8, fontsize=14)
    image = src_page.get_pixmap(dpi=120).tobytes("png")
    source.close()

    doc = pymupdf.open()
    for _ in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=image)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def text_pdf(tmp_path: Path) -> Path:
    return _write_text_pdf(tmp_path / "text.pdf")


@pytest.fixture()
def image_pdf(tmp_path: Path) -> Path:
    return _make_image_only_pdf(tmp_path / "image.pdf")


def test_image_only_pdf_is_detected_as_scanned(image_pdf: Path):
    """没有文字层的一页必须整页判成扫描件，字符数为 0。"""
    report = analyze(str(image_pdf))
    assert report.pages == 1
    assert report.scanned_pages == [1]
    assert report.readable_pages == 0
    assert report.text_chars < scan.MIN_PAGE_TEXT_CHARS
    assert "扫描件" in report.note


def test_all_image_pages_are_reported(tmp_path: Path):
    """多页扫描件必须每一页都被列出，而不是只报第一页。"""
    pdf = _make_image_only_pdf(tmp_path / "scan3.pdf", n_pages=3)
    report = analyze(str(pdf))
    assert report.pages == 3
    assert report.scanned_pages == [1, 2, 3]
    assert report.readable_pages == 0


def test_text_pdf_is_not_misjudged(text_pdf: Path):
    """正常文字 PDF 不能被当成扫描件，否则会平白触发 OCR。"""
    report = analyze(str(text_pdf))
    assert report.pages == 1
    assert report.scanned_pages == []
    assert report.readable_pages == 1
    assert report.text_chars >= scan.MIN_PAGE_TEXT_CHARS
    assert "无需 OCR" in report.note


def test_mixed_pdf_flags_only_the_image_page(tmp_path: Path):
    """一页文字一页图：只有图那页是扫描页。"""
    doc = pymupdf.open()
    text_page = doc.new_page()
    text_page.insert_text((72, 72), "This page has a real digital text layer. " * 4, fontsize=12)

    source = pymupdf.open()
    src_page = source.new_page()
    src_page.insert_text((72, 72), "Rendered to an image only. " * 8, fontsize=14)
    image = src_page.get_pixmap(dpi=120).tobytes("png")
    source.close()

    image_page = doc.new_page(width=595, height=842)
    image_page.insert_image(image_page.rect, stream=image)
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    doc.close()

    report = analyze(str(path))
    assert report.pages == 2
    assert report.scanned_pages == [2]
    assert report.readable_pages == 1
    assert "第 2 页" in report.note


def _engine_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟"本机没有可用的 OCR 引擎"。

    `RAG_OCR=off` 是模块自己的开关，比去 monkeypatch 第三方包更接近真实场景：
    评委机器上 RapidOCR 装不上时走的就是这条路径。
    """
    monkeypatch.setenv("RAG_OCR", "off")
    scan._engine.cache_clear()
    scan.ocr_enabled.cache_clear()


def test_ocr_is_reported_unavailable_when_the_engine_is_off(
    monkeypatch: pytest.MonkeyPatch, image_pdf: Path
):
    """没有引擎时必须明说，而不是让用户以为"识别出来就是空的"。"""
    _engine_off(monkeypatch)
    try:
        report = analyze(str(image_pdf))
        assert report.ocr_available is False
        assert "RapidOCR" in report.note
        assert "扫描件" in report.note
    finally:
        monkeypatch.delenv("RAG_OCR", raising=False)
        scan._engine.cache_clear()
        scan.ocr_enabled.cache_clear()


def test_ocr_pages_returns_empty_dict_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, image_pdf: Path
):
    """OCR 不可用时返回空字典且不抛异常 —— 调用方据此降级，而不是收到 500。"""
    _engine_off(monkeypatch)
    try:
        assert ocr_pages(str(image_pdf)) == {}
        # 显式指定页码也必须一样安全
        assert ocr_pages(str(image_pdf), [1]) == {}
    finally:
        monkeypatch.delenv("RAG_OCR", raising=False)
        scan._engine.cache_clear()
        scan.ocr_enabled.cache_clear()


def test_corrupt_pdf_returns_empty_report(tmp_path: Path):
    """损坏文件必须退化成 pages=0 的报告，绝不能把异常抛给 API 层。"""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body\n")

    report = analyze(str(broken))
    assert report.pages == 0
    assert report.readable_pages == 0
    assert report.scanned_pages == []
    assert report.text_chars == 0
    assert report.note  # 必须给人一句可读的解释
    assert "无法打开" in report.note


# ---------------------------------------------------------------- 版面与读序


def test_rotated_text_is_dropped_instead_of_jumbled_into_the_body():
    """竖排/旋转的文字必须整行丢掉，不能混进正文。

    论文左边距那个竖排的 arXiv 投稿戳会被切成一堆碎块（"1"、"2"、"un"、"CL]"），
    它们在最左边，读序把它们排到正文前面 —— 实测整页正文因此被推后，
    首页开头变成"1 / 2 / un / J / 2 …"，标题反而排到十几行之后。
    """
    upright = "Attention Is All You Need"
    rotated = "CL]"
    result = [
        ([[100, 100], [400, 100], [400, 116], [100, 116]], upright, 0.99),
        ([[20, 200], [26, 200], [26, 260], [20, 260]], rotated, 0.99),
        ([[100, 300], [420, 300], [420, 316], [100, 316]], "Ashish Vaswani*", 0.99),
    ]
    lines = scan._to_lines(result, (1000, 700), 72 / 120)
    texts = [line.text for line in lines]
    assert texts == ["Attention Is All You Need", "Ashish Vaswani*"]


def test_a_lone_character_is_not_a_line():
    """孤零零一个字符（页码、装饰标记）不是正文，别让它变成一段。"""
    result = [
        ([[100, 100], [130, 100], [130, 116], [100, 116]], "2", 0.99),
        ([[100, 300], [420, 300], [420, 316], [100, 316]], "Results are reported below.", 0.99),
    ]
    assert [line.text for line in scan._to_lines(result, (1000, 700), 1.0)] == [
        "Results are reported below."
    ]


def test_authors_on_one_row_are_read_left_to_right():
    """同一排几个作者，基线差零点几 pt，不能按 y 大小打乱。

    实测首页四个作者被读成「Jakob、Noam、Ashish、Niki」，
    正确顺序是 Ashish、Noam、Niki、Jakob（PDF 文字层就是这个顺序）。
    """
    row = [
        _line("Ashish", 132.0, 184.2, 200.0, 196.0),
        _line("Noam", 237.0, 183.6, 300.0, 195.4),
        _line("Niki", 336.6, 184.2, 400.0, 196.0),
        _line("Jakob", 421.2, 183.0, 490.0, 194.8),
    ]
    assert [line.text for line in scan._row_sort(row)] == ["Ashish", "Noam", "Niki", "Jakob"]


def test_two_columns_are_read_one_column_at_a_time():
    """双栏页面必须整栏读完再读右栏。

    按 y 排会读成「左1 右1 左2 右2」，整篇文章的句子被打散，
    检索和引用都建立在错乱文本上。
    """
    left = [_line(f"L{i}", 60.0, 100.0 + i * 14, 290.0, 112.0 + i * 14) for i in range(6)]
    right = [_line(f"R{i}", 320.0, 100.0 + i * 14, 550.0, 112.0 + i * 14) for i in range(6)]
    mixed = [item for pair in zip(left, right) for item in pair]  # 故意交错着给

    ordered = scan._reading_order(mixed)
    assert [line.text for line in ordered] == [f"L{i}" for i in range(6)] + [
        f"R{i}" for i in range(6)
    ]


def test_a_full_width_heading_above_two_columns_is_read_first():
    """通栏标题会拦腰截断竖缝：先横切剥掉标题带，剩下的才能按栏切。

    不先横切的话整页都找不到贯通竖缝，只能按 y 排，两栏又交错了。
    """
    heading = _line("3 Model Architecture", 60.0, 40.0, 550.0, 60.0)
    left = [_line(f"L{i}", 60.0, 120.0 + i * 14, 290.0, 132.0 + i * 14) for i in range(4)]
    right = [_line(f"R{i}", 320.0, 120.0 + i * 14, 550.0, 132.0 + i * 14) for i in range(4)]

    ordered = scan._reading_order([left[0], right[0], heading, left[1], right[1], *left[2:], *right[2:]])
    assert ordered[0].text == "3 Model Architecture"
    assert [line.text for line in ordered[1:]] == [f"L{i}" for i in range(4)] + [
        f"R{i}" for i in range(4)
    ]


# ---------------------------------------------------------------- 词与缓存


def test_words_keep_the_spaces_the_model_gave_us():
    """模型吐出来的空格必须原样保留，词框按字数据行宽均分。

    PP-OCR 会吃掉一部分词间空格，剩下那些是唯一可靠的词边界 ——
    自己再从像素里找空隙试过，整体更差（见 docs/EXPERIMENTS.md E-018）。
    """
    words = scan._words_from_text("We propose the Transformer", (100, 200, 500, 216), 1.0)
    assert [word.text for word in words] == ["We", "propose", "the", "Transformer"]
    assert words[0].bbox[0] == pytest.approx(100.0)
    assert words[-1].bbox[2] <= 500.0
    # 框必须单调向右，否则高亮框会互相盖住
    assert [word.bbox[0] for word in words] == sorted(word.bbox[0] for word in words)


def test_line_size_is_the_ink_height_not_a_font_size(tmp_path: Path):
    """`size` 是像素墨迹高度 —— 量纲对得上，但**不能**当字号用。

    这条测试原来断言"标题行比正文行高一截，所以切块器可以拿它认标题"。
    真跑一遍扫描件才发现反了：同一份论文的正文行高 13~19pt、真标题 13~20pt，
    因为高度取决于这一行有没有括号、升部、降部。所以切块器现在明确
    **不用**它认标题（见 `Para.size_proxy`），这里只守"它等于框高"这个事实。
    """
    result = [
        ([[60, 100], [540, 100], [540, 116], [60, 116]], "body text here", 0.99),
        ([[60, 200], [540, 200], [540, 226], [60, 226]], "Big Heading", 0.99),
    ]
    lines = scan._to_lines(result, (1000, 700), 1.0)
    body = next(line for line in lines if line.text.startswith("body"))
    head = next(line for line in lines if line.text.startswith("Big"))
    assert body.size == pytest.approx(16.0)
    assert head.size == pytest.approx(26.0)


def test_a_standalone_heading_line_gets_its_own_block(monkeypatch, image_pdf: Path):
    """`References` 必须单独成块 —— 这是扫描件唯一能认出它的机会。

    真实事故：识别出来的 `References` 被并进上一段正文的尾巴
    （`…available soon. References`），切块器的关键词判据要求整块**只有**
    这个词，于是整份 15 页扫描件抽出 0 条引文，而它的文字版有 36 条。
    """
    lines = [
        _line("such as images, audio and video. We intend to make", 60, 100, 540, 116),
        _line("the code we used to train and evaluate available soon.", 60, 120, 540, 136),
        # 行距故意贴得很紧，只有"短行独立成块"这条规则能把它切开
        _line("References", 60, 141, 120, 157),
        _line("[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton.", 60, 162, 540, 178),
    ]
    monkeypatch.setattr(scan, "ocr_enabled", lambda: True)
    monkeypatch.setattr(scan, "ocr_lines", lambda page, dpi=None: lines)
    doc = pymupdf.open(str(image_pdf))
    try:
        result = scan.ocr_blocks_for_page(doc[0])
    finally:
        doc.close()
    texts = [
        " ".join(
            "".join(span["text"] for span in line["spans"]).strip()
            for line in block["lines"]
        )
        for block in result["blocks"]
    ]
    assert "References" in texts, texts


def test_page_width_reference_ignores_a_page_full_of_figure_labels():
    """参考行宽要取 90 分位，不能取中位数。

    带插图的页面上图注、坐标轴标签又多又短：实测第 3 页中位行宽只有 50pt，
    正文满行是 397pt。中位数一被拉低，"短行"的阈值就成了 20pt，
    整页没有一行够格独立成块，`3.2 Attention` 这类标题全被并进正文。
    """
    widths = [14, 18, 25, 27, 31, 36, 50, 64, 132, 157, 191, 397, 397, 397]
    assert scan._width_p90(widths) == pytest.approx(397.0)


def test_recognised_page_survives_a_process_restart(tmp_path: Path):
    """识别结果必须落盘：资料库每次启动都会把所有 PDF 重新切一遍。

    没有磁盘缓存的话，库里只要有一份扫描件，**每次重启都要重跑两三分钟 OCR**，
    演示现场会被当成"服务挂了"。
    """
    lines = [_line("Attention Is All You Need", 100.0, 100.0, 400.0, 116.0)]
    key = "deadbeef"
    scan._store_cached(key, lines)

    assert (scan._cache_dir() / f"{key}.json").exists()
    loaded = scan._load_cached(key)
    assert loaded is not None
    assert loaded[0].text == "Attention Is All You Need"
    assert loaded[0].bbox == (100.0, 100.0, 400.0, 116.0)
    assert loaded[0].words[0].text == "Attention Is All You Need"


def test_corrupt_cache_is_ignored_not_fatal(tmp_path: Path):
    """缓存文件坏了就当没有，重新识别即可 —— 不能因此让整份文档解析失败。"""
    key = "corrupt"
    folder = scan._cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.json").write_text("{ this is not json", encoding="utf-8")
    assert scan._load_cached(key) is None
    assert scan._load_cached("never-written") is None


# ---------------------------------------------------------------- 真跑一次


def test_image_page_is_actually_recognised(image_pdf: Path, tmp_path: Path):
    """扫描页必须真的读出字来：没有文字层也能进检索、能引用。"""
    if not scan.ocr_enabled():
        pytest.skip("本机没有可用的 OCR 引擎")

    result = ocr_pages(str(image_pdf), [1])
    assert result, "扫描页一个字符都没识别出来"
    text = result[1].lower()
    assert "pixels" in text
    # 样本是同一句话重复写在一行里（超出页宽的部分看不见），能认出两遍就算正常
    assert text.count("pixels") >= 2, "只认出零星几个词，说明行合并或读序出了问题"
