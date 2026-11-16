"""切块器的测试。

切块是检索质量的上限：块切错了，再好的 embedding（嵌入）也救不回来。
这里守的是四条底线：
1. 块要**覆盖全文**，不能漏页；
2. 块的 section（章节路径）要**层级正确**，否则引用定位会指错地方；
3. 块不能**超预算**，否则一检索就把整页噪声塞进提示词；
4. 页眉页脚不能混进正文，这是幻觉的常见来源。

运行: uv run --with pymupdf --with pytest pytest apps/api/tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.chunker import Chunk, MAX_TOKENS, _boiler_key, chunk_pdf, extract_paras

ROOT = Path(__file__).resolve().parents[3]
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_GPT3 = ROOT / "pdf" / "2005.14165_GPT-3.pdf"
PDF_BOOK = ROOT / "pdf" / "UnderstandingDeepLearning_02_09_26_C.pdf"


@pytest.fixture(scope="module")
def chunks_v7():
    return chunk_pdf(str(PDF_V7))


def test_chunks_cover_every_page(chunks_v7):
    """每一页都必须至少落进一个块，否则那几页等于没入库。"""
    covered = set()
    for c in chunks_v7:
        covered.update(range(c.page_start, c.page_end + 1))
    assert covered == set(range(1, 16)), f"缺页: {sorted(set(range(1, 16)) - covered)}"


def test_section_path_is_hierarchical(chunks_v7):
    """§3.2.1 的块，路径里应同时出现父级和本级。"""
    hit = [c for c in chunks_v7 if "softmax" in c.text and "3.2.1" in c.section]
    assert hit, "没找到挤点积注意力的块"
    section = hit[0].section
    assert "3.2 Attention" in section
    assert "Scaled Dot-Product Attention" in section


def test_no_chunk_exceeds_budget(chunks_v7):
    assert max(c.n_tokens for c in chunks_v7) <= MAX_TOKENS


def test_header_footer_is_stripped(chunks_v7):
    """arXiv 侧边水印在每页都出现，属于页眉，绝不能进正文。"""
    for c in chunks_v7:
        assert "arXiv:1706.03762v7" not in c.text
        assert "arXiv:1706.03762v7 [cs.CL]" not in c.text


def test_linebreak_hyphen_is_repaired(chunks_v7):
    """行尾断词 signi- / cantly 必须拼回 significantly。"""
    blob = " ".join(c.text for c in chunks_v7).lower()
    assert "significantly" in blob


def test_deterministic(chunks_v7):
    again = chunk_pdf(str(PDF_V7))
    assert [(c.index, c.text) for c in again] == [(c.index, c.text) for c in chunks_v7]


def test_bboxes_are_inside_page():
    """坐标必须落在页面内，否则前端高亮会画到纸外。"""
    import pymupdf

    doc = pymupdf.open(str(PDF_V7))
    out_of_range = []
    for c in chunk_pdf(str(PDF_V7)):
        for page_no, (x0, y0, x1, y1) in c.rects:
            page = doc[page_no - 1]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                out_of_range.append((c.index, page_no, (x0, y0, x1, y1)))
    doc.close()
    assert not out_of_range


def test_ligature_pdf_still_chunks():
    """v1 带连字（ﬁ）：原文形态保留，归一化后能变成正常的 fi。

    注意原文写的是 "encoder-decoder conﬁguration"（连字符，不是空格），
    所以断言要按真实形态来，不能想当然。
    """
    chunks = chunk_pdf(str(PDF_V1))
    assert len(chunks) > 10
    raw = " ".join(c.text for c in chunks)
    assert "conﬁguration" in raw, "连字字符在切块时被弄丢了"
    assert "configuration" in raw.replace("ﬁ", "fi").lower()


def test_extract_paras_returns_pages():
    paras, n_pages = extract_paras(str(PDF_V7))
    assert n_pages == 15
    assert len(paras) > 100
    assert all(p.page >= 1 for p in paras)


def test_the_page_number_does_not_hide_a_repeating_header():
    """`464 Bibliography` 与 `Bibliography 465` 是同一个页眉。

    只在数字后面还跟着编号时才剥前导页码的话，这两个键不一样 ——
    于是一页一个键，谁都不重复，页眉一条都剔不掉。
    """
    assert _boiler_key("464 Bibliography") == _boiler_key("Bibliography 465") == "bibliography"
    # 但 `Figure 1.4` 与 `Figure 1.5` 必须还是两个键，否则图注会被当成页眉删掉
    assert _boiler_key("Figure 1.4 Bivariate") != _boiler_key("Figure 1.5 Bivariate")


@pytest.mark.skipif(not PDF_BOOK.exists(), reason="教材 PDF 不在仓库里")
def test_a_big_heading_in_the_margin_is_not_page_furniture():
    """教材每页页眉都写着 `Bibliography`，真正的节标题在第 477 页。

    剥掉页码后两者同键 —— 不按字号分开，真标题会被当页眉删掉，
    整个书目节就不存在了（实测只剩第一页算数）。
    """
    chunks = chunk_pdf(str(PDF_BOOK))
    pages = sorted({c.page_start for c in chunks if c.section == "Bibliography"})
    assert len(pages) > 20, f"书目只覆盖了 {len(pages)} 页"
    assert not any(c.section[:1].isdigit() for c in chunks if "Bibliograph" in c.section)


@pytest.mark.skipif(not PDF_GPT3.exists(), reason="样本 PDF 不在仓库里")
def test_the_papers_own_table_of_contents_is_not_a_section():
    """GPT-3 第 2 页是它自己的目录，`8 Conclusion 40` 这类行正好符合
    "编号 + 短标题"的标题判据 —— 不加拦截，目录条目会变成章节名，
    还排在 `1 Introduction` 前面，那块目录文本也会被当证据召回。

    判据是点引导线（`3 Results 10 . . . . 11`）：实测五份样本里只有这一页命中。
    """
    chunks = chunk_pdf(str(PDF_GPT3))
    sections = [c.section for c in chunks]
    assert not any("Conclusion 40" in s for s in sections)
    assert sections[:2] == ["Language Models are Few-Shot Learners", "Abstract"]
    assert "1 Introduction" in sections[:4]


# ---------------------------------------------------------------- 扫描件 OCR 接线


def _scanned_pdf(tmp_path: Path, text_pages: int = 0):
    """造一份"只有图片、没有文字层"的 PDF，用来验证 OCR 钩子真的被调用。

    做法：先在内存里画一页正常文字 → 转成像素图 → 塞进新文档当图片。
    这样新文档的文字层是空的，和真实扫描件在解析层面等价。
    """
    import pymupdf

    out = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 120), "This text only exists as pixels.", fontsize=14)
    pix = page.get_pixmap(dpi=110)
    doc.close()

    scanned = pymupdf.open()
    target = scanned.new_page()
    target.insert_image(target.rect, pixmap=pix)
    for _ in range(text_pages):
        normal = scanned.new_page()
        normal.insert_text((72, 120), "A real text layer on this page.", fontsize=12)
    scanned.save(str(out))
    scanned.close()
    return out


def _fake_blocks(text: str) -> dict:
    """伪造一份 `get_text("dict")` 结构：OCR 钩子该返回的样子。"""
    return {
        "blocks": [
            {
                "type": 0,
                "bbox": (60.0, 100.0, 540.0, 140.0),
                "lines": [
                    {
                        "spans": [
                            {"text": text, "size": 11.0, "bold": False, "flags": 0},
                        ]
                    }
                ],
            }
        ]
    }


def test_ocr_hook_is_used_for_pages_without_a_text_layer(tmp_path):
    """没有文字层的页，注入的 OCR 钩子必须被调用，而且文字要进块。"""
    pdf = _scanned_pdf(tmp_path)
    calls = []

    def fake_ocr(page):
        calls.append(page.number)
        return _fake_blocks("OCR recovered paragraph about attention windows.")

    chunks = chunk_pdf(str(pdf), ocr=fake_ocr)
    assert calls, "文字层为空的页没有触发 OCR 钩子"
    assert any("OCR recovered paragraph" in c.text for c in chunks), [c.text for c in chunks]


def test_ocr_hook_is_skipped_when_a_text_layer_exists(tmp_path):
    """有文字层的页不许走 OCR：OCR 会丢公式、丢字号，反而更差。"""
    import pymupdf

    pdf = tmp_path / "text.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 120), "A perfectly readable text layer lives here.", fontsize=12)
    doc.save(str(pdf))
    doc.close()

    calls = []

    def fake_ocr(page):
        calls.append(page.number)
        return _fake_blocks("SHOULD NOT APPEAR")

    chunks = chunk_pdf(str(pdf), ocr=fake_ocr)
    assert calls == [], "有文字层的页不该调用 OCR"
    assert all("SHOULD NOT APPEAR" not in c.text for c in chunks)


def test_ocr_hook_returning_none_is_not_fatal(tmp_path):
    """钩子交不出结果（引擎缺失/单页失败）时，这份文档只是没有正文，不该抛异常。"""
    pdf = _scanned_pdf(tmp_path)
    assert chunk_pdf(str(pdf), ocr=lambda page: None) == []


# ---------------------------------------------------------------- 扫描件的标题
# 扫描件没有字体信息：`size` 只能用像素墨迹高度当代理，`bold` 恒为 False。
# 下面三条守的是"在只有代理值的情况下，什么能当标题、什么不能"。


def _ocr_pdf(tmp_path: Path, lines: list[tuple[str, float]]):
    """造一份无文字层的 PDF，并给一个按行返回 OCR 块的钩子。

    `lines` 是 `(文本, 像素行高)`：行高差就是切块器能看到的那点"字号"信息。
    """
    pdf = _scanned_pdf(tmp_path)

    def hook(page) -> dict:
        blocks = []
        y = 100.0
        for text, height in lines:
            blocks.append(
                {
                    "type": 0,
                    "bbox": (60.0, y, 540.0, y + height),
                    "lines": [
                        {
                            "spans": [
                                {"text": text, "size": height, "flags": 0, "font": "ocr"},
                            ]
                        }
                    ],
                }
            )
            y += height + 6
        return {"blocks": blocks}

    return pdf, hook


def test_a_section_number_glued_to_its_title_is_still_a_heading(tmp_path):
    """OCR 会把编号和标题粘成一个词：`7Conclusion` 也得认成一级标题 `7 Conclusion`。

    真实事故：识别出来的扫描件里 `1Introduction`、`2Background`、`7Conclusion`
    全都没有空格，编号判据要求"编号 + 空白 + 内容"，于是整份论文的章节树
    只剩零星几节，引用定位全指错。
    """
    pdf, hook = _ocr_pdf(
        tmp_path,
        [
            ("7Conclusion", 14.0),
            ("In this work, we presented the Transformer.", 16.0),
        ],
    )
    chunks = chunk_pdf(str(pdf), ocr=hook)
    assert [c.section for c in chunks] == ["7 Conclusion"], [c.section for c in chunks]


def test_a_figure_label_is_not_a_section_heading(tmp_path):
    """图表里的碎字不能进章节树。

    真实事故（FSA，arXiv:2508.18224）：目录里冒出三条假标题 ——
    `64 Seqlen (K)`（图里的坐标轴刻度）、`2.78 H100 NVL`（表格单元格）、
    `Q3 Q6`（图注里的问答编号）。它们的字号只有 3.2~5.5pt，而正文与真标题
    都是 10pt：**"编号 + 空格 + 内容"那条判据从不看字号**，图表里的字于是
    全被抓成了章节标题，连 4.4 节都被挂到了假标题底下。
    """
    import pymupdf

    path = tmp_path / "figures.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "1 INTRODUCTION", fontsize=10.0)
    page.insert_text(
        (72, 100),
        "This paper accelerates a sparse attention kernel used for long context.",
        fontsize=10.0,
    )
    page.insert_text((72, 140), "2 ATTENTION KERNEL", fontsize=10.0)
    page.insert_text((72, 160), "64 Seqlen (K)", fontsize=5.5)      # 坐标轴刻度
    page.insert_text((72, 176), "2.78 H100 NVL", fontsize=4.6)      # 表格单元格
    page.insert_text((72, 192), "Q3 Q6", fontsize=3.2)              # 图注编号
    page.insert_text((72, 220), "2.1 PERFORMANCE BREAKDOWN", fontsize=10.0)
    page.insert_text((72, 240), "Body text after the real heading, also ten point.", fontsize=10.0)
    doc.save(str(path))
    doc.close()

    sections = {c.section for c in chunk_pdf(str(path))}
    for fake in ("64 Seqlen", "2.78 H100 NVL", "Q3 Q6"):
        assert not any(fake in s for s in sections), (fake, sorted(sections))
    # 真标题还得认得出来，层级还得干净：2.1 挂在 2 底下，而不是挂在某个假标题底下
    assert any(
        s == "2 ATTENTION KERNEL › 2.1 PERFORMANCE BREAKDOWN" for s in sections
    ), sorted(sections)


def test_a_title_merged_into_its_paragraph_is_not_a_heading(tmp_path):
    """标题和下一段正文并成一块时，整块不能当标题。

    真实事故：扫描件的 `3.1 Encoder and Decoder Stacks` 和正文并成了一块，
    于是目录里出现一条二十个词长的"章节名"。代理字号下判据只能是"尾巴够短"。
    """
    pdf, hook = _ocr_pdf(
        tmp_path,
        [
            (
                "3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a "
                "stack of N 6 identical layers. Each layer has two",
                15.0,
            ),
            ("sub-layers. The first is a multi-head self-attention mechanism.", 16.0),
        ],
    )
    chunks = chunk_pdf(str(pdf), ocr=hook)
    assert not any("3.1" in c.section for c in chunks), [c.section for c in chunks]


def test_ocr_line_height_alone_never_makes_a_heading(tmp_path):
    """扫描件禁用"字号大于正文"这条规则：它量的是墨迹高度，不是字号。

    实测同一份论文（15 页扫描件）：正文行高 13~19pt，真标题行高 13~20pt ——
    两个区间几乎完全重叠。按它挑标题，既漏掉大半真标题，又把参考文献里的
    条目行判成标题，整个 References 节被切成七八段。这里给一行"特别高"的
    文献条目，它必须仍然不是标题。
    """
    pdf, hook = _ocr_pdf(
        tmp_path,
        [
            ("References", 16.0),
            ("[13] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars", 26.0),
            ("across languages. In Proceedings of EMNLP, 2009.", 13.0),
        ],
    )
    chunks = chunk_pdf(str(pdf), ocr=hook)
    assert [c.section for c in chunks] == ["References"], [c.section for c in chunks]


def test_false_headings_from_equations_and_captions_are_rejected(tmp_path):
    """四类"像标题其实是别的东西"的段落必须被拒。

    541 页教材实测：406 个章节名里 5 个是假的，其中 4 个以 `0 ` 开头 ——
    显示公式常以「0 …」起头，正好命中「编号 + 空白 + 内容」那条判据。
    这些假标题会直接进入阅读器的目录树，所以按硬特征排除，不猜语义。
    """
    pdf, hook = _ocr_pdf(
        tmp_path,
        [
            ("Introduction", 16.0),
            ("1.1 Supervised learning", 14.0),
            ("0 z < 0 1 z >=0 rect[z] =", 14.0),                    # 公式碎片
            ("0 that controls the overall height is added. (Interactive figure)", 14.0),
            ("1. This allows us to calculate the parameters and then apply the inverse", 14.0),
            ("body text follows here to keep the page non-empty", 10.0),
        ],
    )
    chunks = chunk_pdf(str(pdf), ocr=hook)
    sections = list(dict.fromkeys(c.section for c in chunks))
    assert not any(s.startswith("0 ") for s in sections), sections
    assert not any("Interactive figure" in s for s in sections), sections
    assert not any(s.startswith("1. This allows") for s in sections), sections
    # 真标题不能被误伤（块记录的是最深一级标题，所以这里是包含判断）
    assert any("Introduction" in s for s in sections), sections


def test_abstract_is_extracted_without_author_prefix():
    """摘要要能从切块里取出来，且不把作者名、机构名、邮箱块粘在开头。

    实测四篇：Transformer 干净；BERT 那块把作者与邮箱并进了 Abstract；
    GPT-4 开头是 `OpenAI∗`。剥的判据只有"已知名字/机构"与"邮箱块"两条。
    """
    from app.api.registry import _abstract_of

    chunks = [
        Chunk(index=0, text="Attention Is All You Need Ashish Vaswani Noam Shazeer",
              section="Attention Is All You Need", page_start=1, page_end=1,
              n_tokens=8, rects=()),
        Chunk(index=1, text="The dominant sequence transduction models are based on complex networks.",
              section="1 Introduction", page_start=2, page_end=2, n_tokens=10, rects=()),
    ]
    assert _abstract_of(chunks) == ""          # 没有 Abstract 块就返回空，不猜

    chunks.insert(1, Chunk(index=9, text="Ashish Vaswani Noam Shazeer We propose a new model.",
                           section="Abstract", page_start=1, page_end=1, n_tokens=9, rects=()))
    got = _abstract_of(chunks, ["Ashish Vaswani", "Noam Shazeer"])
    assert got == "We propose a new model.", got


def test_abstract_strips_email_block():
    from app.api.registry import _abstract_of

    chunk = Chunk(index=0, section="Abstract", n_tokens=12, rects=(), page_start=1, page_end=1,
                  text="Google AI Language {jacobdevlin,kentonl}@google.com We introduce BERT.")
    got = _abstract_of([chunk], ["Jacob Devlin"], ["Google AI Language"])
    assert got.startswith("We introduce BERT"), got
