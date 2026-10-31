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

from app.parsing.chunker import MAX_TOKENS, _boiler_key, chunk_pdf, extract_paras

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
