"""参考文献抽取的测试，服务于引用图谱。

这里断言的几乎都是**条数和某一条在不在**，不是"解析得多漂亮"。
原因很直接：图谱只问"这条引用指向的是不是库里已有的论文"，
标题抽错一个字都可能让一条真实的引用消失 —— 而一条消失的引用，
在图上表现为"没人引用你库里的论文"，这是**看着完全正常的错误**。
所以每一条断言都盯着"某条已知存在的引用到底抽出来没有"。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.references import (
    Reference,
    _block_texts,
    _by_markers,
    _looks_complete,
    extract_references,
    normalize_title,
    parse_entry,
    title_similarity,
)

ROOT = Path(__file__).resolve().parents[3]
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"
PDF_GPT3 = ROOT / "pdf" / "2005.14165_GPT-3.pdf"
PDF_GPT4 = ROOT / "pdf" / "2303.08774_GPT-4.pdf"
PDF_BOOK = ROOT / "pdf" / "UnderstandingDeepLearning_02_09_26_C.pdf"

needs_book = pytest.mark.skipif(not PDF_BOOK.exists(), reason="教材 PDF 不在仓库里")

needs = pytest.mark.skipif(
    not (PDF_V7.exists() and PDF_BERT.exists()), reason="样本 PDF 不在仓库里"
)


def _extract(path: Path) -> list[Reference]:
    from app.parsing.chunker import chunk_pdf

    return extract_references(chunk_pdf(str(path)), path)


# ---------------------------------------------------------------- 单条解析


def test_marked_entry_gets_title_and_venue():
    r = parse_entry(
        "12",
        "[12] Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string\n"
        "embeddings for sequence labeling. In COLING, pages 1638–1649.",
    )
    assert r.title == "Contextual string embeddings for sequence labeling"
    assert r.year == "2018"
    assert r.venue.startswith("In COLING")
    assert r.is_parsed


def test_marker_never_leaks_into_the_author_field():
    """作者串会拿去显示，开头挂着 "[12]" 很难看，也会污染后续的姓名匹配。"""
    r = parse_entry("12", "[12] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization.")
    assert not r.authors.startswith("[")
    assert r.authors.startswith("Jimmy")


def test_year_final_entry_still_finds_the_title():
    """ACL 体里年份可以放句尾（"…, 2019."），不能因为年份段被当成标题就丢标题。"""
    r = parse_entry(
        "XDH+19",
        "[XDH+19] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. "
        "Unsupervised data augmentation for consistency training, 2019.",
    )
    assert r.title == "Unsupervised data augmentation for consistency training"
    assert r.year == "2019"


def test_initials_do_not_split_the_author_list():
    """"Quoc V. Le" 里的点后面也跟空格，按句点硬切会把作者切成两半。"""
    r = parse_entry("7", "[7] Quoc V. Le and Tomas Mikolov. Distributed representations, 2014.")
    assert "Quoc V. Le" in r.authors


def test_hyphenated_line_breaks_are_repaired():
    """两种断词形态都要修；不修，标题和原文一个词都对不上。"""
    a = parse_entry("1", "[1] Some One. Uns pervised aug- mentation for consistency training.")
    b = parse_entry("2", "[2] Some One. Uns pervised aug-\nmentation for consistency training.")
    assert "augmentation" in a.title and "-" not in a.title
    assert a.title == b.title


def test_arxiv_id_is_extracted_for_preprints():
    r = parse_entry("1", "[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.")
    assert r.arxiv_id == "1607.06450"


def test_year_is_not_taken_from_the_arxiv_number():
    """`arXiv:2004.08994` 里的 2004 是月份。当成出版年，GPT-4 的参考文献里
    就会有一条 "Adversarial training..." 标成 2004 年（真正是 2020）。"""
    r = parse_entry(
        "1",
        "Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, and "
        "Jianfeng Gao. Adversarial training for large neural language models. "
        "arXiv preprint arXiv:2004.08994, 2020.",
    )
    assert r.year == "2020"


def test_year_comes_from_the_end_of_the_entry():
    """三种版面都把年份写在条目末尾，取最后一个才稳。"""
    acl = parse_entry(
        "1",
        "Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: "
        "A simple and general method for semi-supervised learning. ACL.",
    )
    numeric = parse_entry(
        "2",
        "[2] A. Vaswani, N. Shazeer, et al. Attention is all you need. In Advances in "
        "Neural Information Processing Systems, pages 5998-6008, 2017.",
    )
    assert acl.year == "2010"
    assert numeric.year == "2017"


def test_entry_without_any_year_says_so_instead_of_guessing():
    """宁可空着，也不能拿 arXiv 号冒充。空年份在界面上一看就知道缺，
    错年份会被人当真。"""
    r = parse_entry("1", "Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems.")
    assert r.year == ""


# ---------------------------------------------------------------- 切分


def test_completeness_rejects_a_bare_author_head():
    """ACL 体的条目头 "作者. 2010." 有年份也有句点，但显然没写完。

    不加长度判据的话，标题会另起一条，两条引用被劈成"作者头"和"标题+出处"，
    实测 Transformer 那条就是这么被埋进上一条 raw 里的。
    """
    assert not _looks_complete("Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010.")
    assert _looks_complete(
        "Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: "
        "a simple and general method for semi-supervised learning."
    )
    # 年份在句尾的长条目必须算写完 —— 否则它会去吞下一条
    assert _looks_complete(
        "Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. "
        "Unsupervised data augmentation for consistency training, 2019."
    )


def test_few_markers_fall_back_to_paragraph_mode():
    """正文里一两条 "see [3]" 不是参考文献标记，不能把整节都按标记切。"""
    assert _by_markers("see [3] for details. Also [4].") is None


def test_unparsed_entries_are_returned_not_dropped():
    """解析不出来也要返回。图谱要如实显示"N 条没解析"，不能假装它们不存在。"""
    r = parse_entry("1", "%%%%  --  ..")
    assert r.title == "" and not r.is_parsed


# ---------------------------------------------------------------- 标题比对


def test_title_similarity_tolerates_case_punctuation_and_truncation():
    a = normalize_title("Attention Is All You Need")
    assert a == normalize_title("attention is all you need")
    assert title_similarity(
        "Attention is all you need", "Attention Is All You Need."
    ) == pytest.approx(1.0)
    # 参考文献里的标题常被截断
    assert title_similarity("Attention is all you need", "Attention is all") > 0.5
    assert title_similarity("Attention is all you need", "BERT: Pre-training") < 0.1


# ---------------------------------------------------------------- 真 PDF


@needs
def test_bert_finds_the_transformer_citation():
    """回归测试，盯的是一个已经修掉的真 bug。

    BERT 的参考文献是无标记的 ACL 体。修之前 Transformer 那条被粘进了上一条
    Turian 的 raw 里，于是标题比对必然失败 —— 图上会显示"BERT 没引用 Transformer"。
    """
    refs = _extract(PDF_BERT)
    hits = [r for r in refs if "attention is all you need" in r.raw.lower()]
    assert len(hits) == 1
    assert hits[0].title.lower().startswith("attention is all you need")
    assert hits[0].year == "2017"
    assert len(refs) == 54, "BERT 的参考文献条数变了，得重看一眼"


@needs
def test_appendix_is_not_parsed_as_references():
    """参考文献后面紧跟的附录不是引用，附录里**自己那份参考文献**更不是。

    实测 GPT-4：不截断的话抽出 194 条，其中 89 条是附录正文和 System Card
    的独立文献表（编号从 [1] 重新开始，与正文重号）。
    """
    bert = _extract(PDF_BERT)
    assert not any(r.raw.lower().startswith("appendix") for r in bert)
    assert not any("gluebenchmark" in r.raw.lower() for r in bert)

    if PDF_GPT4.exists():
        gpt4 = _extract(PDF_GPT4)
        assert len(gpt4) == 85
        assert [r.key for r in gpt4] == [str(i) for i in range(1, 86)], "编号不连续，说明混进了别处的表"


@needs
def test_gpt3_reference_keys_survive_alphanumeric_markers():
    """GPT-3 用的是 [VSP+17] 这种字母键，按数字切会全军覆没。"""
    if not PDF_GPT3.exists():
        pytest.skip("GPT-3 样本不在")
    refs = _extract(PDF_GPT3)
    assert 100 < len(refs) < 200
    keys = [r.key for r in refs]
    assert "VSP+17" in keys
    hits = [r for r in refs if "attention is all you need" in r.raw.lower()]
    assert hits and hits[0].title.lower().startswith("attention is all you need")


@needs
def test_two_versions_of_the_same_paper_yield_different_reference_lists():
    """v1 只引了 30 多条，v7 有 40 条 —— 图谱要能看出这中间多了什么。"""
    v1, v7 = _extract(PDF_V1), _extract(PDF_V7)
    assert len(v1) < len(v7)
    assert len(v7) == 40


@needs
def test_extraction_is_deterministic():
    """同一份文件抽两次必须一模一样。"""
    a = _extract(PDF_V7)
    b = _extract(PDF_V7)
    assert [r.raw for r in a] == [r.raw for r in b]


def test_author_year_style_takes_the_published_year():
    """教材体把年份写在作者后：`… & Wonka, P. (2019). Title. … 2020.`

    最后那个 2020 是出处里的会议年。取最后一个四位数会报错年 —— 引用年份错了，
    图谱上"这本书引了一篇 2020 年的预印本"就是假话。
    """
    r = parse_entry(
        "1",
        "Abdal, R., Qin, Y., & Wonka, P. (2019). Image2StyleGAN: How to embed images into "
        "the StyleGAN latent space. International Conference on Machine Learning, 2020.",
    )
    assert r.year == "2019"
    assert r.authors.endswith("(2019)")
    assert r.title.startswith("Image2StyleGAN")


def test_trailing_page_reference_still_counts_as_finished():
    """教材体在条目末尾标页码（`… 833–862. 426`）。

    结尾不是句点，`_looks_complete` 就判"没写完"，下一条的作者行被并进上一条；
    一路并下去，900 条书目只解析出 255 条。页码引用因此要先剥掉再判。
    """
    assert _looks_complete(
        "Erasmus, A., Brunet, T. D. P., & Fisher, E. (2021). What is interpretability? "
        "Philosophy & Technology, 34, 833–862. 426"
    )
    assert not _looks_complete("Erasmus, A., Brunet, T. D. P., & Fisher, E. (2021).")


# ---------------------------------------------------------------- 教材（541 页）


@pytest.fixture(scope="module")
def book_refs() -> list[Reference]:
    return _extract(PDF_BOOK)


@needs_book
def test_a_book_writes_bibliography_not_references(book_refs):
    """只认 "References" 的话，教材在引用图谱上一条出边都没有 —— 它写的是 Bibliography。"""
    assert len(book_refs) > 500


@needs_book
def test_the_running_head_does_not_become_an_entry(book_refs):
    """每页页眉是 `464 Bibliography`。不挡的话，跨页的第一条引用会以它开头。"""
    assert not any("Bibliography" in r.raw[:60] for r in book_refs)


@needs_book
def test_the_footer_does_not_swallow_the_next_entries(book_refs):
    """页脚 `This work is subject to a Creative Commons …` 每页都有；
    不被识别成页脚，它就会当成一条"没写完的引用"吃掉后面几条的作者行。"""
    assert not any("Creative Commons" in r.raw for r in book_refs)
    assert not any("send errata" in r.raw for r in book_refs)


@needs_book
def test_a_page_number_never_lands_in_the_author_field(book_refs):
    """上一条的页码引用（`… 862. 426`）混进下条行首时，作者串会以数字开头。"""
    assert not [r for r in book_refs if re.match(r"^\d{1,3}\s", r.authors)]


@needs_book
def test_the_book_cites_three_of_our_library_papers(book_refs):
    """库里四篇论文，教材引了其中三篇（它 2024 年版还没收 GPT-4 报告）。
    这是图谱上唯一一条 541 页的书指向论文的边，抽错就整条边没了。"""
    titles = [r.title for r in book_refs if r.title]
    for target in (
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Language Models are Few-Shot Learners",
    ):
        best = max(title_similarity(t, target) for t in titles)
        assert best >= 0.9, f"{target} 没抽出来（最高相似度 {best:.2f}）"


def test_ocr_blocks_are_split_per_line_so_reference_markers_survive():
    """扫描件的块必须按行拆开。

    真实事故：OCR 按行距分块，把相邻两条文献并成一块
    （`… arXiv, 2014. [3] Denny Britz, …`）。`[n]` 标记必须顶在行首才认得出
    （正文里的 "see [3]" 不能当标记），并成一块之后 `[3]` 跑到句子中间，
    由此之后的所有条目一起丢掉 —— 15 页扫描件抽出 19 条，文字版是 36 条。
    """
    blocks = [
        {
            "type": 0,
            "lines": [
                {"spans": [{"text": "arXiv preprint arXiv, 2014. ", "font": "ocr"}]},
                {"spans": [{"text": "[3] Denny Britz, Anna Goldie. ", "font": "ocr"}]},
            ],
        }
    ]
    assert _block_texts(blocks) == ["arXiv preprint arXiv, 2014.", "[3] Denny Britz, Anna Goldie."]


def test_text_layer_blocks_keep_the_paragraph_the_typesetter_gave_us():
    """有文字层的块**不拆**：块边界是排版给的，拆了反而会把一条引用切碎。"""
    blocks = [
        {
            "type": 0,
            "lines": [
                {"spans": [{"text": "Ashish Vaswani, Noam Shazeer. ", "font": "NimbusRomNo9L-Medi"}]},
                {"spans": [{"text": "Attention is all you need. 2017.", "font": "NimbusRomNo9L-Medi"}]},
            ],
        }
    ]
    assert _block_texts(blocks) == ["Ashish Vaswani, Noam Shazeer. Attention is all you need. 2017."]
