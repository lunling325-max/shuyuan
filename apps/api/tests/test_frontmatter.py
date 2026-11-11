"""首屏元信息（作者与机构）识别。

赛题模块一点名要"标题、作者及机构信息、摘要……"，标题早就有了，这两个是补的。
用例守两件事：
1. 七份样本的首页都要认出**对的人**（名字一个字不差，机构是原文那几家）；
2. 认不出的（扫描件、封面没有作者的报告）必须是空列表。空列表是诚实，
   编一个出来才是错 —— 界面上"作者：Google"看着挺像回事，但它是假的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.frontmatter import extract_front_matter

PDF_DIR = Path(__file__).resolve().parents[3] / "pdf"
V1 = PDF_DIR / "1706.03762v1.pdf"
V7 = PDF_DIR / "1706.03762v7.pdf"
BERT = PDF_DIR / "1810.04805_BERT.pdf"
GPT3 = PDF_DIR / "2005.14165_GPT-3.pdf"
GPT4 = PDF_DIR / "2303.08774_GPT-4.pdf"
SCAN = PDF_DIR / "1706.03762v1_scan.pdf"
BOOK = PDF_DIR / "UnderstandingDeepLearning_02_09_26_C.pdf"


def test_transformer_authors_in_reading_order():
    """八位作者，按首页从左到右、从上到下的顺序。

    顺序不是摆设：用户扫一眼就知道"第一作者是谁"。版面里的邮箱行与等贡献记号
    必须被剔掉，否则名单里会混进 avaswani@google.com 和一个孤零零的 ∗。
    """
    fm = extract_front_matter(V7)
    assert fm.authors == [
        "Ashish Vaswani",
        "Noam Shazeer",
        "Niki Parmar",
        "Jakob Uszkoreit",
        "Llion Jones",
        "Aidan N. Gomez",
        "Łukasz Kaiser",
        "Illia Polosukhin",
    ]


def test_transformer_affiliations_deduped():
    """三家机构，"Google Brain" 重复出现五次也只写一次。"""
    fm = extract_front_matter(V7)
    assert fm.affiliations == ["Google Brain", "Google Research", "University of Toronto"]


def test_two_versions_same_people():
    """v1 与 v7 版式不同（v7 顶上多了一段 Google 授权声明），作者必须一致。

    那段声明里有 "Google" 这个词，落在标题**上方**，不该被当成机构 ——
    这也是版面范围必须有上界的原因。
    """
    assert extract_front_matter(V1).authors == extract_front_matter(V7).authors
    assert extract_front_matter(V1).affiliations == ["Google Brain", "Google Research", "University of Toronto"]


def test_bert_one_line_four_names():
    """四位作者排在同一行，机构只有一家 "Google AI Language"。"""
    fm = extract_front_matter(BERT)
    assert fm.authors == ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
    assert fm.affiliations == ["Google AI Language"]


def test_gpt3_thirty_one_names():
    """31 位作者分五排。第一排第一位是 "Tom B. Brown" ——
    名字里的中间名缩写（`B.`）曾是判"两句话"的误伤点，这条守着它。"""
    fm = extract_front_matter(GPT3)
    assert len(fm.authors) == 31
    assert fm.authors[0] == "Tom B. Brown"
    assert "Ilya Sutskever" in fm.authors
    assert fm.affiliations == ["OpenAI"]


def test_gpt4_report_has_no_author_names():
    """GPT-4 技术报告的封面上只印了 OpenAI 这三个字，没列人名。

    真实情况就是这样，所以作者列表必须为空 —— 这条例和上面几条一样重要：
    它证明这里**没有**从"OpenAI"猜出"某某团队"，也没有拿正文里出现的名字凑数。
    """
    fm = extract_front_matter(GPT4)
    assert fm.authors == []
    assert fm.affiliations == ["OpenAI"]


def test_scanned_pdf_yields_nothing_instead_of_raising():
    """扫描件首页没有文本层：返回空 FrontMatter，不抛异常。

    不带 OCR 跑这条路是故意的：OCR 认人名会错（数字 0 与字母 O、
    rn 与 m），而作者名是要给人看的，错一个字就把另一个人认成了作者。
    """
    fm = extract_front_matter(SCAN)
    assert fm.authors == []
    assert fm.affiliations == []


def test_book_cover_keeps_author_without_inventing_affiliation():
    """教材封面只有作者名和日期；机构在致谢里，不在首页。

    日期行（"February 8, 2026"）带数字，不该被当成人名。
    """
    fm = extract_front_matter(BOOK)
    assert fm.authors == ["Simon J.D. Prince"]
    assert fm.affiliations == []


def test_missing_file_is_not_an_error_for_callers():
    """路径不存在时抛的是普通异常，交给注册表那层兜底（见 `_front_matter`）。

    这条不是在夸"抛异常"好，而是把契约钉住：注册表知道该怎么接住它，
    别的地方不要偷偷改成返回空 —— 空列表和"文件没了"是两件事。
    """
    with pytest.raises(Exception):
        extract_front_matter(PDF_DIR / "nope.pdf")
