"""引用校验守卫的测试。

核心命题：**没在原文里逐字出现过的引用，绝不能通过校验。**
正例证明不误杀，反例证明不漏杀 —— 后者才是这个模块存在的理由。

运行: uv run --with pymupdf --with pytest pytest apps/api/tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.guardrails import Claim, Quote, verify_claims
from app.parsing import TextIndex

ROOT = Path(__file__).resolve().parents[3]
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"


@pytest.fixture(scope="module")
def indexes() -> dict[str, TextIndex]:
    return {"v1": TextIndex(str(PDF_V1)), "v7": TextIndex(str(PDF_V7))}


@pytest.fixture(scope="module")
def bert_index() -> TextIndex:
    if not PDF_BERT.exists():
        pytest.skip("BERT 样本不在仓库里")
    return TextIndex(str(PDF_BERT))


# ---------------------------------------------------------------- 正例：不该误杀

def test_exact_quote_hits_with_bbox(indexes):
    """逐字引用应命中，且给出该页内的坐标框。"""
    claim = Claim(
        text="v7 包含 Google 授权声明",
        quotes=[Quote("v7", "Provided proper attribution is provided, Google hereby grants permission")],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status == "supported"
    assert r.anchors[0].page == 1
    assert r.anchors[0].exact is True
    x0, y0, x1, y1 = r.anchors[0].rect
    assert x1 > x0 and y1 > y0


def test_anchor_source_text_does_not_repeat_compound_words(indexes):
    """回显的"原文"必须真的是原文。

    "intra-attention" 被切成 intra/attention 两个 token，两个 token 都存着整串
    raw，直接 join 会读成 "intra-attention intra-attention is …" —— 实测
    /api/papers/{id}/qa 返回的证据里全是这种重复，看起来像抽取坏了。
    """
    anchor = indexes["v7"].find(
        "intra-attention is an attention mechanism relating different positions of a single sequence"
    )
    assert anchor is not None and anchor.exact is True
    assert anchor.source_text.count("intra-attention") == 1
    assert anchor.source_text.startswith("intra-attention is an attention mechanism")


def test_ligature_pdf_matches(indexes):
    """v1 的 PDF 把 fi 存成连字 ﬁ（conﬁguration），归一化后仍应命中。"""
    claim = Claim(
        text="v1 正文",
        quotes=[Quote("v1", "convolutional neural networks in an encoder decoder configuration")],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status == "supported"


def test_a_quote_across_a_hyphenated_line_break_still_matches(bert_index):
    """跨行断词的引用必须命中。

    PDF 原文是 `In addi-` 换行 `tion to the masked …`，切块器把这块文本修成了
    "addition"（模型看到的就是修好的形态，抄回来也是 "addition"）。
    词索引如果按 PDF 原样建，就只有 "addi" + "tion" 两个 token，
    "addition" 在索引里根本不存在 —— **真的引用被判成幻觉**。
    实测 BERT 上三条真结论全部 unsupported，抗幻觉模块反而变成了"永远答不出来"。
    """
    claim = Claim(
        text="BERT 的第二个预训练任务是下一句预测",
        quotes=[
            Quote(
                "bert",
                "In addition to the masked language model, we also use a “next sentence "
                "prediction” task that jointly",
            )
        ],
    )
    [r] = verify_claims([claim], {"bert": bert_index})
    assert r.status == "supported"
    assert r.anchors[0].exact is True
    assert r.anchors[0].page == 2


def test_index_and_chunker_speak_the_same_words(bert_index):
    """索引的词流和切块器的文本必须接成同一个词，否则引用永远对不上。"""
    assert "addition" in bert_index.texts
    assert "addi" not in bert_index.texts


def test_truncated_quote_is_tolerated(indexes):
    """模型常在句中截断引用；只要前缀足够长且逐字命中，仍可通过并标记 truncated。"""
    claim = Claim(text="截断引用", quotes=[Quote("v7", "Provided proper attribution is provided, Google hereby grants permission to repr")])
    [r] = verify_claims([claim], indexes)
    assert r.status == "supported"
    assert r.anchors[0].truncated is True


def test_multi_doc_claim(indexes):
    """版本对比类结论应能同时锚定 v1 与 v7。"""
    claim = Claim(
        text="代码可用性表述变化",
        quotes=[
            Quote("v1", "We intend to make the code we used to train and evaluate our models available soon"),
            Quote("v7", "The code we used to train and evaluate our models is available at https://github.com/"),
        ],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status == "supported"
    assert {a.page for a in r.anchors} == {10}


# ---------------------------------------------------------------- 反例：必须拦住

def test_fabricated_quote_is_rejected(indexes):
    """完全编造的引用必须被驳回。"""
    claim = Claim(
        text="v7 声称模型在 ImageNet 上达到 90% 准确率",
        quotes=[Quote("v7", "Our model achieves 90.2 percent top-1 accuracy on ImageNet classification")],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status == "unsupported"


def test_claim_without_quotes_is_rejected(indexes):
    """没有证据的结论一律不采信，哪怕它碰巧是对的。"""
    [r] = verify_claims([Claim(text="v7 增加了致谢段落", quotes=[])], indexes)
    assert r.status == "unsupported"
    assert "未提供任何引用" in r.reason


def test_quote_from_wrong_document_is_rejected(indexes):
    """v7 才有的文字，硬说成来自 v1，必须驳回。"""
    claim = Claim(
        text="v1 有授权声明",
        quotes=[Quote("v1", "Provided proper attribution is provided, Google hereby grants permission")],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status == "unsupported"


def test_unknown_document_is_rejected(indexes):
    claim = Claim(text="来自不存在的文档", quotes=[Quote("v9", "Provided proper attribution is provided")])
    [r] = verify_claims([claim], indexes)
    assert r.status == "unsupported"
    assert "v9" in r.reason


def test_subtly_altered_quote_never_passes_as_supported(indexes):
    """只改一个词的真实句子，最多只能算 weak，绝不能判 supported。"""
    claim = Claim(
        text="篡改过的引用",
        quotes=[
            Quote(
                "v7",
                "The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor-api-v2",
            )
        ],
    )
    [r] = verify_claims([claim], indexes)
    assert r.status != "supported"


def test_fuzzy_similarity_is_never_printed_as_a_decimal_that_reads_like_pass():
    """「仅模糊命中」旁边的相似度不能印成 1.00。

    实测里出现过 reason=「仅模糊命中（最低相似度 1.00），原文需人工确认」——
    两位小数把 0.9996 四舍五入成了 1.00，用户只会看到「相似度 1.00」，
    以为这条被认可了。宁可多印两位，也不能让文字和判定打架。
    """
    from app.guardrails.citation import _format_similarity

    assert _format_similarity(0.9996) == "0.9996"
    assert _format_similarity(0.87) == "0.87"
    assert _format_similarity(1.0) == "1"
    assert _format_similarity(0.0) == "0"
