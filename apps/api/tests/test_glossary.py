"""内置术语表（离线跨语言检索）的测试。

守住三件事：
1. **最长匹配**：`多头注意力` 不能切成 `多头` + `注意力`；
2. **不乱来**：纯英文问题和不含术语的中文问题都返回空，交给调用方决定，
   绝不硬造一条检索式；
3. **可解释**：命中了哪些词能一句话说清楚 —— 这张表的价值有一半在这里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.chunker import chunk_pdf
from app.rag.glossary import describe, expand_offline, extract, is_chinese
from app.rag.query import expand_query
from app.retrieval import BM25

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"


def test_longest_match_wins_over_shorter_keys():
    """`多头注意力` 是术语表里最长的那个键，必须整条命中。

    切碎了会变成 `多头` + `注意力`，两条都往检索式里加重复词，
    BM25 的词频被稀释，"多头注意力"那一节反而排不到第一。
    """
    keys = [zh for zh, _ in extract("多头注意力用了几个头？")]
    assert "多头注意力" in keys
    assert "注意力" not in keys  # 被更长的键吃掉了，不该再单独出现


def test_english_question_does_not_use_the_table():
    """纯英文本来就是 BM25 的强项，走这张表只会添乱。"""
    assert expand_offline("How many attention heads does BERT use?") == []
    assert extract("How many attention heads?") == []


def test_unmatched_chinese_returns_nothing_rather_than_guessing():
    """表里查不到就是查不到 —— 返回空，不硬拼一条可能带偏的检索式。"""
    assert expand_offline("这句话里没有一个专业名词") == []


def test_latin_tokens_in_the_question_are_kept():
    """问题里本来就有的英文词是最强锚点，必须原样进检索式。"""
    queries = expand_offline("注意力机制里的 sqrt(dk) 是怎么算的？")
    assert queries
    assert "attention" in queries[0]
    assert "sqrt" in queries[0] and "dk" in queries[0]


def test_a_question_whose_terms_are_already_english_is_left_alone():
    """`label smoothing`、`epsilon` 本来就是英文，原句直接交给 BM25 就够。

    这时候拼一条"半个中文问题"的检索式反而添乱，所以返回空、由调用方退回原问题。
    """
    assert expand_offline("label smoothing 的 epsilon 是多少？") == []


def test_one_query_is_returned_for_a_single_term():
    """只有一个术语时不硬凑第二条 —— 平行几路会互相稀释（实测 Hit@1 掉 3 题）。"""
    queries = expand_offline("注意力是什么？")
    assert len(queries) == 1
    assert "attention" in queries[0]


def test_describe_names_the_terms_it_matched():
    """界面上要能说清"我看懂了哪几个词"，这是它区别于黑盒改写的地方。"""
    text = describe("位置编码为什么用正弦函数？")
    assert "位置编码" in text and "正弦函数" in text
    assert "未命中" in describe("完全对不上的一句话")


def test_is_chinese_only_looks_for_han_characters():
    assert is_chinese("BERT 有多少参数？")
    assert not is_chinese("How many parameters does BERT have?")


def test_query_expansion_falls_back_to_the_table_when_the_model_dies():
    """模型挂了，中文提问不能跟着 0 召回 —— 这是加这张表的全部理由。"""

    def boom(*_a, **_k):
        raise RuntimeError("network down")

    queries = expand_query("位置编码为什么用正弦函数？", llm_call=boom)
    assert queries != ["位置编码为什么用正弦函数？"]
    assert any("positional encoding" in q for q in queries)


def test_model_output_comes_first_and_the_table_is_appended():
    """模型没挂的时候：模型那几条原样排在最前，术语表的排在后面。

    两条通道错的地方不一样（模型懂因果问法，表钉得住专有名词），
    所以是"并"不是"替换"—— 顺序保留是为了让人一眼看出谁说的。
    """

    def fake(user, *, system=None, **_):
        return {"queries": ["scaled dot-product attention"]}

    queries = expand_query("位置编码为什么用正弦函数？", llm_call=fake)
    assert queries[0] == "scaled dot-product attention"
    assert any("positional encoding" in q for q in queries)


def test_offline_expansion_finds_the_right_section():
    """端到端看一眼：纯中文问题只靠术语表，能不能把 3.5 节顶上来。"""
    bm25 = BM25(chunk_pdf(str(PDF_V7)))
    queries = expand_offline("位置编码为什么用正弦函数？")
    hits = bm25.search(queries[0], k=5)
    assert hits and "3.5" in hits[0].chunk.section
