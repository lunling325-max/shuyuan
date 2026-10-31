"""检索词扩展（query expansion）+ 多路召回。

为什么需要这一层？
用户用中文问 "为什么要除以 sqrt(dk)"，论文是英文的。
BM25 把中文切成 0 个 token，一条都召回不了 —— 词法通道对跨语言完全失明。

做法：先让 LLM 把问题改写成 2~3 条**英文检索式**（同义词、术语、缩写各来一条），
每条各召回一批，再用 RRF 融合。这比只查一次稳，也比上多语言向量模型便宜。

失败必须能降级：模型挂了就往回退回原始问题，不能把问答整条链路拖死。
"""

from __future__ import annotations

from app.retrieval.bm25 import Hit, reciprocal_rank_fusion

EXPAND_SYSTEM = """You turn a research question into search queries for a keyword (BM25) index of an English paper.

Rules:
- Output 2-3 SHORT English queries (2-7 words each), not sentences.
- Use the paper's likely terminology (e.g. "scaled dot-product attention", "sqrt dk", "learning rate warmup").
- If the question is already English, give synonyms / alternate phrasings instead.
- Only output JSON: {"queries": ["...", "..."]}"""


def expand_query(question: str, *, llm_call=None) -> list[str]:
    """把问题改写成若干条检索式；任何失败都退回原问题。"""
    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    try:
        raw = llm_call(
            f"Question: {question}", system=EXPAND_SYSTEM, temperature=0.0
        )
        queries = [str(q).strip() for q in (raw.get("queries") or []) if str(q).strip()]
    except Exception:  # noqa: BLE001 —— 扩展是增强项，不能变成单点故障
        return [question]

    return queries or [question]


def retrieve(
    question: str,
    *,
    retriever=None,
    dense=None,
    k: int = 6,
    llm_call=None,
    expand: bool = True,
    queries: list[str] | None = None,
) -> list[Hit]:
    """多路召回 + RRF。词法通道与向量通道并行，各查每一条检索式。

    `dense` 是可选通道，**默认不传**：实测在这套语料上它让 Hit@3 从 100% 掉到 88.9%
    （docs/EXPERIMENTS.md E-004），所以只有显式做消融实验时才该打开。

    `queries` 允许调用方传入已经算好的检索式：CLI 要打印它们，
    不能让"屏幕上展示的那一套"和"真正检索用的那一套"是两次独立采样。
    """
    if queries is None:
        queries = expand_query(question, llm_call=llm_call) if expand else [question]
    searchers = [s for s in (retriever, dense) if s is not None]
    rankings: list[list[Hit]] = []
    for q in queries:
        for searcher in searchers:
            hits = searcher.search(q, k=k)
            if hits:
                rankings.append(hits)
    if not rankings:
        return []
    # 等权融合。给首条检索式加权（[1.0, 0.6, 0.6]）在固定检索式的对照实验里
    # 没有带来任何提升（两者都是 66.7%），所以保持最简单的等权。
    # 细节见 docs/EXPERIMENTS.md E-003。
    return reciprocal_rank_fusion(rankings, limit=k)
