"""检索词扩展（query expansion）+ 多路召回。

为什么需要这一层？
用户用中文问 "为什么要除以 sqrt(dk)"，论文是英文的。
BM25 把中文切成 0 个 token，一条都召回不了 —— 词法通道对跨语言完全失明。

做法：先让 LLM 把问题改写成 2~3 条**英文检索式**（同义词、术语、缩写各来一条），
每条各召回一批，再用 RRF 融合。这比只查一次稳，也比上多语言向量模型便宜。

失败必须能降级：模型挂了就退回**内置术语表**（`app.rag.glossary`），表也查不到
才退回原始问题 —— 不能把问答整条链路拖死，也不能让"中文提问"这件事
单点依赖一次大模型调用。
"""

from __future__ import annotations

from app.rag.glossary import expand_offline
from app.retrieval.bm25 import Hit, reciprocal_rank_fusion

EXPAND_SYSTEM = """You turn a research question into search queries for a keyword (BM25) index of an English paper.

Rules:
- Output 2-3 SHORT English queries (2-7 words each), not sentences.
- Use the paper's likely terminology (e.g. "scaled dot-product attention", "sqrt dk", "learning rate warmup").
- If the question is already English, give synonyms / alternate phrasings instead.
- Only output JSON: {"queries": ["...", "..."]}"""


def with_offline(question: str, queries: list[str]) -> list[str]:
    """把术语表的检索式并进已有的一组（去重、保持原顺序），空则退回原问题。

    为什么要**并**而不是"二选一"：两条通道错的地方不一样。
    模型改写擅长读懂"会影响吗"这种因果问法，术语表擅长钉住专有名词的精确词形。
    实测 18 题开发集上，模型单独 15/18、术语表单独 16/18，
    合并后 **16/18 且 Hit@3 补齐到 18/18**；留出集 13 题上合并也是 13/13。
    两条都不比对方差，就没有理由只留一条。
    """
    merged = list(queries)
    for extra in expand_offline(question):
        if extra not in merged:
            merged.append(extra)
    return merged or [question]


def expand_query(question: str, *, llm_call=None) -> list[str]:
    """把问题改写成若干条检索式；模型挂了也还有**离线术语表**兜着。"""
    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    try:
        raw = llm_call(
            f"Question: {question}", system=EXPAND_SYSTEM, temperature=0.0
        )
        queries = [str(q).strip() for q in (raw.get("queries") or []) if str(q).strip()]
    except Exception:  # noqa: BLE001 —— 扩展是增强项，不能变成单点故障
        # 大模型挂了不等于跨语言检索要一起挂：内置术语表是不联网的那条备胎。
        # 纯英文问题表里查不到东西，`expand_offline` 返回空，退回原问题，
        # 行为与加表之前完全一致。
        return with_offline(question, [])

    return with_offline(question, queries)


# 低于这个块数不做两级检索：25 块的论文上，粗筛省的是毫秒，赔的是召回
MIN_CHUNKS_FOR_SECTIONS = 150


def _section_index(retriever):
    """取（或懒建）该检索器的章节索引；小文档返回 None，表示"不适用"。"""
    cached = getattr(retriever, "_section_index_cache", None)
    if cached is not None:
        return cached if cached is not False else None
    chunks = getattr(retriever, "chunks", None) or []
    if len(chunks) < MIN_CHUNKS_FOR_SECTIONS:
        retriever._section_index_cache = False
        return None
    from app.retrieval.sections import build

    index = build(chunks)
    retriever._section_index_cache = index
    return index


def retrieve(
    question: str,
    *,
    retriever=None,
    dense=None,
    k: int = 6,
    llm_call=None,
    expand: bool = True,
    queries: list[str] | None = None,
    trace: dict | None = None,
) -> list[Hit]:
    """多路召回 + RRF。词法通道与向量通道并行，各查每一条检索式。

    `dense` 是可选通道，**默认不传**：实测在这套语料上它让 Hit@3 从 100% 掉到 88.9%
    （docs/EXPERIMENTS.md E-004），所以只有显式做消融实验时才该打开。

    `queries` 允许调用方传入已经算好的检索式：CLI 要打印它们，
    不能让"屏幕上展示的那一套"和"真正检索用的那一套"是两次独立采样。
    """
    if queries is None:
        queries = expand_query(question, llm_call=llm_call) if expand else [question]

    # 两级检索：大文档先在章节层粗筛，再只在候选章节内做块级检索。
    # 只在纯词法通道上做 —— 向量通道没有"章节"这个概念，混进来只会让两条路不一致。
    # 粗筛结果不够 k 条时**整条退回全局检索**：宁可多花时间，也不能让召回缩水。
    if trace is not None:
        trace.update({"narrowed": False, "candidates": 0, "corpus": 0})
    if retriever is not None and dense is None:
        sections = _section_index(retriever)
        if sections is not None:
            corpus = len(retriever.chunks)
            candidates = sections.candidates(queries)
            if candidates:
                narrowed: list[list[Hit]] = []
                for q in queries:
                    hits = retriever.search(q, k=k, restrict=candidates)
                    if hits:
                        narrowed.append(hits)
                fused_narrow = reciprocal_rank_fusion(narrowed, limit=k) if narrowed else []
                if len(fused_narrow) >= k:
                    # 只有真的用了粗筛才记——退回去用全局时，别让界面说"粗筛过"
                    if trace is not None:
                        trace.update({"narrowed": True, "candidates": len(candidates),
                                      "corpus": corpus})
                    return fused_narrow

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
