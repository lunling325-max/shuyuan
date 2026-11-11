"""跨文献对比：`POST /api/papers/compare`。

赛题模块二要的是"多篇论文跨文献对比"（BERT vs Transformer、GPT-1/2/3 的演进）。
单篇问答那条链路一次只认一篇，所以这里单开一个入口：**同一个问题分别去两到三篇里
各召回一批证据**，一次生成，再按各自的原文校验。

为什么不用 SSE：
- 对比答案短（几条结论），流式带来的"看着它在动"收益很小；
- 真正耗时的部分是三篇的检索与校验，这两步在流式里也发不出去；
- 前端只要一个"对比中…"的状态就够，出错时还能用 HTTP 状态码如实表达。

与单篇问答一致的底线：
- 没有 key → 200 + `degraded`，证据照样返回，界面不白屏；
- 取不到文献 → 404；同一篇传两次 → 400（那不是对比，是复制）；
- 模型给的引用逐条标注出自哪一篇，校验时按 paper_id 各自对齐。
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api import deps
from app.rag.compare import (
    LABELS,
    SYSTEM_PROMPT,
    Comparison,
    PaperRef,
    build_user,
    parse_claims,
    verify_comparison,
)
from app.rag.query import expand_query, retrieve

router = APIRouter()

# 每篇最多取几块证据：三篇 × 4 块已经足够做对比，再多只是把上下文撑长。
MAX_K = 4


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(
        min_length=2, max_length=3, description="要对比的文献 id，2 到 3 篇"
    )
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=3, description="每篇召回几块证据，夹到 1..4")

    def clipped_k(self) -> int:
        return max(1, min(int(self.k or 3), MAX_K))


class ComparePaperOut(BaseModel):
    label: str
    id: str
    title: str


class CompareAnchorOut(BaseModel):
    paper: str = Field(description="这条引用出自哪一篇（A/B/C）")
    page: int
    bbox: list[float]
    source_text: str
    exact: bool
    similarity: float


class CompareClaimOut(BaseModel):
    text: str
    status: str
    reason: str
    anchors: list[CompareAnchorOut]


class CompareEvidenceOut(BaseModel):
    chunk_index: int
    page: int
    section: str
    score: float
    preview: str


class CompareOut(BaseModel):
    papers: list[ComparePaperOut]
    claims: list[CompareClaimOut]
    stats: dict[str, int]
    evidence: dict[str, list[CompareEvidenceOut]]
    took_ms: int
    degraded: bool = False
    notes: list[str] = Field(default_factory=list, description="降级或跳过的原因")


def _resolve(paper_ids: list[str]) -> list[tuple[str, object]]:
    """按顺序取文献并配标签。404 在这里抛，状态码还来得及如实表达。"""
    if len(set(paper_ids)) != len(paper_ids):
        raise HTTPException(400, "同一篇文献传了两次，这不是对比")
    return [(LABELS[i], deps.get_paper(pid)) for i, pid in enumerate(paper_ids)]


def _retrieve_all(pairs, question: str, k: int, *, expand: bool) -> tuple[list[str], dict]:
    """每篇各召回一批。检索式只扩展一次，三篇共用同一组检索词 ——

    分别扩展会让"B 里召回的都是 B 的术语"，两边看起来都很有料，
    实际问的已经不是同一个问题。
    """
    queries = expand_query(question) if expand else [question]
    evidence: dict[str, list] = {}
    for label, paper in pairs:
        evidence[label] = retrieve(question, retriever=paper.bm25, k=k, queries=queries)
    return queries, evidence


def _to_comparison(pairs, question: str, evidence: dict, raw: dict) -> Comparison:
    """解析模型输出并按各自的原文校验。TextIndex 首次构建可能较慢，由线程池调用。"""
    papers = [PaperRef(label=label, id=paper.id, title=paper.title) for label, paper in pairs]
    comparison = Comparison(question=question, papers=papers, evidence=evidence)
    claims = parse_claims(raw, {label: paper.id for label, paper in pairs})
    indexes = {paper.id: paper.text_index for _label, paper in pairs}
    comparison.claims = verify_comparison(claims, indexes)
    return comparison


@router.post("/api/papers/compare", response_model=CompareOut)
async def compare_papers(req: CompareRequest) -> CompareOut:
    t0 = time.perf_counter()
    pairs = _resolve(req.paper_ids)
    k = req.clipped_k()

    try:
        _queries, evidence = await run_in_threadpool(
            _retrieve_all, pairs, req.question, k, expand=True
        )
    except Exception:  # noqa: BLE001 —— 检索失败不该把请求变成 500
        raise HTTPException(503, "检索阶段失败，请稍后重试") from None

    def out(comparison: Comparison, *, degraded: bool, notes: list[str]) -> CompareOut:
        return CompareOut(
            papers=[
                ComparePaperOut(label=p.label, id=p.id, title=p.title)
                for p in comparison.papers
            ],
            claims=[
                CompareClaimOut(
                    text=c.claim.text,
                    status=c.status,
                    reason=c.reason,
                    anchors=[
                        CompareAnchorOut(
                            paper=next(
                                (lb for lb, pp in pairs if pp.id == a.doc), a.doc or "",
                            ),
                            page=a.page,
                            bbox=[round(v, 1) for v in a.rect],
                            source_text=a.source_text,
                            exact=a.exact,
                            similarity=a.similarity,
                        )
                        for a in c.anchors
                    ],
                )
                for c in comparison.claims
            ],
            stats=comparison.stats,
            evidence={
                label: [
                    CompareEvidenceOut(
                        chunk_index=h.chunk.index,
                        page=h.chunk.page_start,
                        section=h.chunk.section,
                        score=h.score,
                        preview=h.chunk.preview(160),
                    )
                    for h in hits
                ]
                for label, hits in comparison.evidence.items()
            },
            took_ms=int((time.perf_counter() - t0) * 1000),
            degraded=degraded,
            notes=notes,
        )

    papers = [PaperRef(label=label, id=paper.id, title=paper.title) for label, paper in pairs]
    skeleton = Comparison(question=req.question, papers=papers, evidence=evidence)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        return out(
            skeleton,
            degraded=True,
            notes=["大模型未配置（缺少 DEEPSEEK_API_KEY），以上仅为各篇召回的原文块"],
        )

    titles = {label: paper.title for label, paper in pairs}
    user = build_user(req.question, evidence, titles)

    from app.llm.client import LLMError, chat_json

    try:
        raw = await run_in_threadpool(chat_json, user, system=SYSTEM_PROMPT, temperature=0.1)
    except LLMError as exc:
        text = str(exc).strip()
        message = (
            "大模型未配置（缺少 DEEPSEEK_API_KEY），以上仅为各篇召回的原文块"
            if "DEEPSEEK_API_KEY" in text
            else (text[:300] or "大模型暂不可用")
        )
        return out(skeleton, degraded=True, notes=[message])
    except Exception:  # noqa: BLE001
        return out(skeleton, degraded=True, notes=["模型返回异常，以上仅为各篇召回的原文块"])

    try:
        comparison = await run_in_threadpool(_to_comparison, pairs, req.question, evidence, raw)
    except Exception:  # noqa: BLE001
        return out(skeleton, degraded=True, notes=["引用校验失败，以上仅为各篇召回的原文块"])

    return out(comparison, degraded=False, notes=[])
