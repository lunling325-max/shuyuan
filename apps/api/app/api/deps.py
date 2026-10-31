"""路由层的共享件：注册表实例、404 包装、响应模型组装。

为什么单独一层：功能一多，每个功能自带一个 `APIRouter` 是唯一不互相踩的办法。
但路由模块若去 `import app.api.main` 拿 `registry`，就绕成了环 —— main 要
`include_router` 它们。所以把"谁都要用"的东西下沉到这里：

    deps.py  ←  routers/*.py
       ↑
     main.py（只负责挂载与生命周期）

代价是所有路由都隐式共享同一个注册表实例。这在单进程服务里是好事
（解析一次、处处命中缓存），但也意味着**路由模块不该自己去建第二个 registry**。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

from app.api.registry import Paper, PaperNotFound, PaperRegistry
from app.api.schemas import AnchorOut, EvidenceOut, PaperOut
from app.retrieval.bm25 import Hit
from app.versioning import Relation

ROOT = Path(__file__).resolve().parents[4]
# 测试用 PAPER_STORE_DIR 指到临时目录，别把测试产生的 PDF 混进真实资料库
STORE_DIR = Path(os.environ.get("PAPER_STORE_DIR") or ROOT / "data" / "papers")
WEB_DIR = ROOT / "apps" / "web"

registry = PaperRegistry(STORE_DIR)


def recommend_version(new: Paper, old: Paper) -> str:
    """两份 PDF 谁更新：只看生成日期，取不到就说不知道。

    为什么不用文件名里的 v1/v7：那是作者自己写的，不保证单调
    （有人 v2 之后又发了 v1.1，也有人在文件名里编号和内容无关）。
    两份日期都取不到时返回空串 —— 猜错了会把用户手里更新的版本删掉。
    """
    if new.created_at and old.created_at:
        if new.created_at > old.created_at:
            return "replace_existing"
        if new.created_at < old.created_at:
            return "keep_existing"
    return ""


def paper_out(p: Paper, dup: tuple[Paper, Relation] | None = None) -> PaperOut:
    out = PaperOut(
        id=p.id,
        filename=p.filename,
        title=p.title,
        pages=p.pages,
        chunks=len(p.chunks),
        uploaded_at=p.uploaded_at,
        created_at=p.created_at,
    )
    if dup is not None:
        other, relation = dup
        out.duplicate_of = other.id
        out.duplicate_title = other.title
        out.relation = relation.kind
        out.similarity = relation.similarity
        out.recommendation = recommend_version(p, other)
    return out


def anchor_out(a) -> AnchorOut:
    return AnchorOut(
        page=a.page,
        bbox=[round(v, 1) for v in a.rect],
        source_text=a.source_text,
        exact=a.exact,
        similarity=a.similarity,
        truncated=getattr(a, "truncated", False),
    )


def evidence_out(hits: list[Hit]) -> list[EvidenceOut]:
    return [
        EvidenceOut(
            rank=i,
            section=h.chunk.section,
            page_start=h.chunk.page_start,
            page_end=h.chunk.page_end,
            preview=h.chunk.preview(160),
            score=h.score,
            source=h.source,
        )
        for i, h in enumerate(hits, start=1)
    ]


def get_paper(paper_id: str) -> Paper:
    """按 id 取文献，取不到就 404 —— 路由层统一这一处，别各写各的文案。"""
    try:
        return registry.get(paper_id)
    except PaperNotFound as exc:
        raise HTTPException(404, f"没有这篇文献：{paper_id}") from exc
