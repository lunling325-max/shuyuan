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

from app.api.registry import Paper, PaperNotFound, PaperRegistry, order_versions
from app.api.schemas import AnchorOut, EvidenceOut, PaperOut
from app.retrieval.bm25 import Hit
from app.versioning import Relation

ROOT = Path(__file__).resolve().parents[4]
# 测试用 PAPER_STORE_DIR 指到临时目录，别把测试产生的 PDF 混进真实资料库
STORE_DIR = Path(os.environ.get("PAPER_STORE_DIR") or ROOT / "data" / "papers")
WEB_DIR = ROOT / "apps" / "web"

registry = PaperRegistry(STORE_DIR)


def recommend_version(new: Paper, old: Paper) -> tuple[str, str]:
    """这次上传的这份该不该顶掉库里那份 —— 判据直接复用 `order_versions`。

    以前这里只看 PDF 元数据里的生成日期，而版本谱系页用的是 arXiv 版本号，
    于是同一对文件在两个页面上会给出相反的结论。比如库里存着 v2、用户新传
    一份 v7，但 v7 那份是早些时候导出的、v2 是后来重导的 —— 版本谱系说 v7
    是新版，上传页却建议"保留库里那份"，两个结论摆在一起没法解释。

    `created_at` 的本义是"这份 PDF 什么时候导出的"，不是"这篇论文什么时候
    发版的"：谁把 2017 年的旧稿在 2024 年重新导出一遍，日期反而更"新"。

    判据只落到上传时间时返回空串 —— 那只说明"谁先传的"，据此建议删哪一份
    都是猜，而猜错删掉的是用户手里唯一的那份。

    返回值是 `(建议, 依据)`：依据一并回给前端，因为界面要照着它说话 ——
    "版本号更高"和"PDF 生成日期更晚"是两回事，同一个建议用错依据就变成假话
    （v7 顶掉 v1，但 v1 那份是后来重导的，说"日期更晚"就与事实相反）。
    """
    older, _newer, by = order_versions(new, old)
    if by == "uploaded_at":
        return "", ""
    return ("replace_existing" if older is old else "keep_existing"), by


def paper_out(p: Paper, dup: tuple[Paper, Relation] | None = None) -> PaperOut:
    out = PaperOut(
        id=p.id,
        filename=p.filename,
        title=p.title,
        pages=p.pages,
        chunks=len(p.chunks),
        uploaded_at=p.uploaded_at,
        created_at=p.created_at,
        arxiv_id=p.arxiv_id,
        arxiv_version=p.arxiv_version,
        arxiv_date=p.arxiv_date,
        scanned=p.scanned,
    )
    if dup is not None:
        other, relation = dup
        out.duplicate_of = other.id
        out.duplicate_title = other.title
        out.relation = relation.kind
        out.similarity = relation.similarity
        out.recommendation, out.recommendation_by = recommend_version(p, other)
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
        raise HTTPException(404, f"库内不存在该文献：{paper_id}") from exc
