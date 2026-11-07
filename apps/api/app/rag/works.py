"""同一篇论文的多个版本，在检索层只占一个名额。

库里常有「v1 扫描件 + v7 电子版」这种组合 —— 版本谱系页要的就是它们，
但**综述和写作 Copilot 的检索预算不该被同一篇吃掉两份**：6 篇的预算被
同一篇占去两格，页面写着"用到 5 篇"，其实库里只有 4 篇不同的工作。

单独放一个模块，是因为 rag 层的两个检索入口（综述、Copilot）都要用它，
而它们都不该在运行时反向依赖 HTTP 层的 registry —— 那条线上已经写着
"只在 TYPE_CHECKING 里拿 Paper 的类型"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from app.references import normalize_title

if TYPE_CHECKING:
    from app.api.registry import Paper


def one_version_per_work(papers: Sequence["Paper"]) -> list["Paper"]:
    """每个"工作"（同一标题的若干版本）只留一版。

    留哪一版，按这个顺序定：

    1. **不是扫描件**优先。OCR 会把 l 认成 1、0 认成 O，拿它当"作者原话"引用
       等于把识别错误当成原文，溯源反而失真 —— 越像原文越错。
    2. 版本号高的（作者发版时自己打的，对同一篇来说最硬）。
    3. PDF 生成日期新的。
    4. 最后按 id 定序，保证同样输入两次跑选的是同一版。
    """
    best: dict[str, tuple[tuple, "Paper"]] = {}
    for paper in papers:
        key = normalize_title(getattr(paper, "title", "") or "") or paper.id
        rank = (
            0 if getattr(paper, "scanned", False) else 1,
            getattr(paper, "arxiv_version", 0) or 0,
            getattr(paper, "created_at", "") or "",
            getattr(paper, "uploaded_at", "") or "",
        )
        kept = best.get(key)
        if kept is None or rank > kept[0]:
            best[key] = (rank, paper)
    return [paper for _rank, paper in sorted(best.values(), key=lambda kv: kv[1].id)]
