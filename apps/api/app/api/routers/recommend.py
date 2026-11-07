"""推荐路由：arXiv 最新论文 × 本地文献库相关性 × 开源代码 × 点赞/踩。

三个设计决定：

1. **缓存 15 分钟**。追踪页会被人反复打开，每次真出网就是 9 秒白屏。
   缓存键带上"分类 + 条数 + 天数"，换条件就重新取；`fresh=true` 可以强制刷新。
   这就是"定期刷新追踪"的落地形态 —— 对用户是透明的，对 arXiv 是低频的。
2. **相对时间而不是绝对时间**。`days=7` 的语义是"最近七天提交的"，
   用的是服务器当前日期，过滤发生在打分之前 —— 先打分再过滤会把窗口外的
   高分论文塞给用户。
3. **投票按用户分开存**，`user` 缺省为 `local`。没有登录体系时这就是"本机这个人"，
   以后接多用户只需要前端把 user 传成真实 id，存储格式不用动。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import recommender
from app.api import deps
from app.tools import arxiv as arxiv_tool

router = APIRouter()

CACHE_TTL_SECONDS = 15 * 60
_CACHE: dict[tuple, tuple[float, arxiv_tool.SearchResult]] = {}
_feedback = recommender.FeedbackStore(deps.STORE_DIR.parent / "feedback.json")

# 上一次成功刷到的整批候选，落盘一份。内存缓存活不过进程重启，
# 而"刚打开服务就被限流/断网"正是演示最怕的那一下 —— 有这份文件，
# 页面至少还能给出上次那批，并注明是几小时前的。
BATCH_PATH = deps.STORE_DIR.parent / "recommend_batch.json"
BATCH_TTL_SECONDS = 12 * 3600


class MatchOut(BaseModel):
    paper_id: str
    title: str
    similarity: float


class RecommendItemOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    abstract: str = ""
    comment: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    score: float = 0.0
    relevance: float = 0.0
    matches: list[MatchOut] = Field(default_factory=list)
    code_url: str = ""
    reason: str = ""
    votes_up: int = 0
    votes_down: int = 0
    my_vote: str = ""


class RecommendOut(BaseModel):
    categories: list[str] = Field(default_factory=list)
    days: int = 7
    items: list[RecommendItemOut] = Field(default_factory=list)
    degraded: bool = False
    note: str = ""
    library_size: int = 0
    cached: bool = False


class VoteRequest(BaseModel):
    arxiv_id: str
    vote: str = "up"          # up / down / 空串=取消
    user: str = "local"


class VoteOut(BaseModel):
    arxiv_id: str
    votes_up: int
    votes_down: int
    my_vote: str


def _save_batch(result: arxiv_tool.SearchResult) -> None:
    """把这一批候选原样存下来；写盘失败不影响这次请求。"""
    payload = {
        "query": result.query,
        "saved_at": time.time(),
        "entries": [asdict(entry) for entry in result.results],
    }
    try:
        BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BATCH_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, BATCH_PATH)
    except OSError:
        pass


def _load_batch() -> tuple[arxiv_tool.SearchResult, int] | None:
    """返回 (上次那批, 存了多久的秒数)；太旧或读不出来就当作没有。"""
    try:
        payload = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
        age = int(time.time() - float(payload.get("saved_at") or 0))
        entries = [arxiv_tool.ArxivEntry(**item) for item in payload.get("entries") or []]
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if not entries or age < 0 or age > BATCH_TTL_SECONDS:
        return None
    return (
        arxiv_tool.SearchResult(
            query=str(payload.get("query") or ""), results=entries, degraded=False
        ),
        age,
    )


def _age_text(seconds: int) -> str:
    """说人话：几分钟 / 几小时。落盘那份可能存了五小时，别写「312 分钟前」。"""
    minutes = max(1, seconds // 60)
    if minutes < 90:
        return f"{minutes} 分钟前"
    return f"{minutes // 60} 小时前"


def _latest_cached(categories: list[str], fetch_limit: int, fresh: bool):
    """返回 (结果, 是否来自缓存, 缓存龄秒数, 这次失败的说明)。

    失败时退到上一次成功的结果（哪怕已经过期）：arXiv 限流时（实测 429 要等
    50 秒才回）整页空白是最差的体验，而"这是 8 分钟前刷到的"完全够用 ——
    推荐页给的是候选论文，不是必须秒级的实时数据。内存里没有就翻落盘那份，
    这样重启服务后的第一次刷新失败也不至于白屏。
    """
    key = (tuple(categories), fetch_limit)
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and not fresh and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1], True, int(now - hit[0]), ""

    result = arxiv_tool.latest(categories, limit=fetch_limit)
    if not result.degraded:
        _CACHE[key] = (now, result)
        _save_batch(result)
        return result, False, 0, ""

    if hit is not None:
        return hit[1], True, int(now - hit[0]), result.note
    saved = _load_batch()
    if saved is not None:
        return saved[0], True, saved[1], result.note
    return result, False, 0, result.note


@router.get("/api/recommend", response_model=RecommendOut)
def recommend(
    categories: str = "",
    limit: int = 12,
    days: int = 7,
    fresh: bool = False,
    user: str = "local",
) -> RecommendOut:
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    limit = max(1, min(int(limit), 40))
    days = max(1, min(int(days), 60))
    # 多抓一些再按时间窗过滤：只抓 limit 条的话，窗口外的会挤掉窗口内的。
    # 但别抓太多 —— 一条 Atom 条目约 2.5KB，48 条就是 120KB，而这台机器上
    # Python 的 TLS 吞吐只有 10KB/s 上下，多抓的每一千字节都是白等的一秒。
    result, cached, age, failure = _latest_cached(cats, min(limit * 2, 40), fresh)

    cutoff = date.today() - timedelta(days=days)
    entries = [
        entry for entry in result.results
        if _on_or_after(entry.published, cutoff)
    ]

    profile = recommender.build_profile(deps.registry.all())
    ranked = recommender.rank(entries, profile, today=date.today().isoformat(), limit=limit)

    items: list[RecommendItemOut] = []
    for rec in ranked:
        entry = rec.entry
        up, down, mine = _feedback.counts(entry.arxiv_id, user)
        items.append(
            RecommendItemOut(
                arxiv_id=entry.arxiv_id,
                title=entry.title,
                authors=entry.authors,
                published=entry.published,
                abstract=entry.abstract,
                comment=entry.comment,
                abs_url=entry.abs_url,
                pdf_url=entry.pdf_url,
                score=rec.score,
                relevance=rec.relevance,
                matches=[MatchOut(**asdict(m)) for m in rec.matches],
                code_url=rec.code_url,
                reason=rec.reason,
                votes_up=up,
                votes_down=down,
                my_vote=mine,
            )
        )

    note = failure
    if failure and cached:
        note = f"{failure}；下面是 {_age_text(age)}刷到的那批"

    return RecommendOut(
        categories=cats or ["cs.CL", "cs.LG", "cs.AI", "cs.CV"],
        days=days,
        items=items,
        # degraded 说的是"这次刷新失败"，不是"这批数据不好"
        degraded=bool(failure),
        note=note,
        library_size=len(profile),
        cached=cached,
    )


def _on_or_after(published: str, cutoff: date) -> bool:
    """日期解析不了就放行：宁可多给一条，也不要因为格式怪就漏掉新论文。"""
    if not published:
        return True
    try:
        return date.fromisoformat(published[:10]) >= cutoff
    except ValueError:
        return True


@router.post("/api/recommend/vote", response_model=VoteOut)
def vote(req: VoteRequest) -> VoteOut:
    try:
        arxiv_id = arxiv_tool.normalize_arxiv_id(req.arxiv_id)
    except arxiv_tool.InvalidArxivId as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        up, down, mine = _feedback.cast(arxiv_id, req.user, req.vote)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return VoteOut(arxiv_id=arxiv_id, votes_up=up, votes_down=down, my_vote=mine)
