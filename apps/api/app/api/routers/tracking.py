"""追踪路由：arXiv 订阅的增删查、立即抓取、"看过了"的标记。

读和写分得很清楚，只有一处例外值得说明：

- `GET /api/tracking` **不写任何东西**。新增徽标靠 `viewed_at` 和条目自己的
  `first_seen` 比出来，标记"看过了"是前端渲染完之后发的一次 `POST /ack`。
  把这一步塞进 GET 会好看一点（少一次请求），但刷新两次页面徽标就没了 ——
  "新增"变成了"上一次刷新之后新增"，没人能理解那是什么。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app import tracking
from app.api import deps

router = APIRouter()

STORE = tracking.TrackStore(deps.STORE_DIR.parent / "tracking.json")


# ------------------------------------------------------------------ 模型

class TrackItemOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    abstract: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    comment: str = ""
    first_seen: str = ""
    is_new: bool = False
    in_library: bool = False


class TrackOut(BaseModel):
    id: str
    name: str = ""
    query: str
    categories: list[str] = Field(default_factory=list)
    interval_minutes: int
    created_at: str = ""
    last_run_at: str = ""
    last_status: str = ""
    last_note: str = ""
    next_run_at: str = ""
    due: bool = False
    new_count: int = 0
    seen_count: int = 0
    items_total: int = 0
    items: list[TrackItemOut] = Field(default_factory=list)


class SchedulerOut(BaseModel):
    enabled: bool
    tick_seconds: int


class TrackingOut(BaseModel):
    tracks: list[TrackOut] = Field(default_factory=list)
    scheduler: SchedulerOut
    note: str = ""


class TrackCreate(BaseModel):
    query: str
    name: str = ""
    categories: str = ""          # 逗号分隔，和「每日推荐」那一栏同一个写法
    interval_minutes: int = tracking.DEFAULT_INTERVAL_MINUTES


class AckRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ 组装

def _library_ids() -> set[str]:
    """库内已有的 arXiv id：条目上标一个「已入库」，省得有人重复下载一遍。

    库只有十几篇，每次现算最省事；真涨到几千篇再谈索引。
    """
    return {
        paper.arxiv_id
        for paper in deps.registry.all()
        if getattr(paper, "arxiv_id", "")
    }


# 一次返回多少条：条目表按 200 条封顶，但界面上一次只看得到最近二三十条，
# 把 200 条摘要一起塞进响应(~160KB)只是让"追踪"这一栏多等一会儿。
RETURN_ITEMS = 30


def _track_out(track: tracking.Track, library: set[str]) -> TrackOut:
    fresh = {item.arxiv_id for item in track.new_items()}
    now = tracking.utcnow()
    return TrackOut(
        id=track.id,
        name=track.name,
        query=track.query,
        categories=list(track.categories),
        interval_minutes=track.interval_minutes,
        created_at=track.created_at,
        last_run_at=track.last_run_at,
        last_status=track.last_status,
        last_note=track.last_note,
        next_run_at=tracking.iso(track.next_run_at()),
        due=track.is_due(now),
        new_count=len(fresh),
        seen_count=len(track.seen),
        items_total=len(track.items),
        items=[
            TrackItemOut(
                arxiv_id=item.arxiv_id,
                title=item.title,
                authors=list(item.authors),
                published=item.published,
                abstract=item.abstract,
                abs_url=item.abs_url,
                pdf_url=item.pdf_url,
                comment=item.comment,
                first_seen=item.first_seen,
                is_new=item.arxiv_id in fresh,
                in_library=item.arxiv_id in library,
            )
            for item in track.items[:RETURN_ITEMS]
        ],
    )


def _all_out() -> list[TrackOut]:
    library = _library_ids()
    return [_track_out(track, library) for track in STORE.all()]


# ------------------------------------------------------------------ 路由

@router.get("/api/tracking", response_model=TrackingOut)
def list_tracks() -> TrackingOut:
    return TrackingOut(
        tracks=_all_out(),
        scheduler=SchedulerOut(
            enabled=tracking.scheduler_enabled(), tick_seconds=tracking.tick_seconds()
        ),
        note=STORE.load_error,
    )


@router.post("/api/tracking", response_model=TrackOut, status_code=201)
def create_track(req: TrackCreate) -> TrackOut:
    cats = [c.strip() for c in req.categories.split(",") if c.strip()]
    try:
        track = STORE.add(
            query=req.query,
            categories=cats,
            interval_minutes=req.interval_minutes,
            name=req.name,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _track_out(track, _library_ids())


@router.delete("/api/tracking/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    if not STORE.remove(track_id):
        raise HTTPException(404, f"没有这条订阅：{track_id}")
    return Response(status_code=204)


@router.post("/api/tracking/{track_id}/run", response_model=TrackOut)
def run_track(track_id: str) -> TrackOut:
    try:
        track = tracking.run_one(STORE, track_id)
    except KeyError as exc:
        raise HTTPException(404, f"没有这条订阅：{track_id}") from exc
    except tracking.Busy as exc:
        raise HTTPException(409, str(exc)) from exc
    return _track_out(track, _library_ids())


@router.post("/api/tracking/run", response_model=TrackingOut)
def run_due(force: bool = False) -> TrackingOut:
    """跑到期的订阅；`force=true` 时不管到没到期全跑一遍。"""
    try:
        tracking.run_due(STORE, force=force)
    except tracking.Busy as exc:
        raise HTTPException(409, str(exc)) from exc
    return TrackingOut(
        tracks=_all_out(),
        scheduler=SchedulerOut(
            enabled=tracking.scheduler_enabled(), tick_seconds=tracking.tick_seconds()
        ),
        note=STORE.load_error,
    )


@router.post("/api/tracking/ack", response_model=TrackingOut)
def ack(req: AckRequest) -> TrackingOut:
    """前端把这一栏画完之后调一次：把这几条订阅的"上次看过"推到此刻。"""
    tracking.ack(STORE, req.ids)
    return TrackingOut(
        tracks=_all_out(),
        scheduler=SchedulerOut(
            enabled=tracking.scheduler_enabled(), tick_seconds=tracking.tick_seconds()
        ),
        note=STORE.load_error,
    )
