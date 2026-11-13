"""arXiv 定期追踪：一份活过重启的订阅表，加上"哪些已经见过"的记忆。

和「每日推荐」的区别不是界面，是状态：

- 推荐是每次打开都现算一遍（按库内相关性排序），**没有记忆**，两小时后再看还是那批；
- 追踪记得**哪些已经见过**，所以"新增"才是一个能被证明的说法。

没有那张已见集合，每隔六小时抓一次，抓回来的永远是同一批 30 条，
界面上写"新增 30 篇"就是在骗人 —— 新增的定义只能是"我没见过"。

三个东西各自单独存（同一个 JSON）：

| 字段 | 是什么 | 为什么不能合并 |
|---|---|---|
| `tracks` | 订阅：分类 + 关键词 + 间隔 | 这是用户设的，删了就没了 |
| `seen` | arxiv_id → 第一次见到的时间 | **只增不减**，是"新增"的唯一依据；和条目列表分开存，条目被截断也不会让它失忆 |
| `items` | 抓回来的条目，最新在前，每条约 200 条封顶 | 可以截断、可以重来，丢了不影响判断 |

两条容易被忽略的规则：

**1. 第一次抓取不算"新增"。** 刚建的订阅第一轮拿回来 20 条，那不是"这几天冒出来 20 篇"，
那是一份基线。所以首轮只把 id 记进 `seen`、条目照常列出，但新增数记 0，
`status` 写成 `baseline`，界面上说的是"首次抓取，作为基线记下 20 条，不计入新增"。

**2. 抓取失败也要写 `last_run_at`。** 不写的话调度器会每分钟重试一次，
而 arXiv 限流（实测 429 要等 50 秒才回）时这是自己给自己加码。
失败只影响这一轮，已见的集合不动。

为什么不用 APScheduler 之类的定时库：单进程、单用户、一个 asyncio 循环，
60 秒醒一次看一眼"有没有到期的"就够；多引一个依赖就多一个起不来的可能。
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.tools import arxiv as arxiv_tool

# 一轮最多拿回多少条。40 条约 100 KB —— 这台机器到 arXiv 的 TLS 吞吐约 10 KB/s，
# 再多就是让「立即抓取」这个按钮按下去要等更久。
FETCH_LIMIT = 40
# 每个订阅保留多少条。只截条目，不截 `seen`：条目是给人看的，`seen` 是判据。
MAX_ITEMS = 200
# 间隔下限 30 分钟：arXiv 自己也要求客户端别太频繁，而且没人需要半分钟一刷。
MIN_INTERVAL_MINUTES = 30
MAX_INTERVAL_MINUTES = 7 * 24 * 60
DEFAULT_INTERVAL_MINUTES = 6 * 60
# 摘要截断：条目表要给人看标题和一句摘要，不需要整段（Atom 里一条摘要 1–2 KB）
ABSTRACT_CHARS = 600

_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "this", "that",
    "these", "those", "are", "was", "were", "its", "our", "their", "via", "using",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    """毫秒不够、百分之一秒也不行：两次抓取可能落在同一秒里（双击「立即抓取」），
    "新增"的判据是 `first_seen > viewed_at`，取整到秒就会把第二次抓到的
    当成"早就看过"—— 而它恰恰是刚到的。"""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_time(text: str) -> datetime | None:
    """解析落盘的 ISO 时间；空的或坏的一律当"没有"。"""
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def query_terms(query: str) -> list[str]:
    """把关键词切成可比较的词。中文词切出来也没关系 —— 它匹配不上英文摘要而已。"""
    words = re.split(r"[,;，；、\s]+", (query or "").lower())
    return [word for word in words if len(word) >= 3 and word not in _STOPWORDS]


def _matches(title: str, abstract: str, terms: list[str]) -> bool:
    """至少命中**两个**词；只写了一个词时就是那一个。空关键词 = 不筛。

    为什么不是"任一命中"：实测 `retrieval augmented generation` 按任一命中筛，
    40 条里留下 10 条，而其中一半是"视频生成"—— 它们命中 `generation` 一个词，
    跟检索增强没有关系。常见词（generation / learning / model）会让"任一"等于不筛。

    为什么也不是"全部命中"：四个分类一天四十条，全中的话绝大多数订阅永远抓不到
    东西，"定期追踪"于是变成"定期无所获" —— 那比噪声更难用。
    """
    if not terms:
        return True
    text = f"{title} {abstract}".lower()
    need = 1 if len(terms) == 1 else 2
    return sum(1 for term in terms if term in text) >= need


@dataclass
class TrackItem:
    arxiv_id: str
    title: str
    published: str = ""
    first_seen: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    comment: str = ""

    @classmethod
    def from_entry(cls, entry: arxiv_tool.ArxivEntry, first_seen: str) -> "TrackItem":
        return cls(
            arxiv_id=entry.arxiv_id,
            title=entry.title,
            published=entry.published,
            first_seen=first_seen,
            authors=list(entry.authors),
            abstract=(entry.abstract or "")[:ABSTRACT_CHARS],
            abs_url=entry.abs_url,
            pdf_url=entry.pdf_url,
            comment=entry.comment,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "TrackItem":
        arxiv_id = str(raw.get("arxiv_id") or "").strip()
        if not arxiv_id:
            raise ValueError("条目缺 arxiv_id")
        return cls(
            arxiv_id=arxiv_id,
            title=str(raw.get("title") or ""),
            published=str(raw.get("published") or ""),
            first_seen=str(raw.get("first_seen") or ""),
            authors=[str(a) for a in raw.get("authors") or []],
            abstract=str(raw.get("abstract") or ""),
            abs_url=str(raw.get("abs_url") or ""),
            pdf_url=str(raw.get("pdf_url") or ""),
            comment=str(raw.get("comment") or ""),
        )


@dataclass
class Track:
    id: str
    query: str
    categories: list[str]
    interval_minutes: int
    created_at: str
    # 显示名。默认等于关键词，用户不填就省一道必填项
    name: str = ""
    last_run_at: str = ""
    # ""（还没跑过）/ baseline（首轮基线）/ ok / degraded
    last_status: str = ""
    last_note: str = ""
    # 上次"看"过的时间：新增徽标就是拿 first_seen 跟它比
    viewed_at: str = ""
    seen: dict[str, str] = field(default_factory=dict)
    items: list[TrackItem] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.name or self.query or self.id

    def next_run_at(self) -> datetime:
        base = parse_time(self.last_run_at) or parse_time(self.created_at) or utcnow()
        return base + timedelta(minutes=max(MIN_INTERVAL_MINUTES, self.interval_minutes))

    def is_due(self, now: datetime | None = None) -> bool:
        """还没跑过的订阅**立刻到期**：刚建好的追踪要拿基线，不该干等一个间隔。

        （这里不能直接拿 `next_run_at()` 比 —— 那样新建的订阅要等到
        "建好时间 + 间隔"才算到期，演示时建完点开是一行空的。）
        """
        if not self.last_run_at:
            return True
        return (now or utcnow()) >= self.next_run_at()

    def new_items(self) -> list[TrackItem]:
        """上次看过之后抓到的。`viewed_at` 空着（从没看过）就全是新的。"""
        since = parse_time(self.viewed_at)
        if since is None:
            return list(self.items)
        return [item for item in self.items if (parse_time(item.first_seen) or since) > since]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "query": self.query,
            "categories": list(self.categories),
            "interval_minutes": self.interval_minutes,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_note": self.last_note,
            "viewed_at": self.viewed_at,
            "seen": dict(self.seen),
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Track":
        track_id = str(raw.get("id") or "").strip()
        if not track_id:
            raise ValueError("订阅缺 id")
        items: list[TrackItem] = []
        for item in raw.get("items") or []:
            try:
                items.append(TrackItem.from_dict(item))
            except (ValueError, TypeError, AttributeError):
                continue  # 单条坏掉不该带走整张订阅表
        seen = {
            str(key): str(value)
            for key, value in (raw.get("seen") or {}).items()
            if str(key).strip()
        }
        return cls(
            id=track_id,
            query=str(raw.get("query") or ""),
            categories=[str(c) for c in raw.get("categories") or []],
            interval_minutes=int(raw.get("interval_minutes") or DEFAULT_INTERVAL_MINUTES),
            created_at=str(raw.get("created_at") or ""),
            name=str(raw.get("name") or ""),
            last_run_at=str(raw.get("last_run_at") or ""),
            last_status=str(raw.get("last_status") or ""),
            last_note=str(raw.get("last_note") or ""),
            viewed_at=str(raw.get("viewed_at") or ""),
            seen=seen,
            items=items,
        )


class TrackStore:
    """订阅表：内存里一份，改完立刻落盘（tmp + 原子替换，和别处的存法一致）。

    读坏了怎么办：把原始文件另存成 `*.bad` 再从头开始，接口会把这件事说出来。
    什么都不做地覆盖掉是最坏的选择 —— 那是把用户设过的订阅静默清空。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.load_error = ""
        self._lock = threading.RLock()
        self._tracks: list[Track] = []
        self._loaded = False

    # ---------------------------------------------------------------- 读写

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    def load(self) -> None:
        with self._lock:
            self._loaded = True
            self.load_error = ""
            if not self.path.exists():
                self._tracks = []
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                raw_tracks = payload.get("tracks") or []
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self.load_error = f"订阅表读取失败（{exc}），已另存为 {self.path.name}.bad 并从头开始"
                try:
                    self.path.with_suffix(self.path.suffix + ".bad").write_bytes(
                        self.path.read_bytes()
                    )
                except OSError:
                    pass
                self._tracks = []
                return
            tracks: list[Track] = []
            for raw in raw_tracks:
                try:
                    tracks.append(Track.from_dict(raw))
                except (ValueError, TypeError, AttributeError):
                    continue
            self._tracks = tracks

    def save(self) -> None:
        with self._lock:
            payload = {"version": 1, "tracks": [track.to_dict() for track in self._tracks]}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass  # 写不进去不影响这次请求，下次改动会再试

    # ---------------------------------------------------------------- 查询

    def all(self) -> list[Track]:
        self._ensure()
        with self._lock:
            return list(self._tracks)      # 浅拷贝：调用方可能改 track 字段再 save

    def get(self, track_id: str) -> Track | None:
        self._ensure()
        with self._lock:
            return next((t for t in self._tracks if t.id == track_id), None)

    # ---------------------------------------------------------------- 增删

    def add(
        self,
        query: str,
        categories: list[str] | None = None,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        name: str = "",
    ) -> Track:
        query = " ".join((query or "").split())
        if not query:
            raise ValueError("追踪关键词不能为空")
        cats = [c.strip() for c in (categories or []) if c and c.strip()] or [
            "cs.CL", "cs.LG", "cs.AI", "cs.CV",
        ]
        minutes = int(interval_minutes or DEFAULT_INTERVAL_MINUTES)
        minutes = max(MIN_INTERVAL_MINUTES, min(minutes, MAX_INTERVAL_MINUTES))
        track = Track(
            id=uuid.uuid4().hex[:12],
            query=query,
            categories=cats,
            interval_minutes=minutes,
            created_at=iso(utcnow()),
            name=" ".join((name or "").split()),
        )
        with self._lock:
            self._ensure()
            self._tracks.append(track)
            self.save()
        return track

    def remove(self, track_id: str) -> bool:
        with self._lock:
            self._ensure()
            before = len(self._tracks)
            self._tracks = [t for t in self._tracks if t.id != track_id]
            if len(self._tracks) == before:
                return False
            self.save()
            return True

    def snapshot(self) -> list[Track]:
        """深拷贝一份，给会改动它的流程用（抓取）。"""
        with self._lock:
            self._ensure()
            return copy.deepcopy(self._tracks)

    def replace(self, track: Track) -> None:
        with self._lock:
            self._ensure()
            for index, existing in enumerate(self._tracks):
                if existing.id == track.id:
                    self._tracks[index] = track
                    break
            else:
                self._tracks.append(track)
            self.save()


# ------------------------------------------------------------------ 抓取
#
# 同一时刻只让一个抓取在跑：调度器和「立即抓取」按钮可能撞在一起，
# 撞在一起时两个请求会各抓一遍、各写一遍 seen —— 结果是重复条目和丢更新。
_RUN_LOCK = threading.Lock()


class Busy(RuntimeError):
    """已经有一轮抓取在跑。"""


def _lock_or_fail() -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        raise Busy("上一轮抓取还没结束，稍等几秒再点")


def run_track(
    track: Track,
    *,
    fetch=None,
    now: datetime | None = None,
    limit: int = FETCH_LIMIT,
) -> int:
    """跑一轮，返回**新增**条数（首轮返回 0，见模块说明第 1 条）。

    抓取失败不抛异常：把 `last_status` 写成 degraded、`last_note` 带上原因，
    订阅本身保持不动 —— 一次限流不该让用户设好的订阅消失。
    """
    moment = now or utcnow()
    stamp = iso(moment)
    first_round = not track.last_run_at
    result = arxiv_tool.latest(track.categories, limit=limit, fetch=fetch)
    track.last_run_at = stamp

    if result.degraded:
        track.last_status = "degraded"
        track.last_note = result.note or "这一轮没有抓到（原因未知）"
        return 0

    terms = query_terms(track.query)
    fresh = [
        entry
        for entry in result.results
        if entry.arxiv_id not in track.seen and _matches(entry.title, entry.abstract, terms)
    ]
    for entry in fresh:
        track.seen[entry.arxiv_id] = stamp
    track.items = [TrackItem.from_entry(entry, stamp) for entry in fresh] + track.items
    track.items = track.items[:MAX_ITEMS]

    skipped = len(result.results) - len(fresh)
    if first_round:
        # 基线条目一开始就算"已看过"：它们不是这几天冒出来的，不该带新增徽标。
        # 不写这一行，新建订阅第一眼就是"新增 20 篇"，而那是它自己刚抓的。
        track.viewed_at = stamp
        track.last_status = "baseline"
        track.last_note = (
            f"首次抓取，作为基线记下 {len(fresh)} 条（本轮拿回 {len(result.results)} 条），"
            "不计入新增"
        )
        return 0
    track.last_status = "ok"
    if fresh:
        track.last_note = (
            f"新增 {len(fresh)} 条（本轮拿回 {len(result.results)} 条，"
            f"{skipped} 条已见过或与关键词无关）"
        )
    else:
        track.last_note = (
            f"没有新增（本轮拿回 {len(result.results)} 条，都已见过或与关键词无关）"
        )
    return len(fresh)


def run_one(store: TrackStore, track_id: str, *, fetch=None, now=None) -> Track:
    """跑指定订阅一轮并落盘。"""
    track = store.get(track_id)
    if track is None:
        raise KeyError(track_id)
    _lock_or_fail()
    try:
        run_track(track, fetch=fetch, now=now)
        store.replace(track)
        return store.get(track_id) or track
    finally:
        _RUN_LOCK.release()


def run_due(store: TrackStore, *, fetch=None, now=None, force: bool = False) -> list[Track]:
    """跑所有到期的订阅（`force` 时全跑）。返回跑过的那几个。"""
    moment = now or utcnow()
    due = [t for t in store.snapshot() if force or t.is_due(moment)]
    if not due:
        return []
    _lock_or_fail()
    try:
        ran: list[Track] = []
        for track in due:
            run_track(track, fetch=fetch, now=moment)
            store.replace(track)
            ran.append(track)
        return ran
    finally:
        _RUN_LOCK.release()


def ack(store: TrackStore, track_ids: list[str], *, now: datetime | None = None) -> int:
    """把"看过了"记下来：新增徽标就是拿 first_seen 跟它比。"""
    stamp = iso(now or utcnow())
    changed = 0
    for track_id in track_ids:
        track = store.get(track_id)
        if track is None:
            continue
        track.viewed_at = stamp
        store.replace(track)
        changed += 1
    return changed


# ------------------------------------------------------------------ 定时器

def scheduler_enabled() -> bool:
    flag = (os.environ.get("PAPER_TRACKING_SCHEDULER") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def tick_seconds() -> int:
    try:
        return max(10, min(int(os.environ.get("PAPER_TRACKING_TICK_SECONDS") or 60), 3600))
    except ValueError:
        return 60


_task: "asyncio.Task | None" = None


def start_scheduler(store: TrackStore) -> "asyncio.Task | None":
    """起一个后台循环：每 `tick_seconds()` 看一眼有没有到期的订阅。

    测试里把它关掉（`PAPER_TRACKING_SCHEDULER=0`）：单测不该因为
    有人建了一条订阅就去连 arXiv。
    """
    global _task
    if not scheduler_enabled() or _task is not None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    _task = loop.create_task(_loop(store))
    return _task


async def _loop(store: TrackStore) -> None:
    while True:
        try:
            # 抓取是阻塞的网络调用，丢到线程里跑，别把事件循环卡住
            await asyncio.to_thread(run_due, store)
        except Busy:
            pass
        except Exception:  # noqa: BLE001 —— 后台循环不许因为一次异常就整体退出
            pass
        await asyncio.sleep(tick_seconds())


def stop_scheduler() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
