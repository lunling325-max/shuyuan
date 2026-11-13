"""arXiv 定期追踪：订阅表、已见集合、抓取语义（全程离线）。

这里守的不是"能不能抓到论文"（那是 arXiv 的事），而是三件关于**记忆**的事：

1. **首轮是基线，不是"新增 20 篇"。** 刚建的订阅第一轮拿回来的是"现在这一批"，
   把它算成新增，等于每次新建订阅都先撒一次谎；
2. **第二轮只报没见过的那几条。** 同一批论文再抓一次，"新增 0"必须是真的 0 ——
   这正是"追踪"和"推荐"的区别；
3. **抓取失败也要记下这次跑过。** 不然调度器会在限流期间每分钟重试一次，
   自己给自己加码。

另外守两条与"活得久"有关的：订阅表能读回来（重启不重来）、读坏了不静默清空。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app import tracking
from app.tracking import (
    Busy,
    Track,
    TrackItem,
    TrackStore,
    ack,
    query_terms,
    run_due,
    run_one,
    run_track,
)

NOW = datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)


def _entry(arxiv_id: str, title: str = "", abstract: str = "", published: str = "2026-11-12"):
    """默认摘要里带 `long context`：大多数用例测的是"见过没"而不是"关键词匹配"，
    让默认条目能被默认订阅命中，免得每条用例都得先想一遍关键词。"""
    from app.tools.arxiv import ArxivEntry

    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title or f"Paper {arxiv_id}",
        authors=["A. Author"],
        published=published,
        abstract=abstract or "We study long context models in this paper.",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
{entries}
</feed>
"""


def _feed(*entries) -> bytes:
    """造一段 arXiv 会回的 Atom。`fetch` 交出去的是**字节**，不是解析后的结果 ——
    绕过解析就等于把"XML 能不能读"这一段一起测没了。"""
    from xml.sax.saxutils import escape

    body = "".join(
        "<entry>"
        f"<id>http://arxiv.org/abs/{e.arxiv_id}v1</id>"
        f"<published>{e.published}T00:00:00Z</published>"
        f"<title>{escape(e.title)}</title>"
        f"<summary>{escape(e.abstract)}</summary>"
        + "".join(f"<author><name>{escape(a)}</name></author>" for a in e.authors)
        + f'<link title="pdf" href="{e.pdf_url}" rel="related" type="application/pdf"/>'
        f'<link href="{e.abs_url}" rel="alternate" type="text/html"/>'
        "</entry>"
        for e in entries
    )
    return ATOM.format(entries=body).encode("utf-8")


def _boom(url: str) -> bytes:
    """arXiv 限流时 `fetch` 是抛异常的（HTTPError 之类）。"""
    raise RuntimeError("429")


def _track(**kwargs) -> Track:
    base = dict(
        id="t1",
        query="long context",
        categories=["cs.CL"],
        interval_minutes=360,
        created_at=tracking.iso(NOW - timedelta(hours=1)),
    )
    base.update(kwargs)
    return Track(**base)


def _store(tmp_path) -> TrackStore:
    return TrackStore(tmp_path / "tracking.json")


# ------------------------------------------------------------------ 抓取语义

def test_first_round_is_a_baseline_not_20_new_papers():
    track = _track()
    added = run_track(
        track,
        fetch=lambda url: _feed(_entry("2601.00001"), _entry("2601.00002")),
        now=NOW,
    )
    assert added == 0
    assert track.last_status == "baseline"
    assert "基线" in track.last_note and "不计入新增" in track.last_note
    # 条目照样列出来，`seen` 也照样记下 —— 基线只是"不算新增"，不是"不记"
    assert [item.arxiv_id for item in track.items] == ["2601.00001", "2601.00002"]
    assert set(track.seen) == {"2601.00001", "2601.00002"}


def test_second_round_reports_only_papers_it_has_never_seen():
    track = _track()
    run_track(track, fetch=lambda url: _feed(_entry("2601.00001"), _entry("2601.00002")), now=NOW)
    later = NOW + timedelta(hours=6)
    added = run_track(
        track,
        fetch=lambda url: _feed(_entry("2601.00003"), _entry("2601.00002"), _entry("2601.00001")),
        now=later,
    )
    assert added == 1
    assert track.last_status == "ok"
    assert track.items[0].arxiv_id == "2601.00003"      # 新的排最前
    assert len(track.items) == 3                        # 旧的不重复入列


def test_same_batch_twice_reports_zero_new():
    """"追踪"的自证：什么都不变的时候，新增必须是 0，而不是把 20 条再报一遍。"""
    track = _track()
    feed = _feed(_entry("2601.00001"), _entry("2601.00002"))
    run_track(track, fetch=lambda url: feed, now=NOW)
    added = run_track(track, fetch=lambda url: feed, now=NOW + timedelta(hours=6))
    assert added == 0
    assert "没有新增" in track.last_note


def test_keyword_filter_needs_two_terms_and_drops_the_rest():
    track = _track(query="retrieval augmented generation")
    run_track(
        track,
        fetch=lambda url: _feed(
            _entry("2601.00001", title="Retrieval Augmented Generation for Long Documents"),
            # 只命中 generation 一个词：常见词凑不出"在讲这件事"，不该进来
            _entry("2601.00002", title="Pixel-Space Video Generation", abstract="We generate pixels."),
            _entry("2601.00003", title="A Study of Protein Folding"),
        ),
        now=NOW,
    )
    assert [item.arxiv_id for item in track.items] == ["2601.00001"]
    # 没命中的**不该**进 seen —— 否则用户放宽关键词后它们永远不算新增
    assert set(track.seen) == {"2601.00001"}


def test_single_keyword_is_enough_on_its_own():
    track = _track(query="protein")
    run_track(
        track,
        fetch=lambda url: _feed(_entry("2601.00001", title="A Study of Protein Folding")),
        now=NOW,
    )
    assert [item.arxiv_id for item in track.items] == ["2601.00001"]


def test_failed_fetch_is_recorded_and_does_not_hurt_what_we_already_have():
    track = _track()
    run_track(track, fetch=lambda url: _feed(_entry("2601.00001")), now=NOW)
    seen_before = dict(track.seen)
    later = NOW + timedelta(hours=6)
    added = run_track(track, fetch=_boom, now=later)
    assert added == 0
    assert track.last_status == "degraded"
    assert "429" in track.last_note
    assert track.seen == seen_before
    # 关键：`last_run_at` 也要往前走，否则调度器每分钟重试一次
    assert track.last_run_at == tracking.iso(later)
    assert not track.is_due(later + timedelta(minutes=1))


def test_due_uses_the_interval_counted_from_the_last_run():
    track = _track(interval_minutes=360)
    assert track.is_due(NOW)                       # 从没跑过 = 到期
    track.last_run_at = tracking.iso(NOW)
    assert not track.is_due(NOW + timedelta(minutes=359))
    assert track.is_due(NOW + timedelta(minutes=361))


def test_run_due_only_runs_what_is_due_and_force_runs_everything(tmp_path):
    store = _store(tmp_path)
    fresh = store.add(query="long context", interval_minutes=60)
    idle = store.add(query="protein", interval_minutes=60)
    idle_track = store.get(idle.id)
    idle_track.last_run_at = tracking.iso(NOW)
    store.replace(idle_track)

    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return _feed(_entry("2601.00001"))

    ran = run_due(store, fetch=fetch, now=NOW + timedelta(minutes=1))
    assert [t.id for t in ran] == [fresh.id]
    assert len(calls) == 1

    ran = run_due(store, fetch=fetch, now=NOW + timedelta(minutes=1), force=True)
    assert {t.id for t in ran} == {fresh.id, idle.id}


def test_two_runs_at_the_same_time_raises_busy(tmp_path):
    store = _store(tmp_path)
    track = store.add(query="long context")
    tracking._RUN_LOCK.acquire()
    try:
        with pytest.raises(Busy):
            run_one(store, track.id, fetch=lambda url: _feed(_entry("2601.00001")), now=NOW)
    finally:
        tracking._RUN_LOCK.release()


def test_query_terms_drop_stopwords_and_short_words():
    assert query_terms("  Long   Context, the model; of GPT  ") == ["long", "context", "model", "gpt"]
    assert query_terms("") == []


# ------------------------------------------------------------------ 存储

def test_store_round_trips_everything_that_makes_tracking_useful(tmp_path):
    store = _store(tmp_path)
    track = store.add(query="long context", categories=["cs.CL", "cs.LG"], interval_minutes=120)
    run_track(track, fetch=lambda url: _feed(_entry("2601.00001")), now=NOW)
    store.replace(track)

    again = TrackStore(tmp_path / "tracking.json")
    restored = again.get(track.id)
    assert restored is not None
    assert restored.query == "long context"
    assert restored.categories == ["cs.CL", "cs.LG"]
    assert restored.seen == {"2601.00001": tracking.iso(NOW)}
    assert [item.arxiv_id for item in restored.items] == ["2601.00001"]
    assert again.load_error == ""


def test_corrupt_store_is_kept_as_bad_file_instead_of_being_wiped(tmp_path):
    path = tmp_path / "tracking.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")
    store = TrackStore(path)
    assert store.all() == []
    assert "读取失败" in store.load_error
    bad = tmp_path / "tracking.json.bad"
    assert bad.exists() and "这不是 JSON" in bad.read_text(encoding="utf-8")
    # 读坏了之后仍然能继续用：新加一条、落盘、下次读回来
    store.add(query="attention")
    assert len(TrackStore(path).all()) == 1


def test_empty_query_is_rejected_and_interval_is_clamped(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.add(query="   ")
    short = store.add(query="attention", interval_minutes=1)
    assert short.interval_minutes == tracking.MIN_INTERVAL_MINUTES
    long = store.add(query="attention", interval_minutes=10_000_000)
    assert long.interval_minutes == tracking.MAX_INTERVAL_MINUTES
    # 分类不填就给默认那四个，和「每日推荐」那一栏保持一致
    assert short.categories == ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]


def test_ack_clears_the_new_badge_and_get_alone_does_not(tmp_path):
    store = _store(tmp_path)
    track = store.add(query="long context")
    # 首轮基线（条目算"已看过"），第二轮才有一条真新增
    run_track(track, fetch=lambda url: _feed(_entry("2601.00001")), now=NOW)
    store.replace(track)
    run_track(
        track,
        fetch=lambda url: _feed(_entry("2601.00002"), _entry("2601.00001")),
        now=NOW + timedelta(hours=6),
    )
    store.replace(track)
    assert [item.arxiv_id for item in store.get(track.id).new_items()] == ["2601.00002"]

    # 只是读一遍，不该把"新"清掉
    assert [item.arxiv_id for item in store.get(track.id).new_items()] == ["2601.00002"]

    ack(store, [track.id], now=NOW + timedelta(hours=7))
    assert store.get(track.id).new_items() == []
    # 清掉的是徽标，不是内容
    assert len(store.get(track.id).items) == 2


def test_remove_reports_whether_anything_went_away(tmp_path):
    store = _store(tmp_path)
    track = store.add(query="attention")
    assert store.remove(track.id) is True
    assert store.remove(track.id) is False
    assert TrackStore(tmp_path / "tracking.json").all() == []


def test_item_without_arxiv_id_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "tracking.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "tracks": [
                    {
                        "id": "t1",
                        "query": "attention",
                        "categories": ["cs.CL"],
                        "interval_minutes": 60,
                        "created_at": tracking.iso(NOW),
                        "items": [{"title": "没有 id 的条目"}, {"arxiv_id": "2601.00009", "title": "好的"}],
                        "seen": {"2601.00009": tracking.iso(NOW)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    track = TrackStore(path).get("t1")
    assert [item.arxiv_id for item in track.items] == ["2601.00009"]


# ------------------------------------------------------------------ 定时器

def test_scheduler_can_be_switched_off_by_env(monkeypatch):
    monkeypatch.setenv("PAPER_TRACKING_SCHEDULER", "0")
    assert tracking.scheduler_enabled() is False
    monkeypatch.setenv("PAPER_TRACKING_SCHEDULER", "1")
    assert tracking.scheduler_enabled() is True


def test_tick_seconds_is_clamped(monkeypatch):
    monkeypatch.setenv("PAPER_TRACKING_TICK_SECONDS", "1")
    assert tracking.tick_seconds() == 10
    monkeypatch.setenv("PAPER_TRACKING_TICK_SECONDS", "有很多秒")
    assert tracking.tick_seconds() == 60


def test_track_item_keeps_the_fields_the_feed_needs():
    item = TrackItem.from_entry(_entry("2601.00001", title="T"), first_seen=tracking.iso(NOW))
    assert item.to_dict()["first_seen"] == tracking.iso(NOW)
    assert Track.from_dict(_track().to_dict()).query == "long context"
