"""追踪端点的 HTTP 层：增删、抓取、"看过了"，以及**GET 不改状态**这一条。

为什么单拎出"GET 不改状态"：把"标记已读"塞进 GET 是最省事也最常见的写法，
省一次请求，但刷新两次页面徽标就没了 —— "新增"于是变成"上一次刷新之后新增"，
没人讲得清那是什么。所以这里逐个请求验：连读两次，`new_count` 不许变。

`arxiv_tool.latest` 在 HTTP 层被换成一个假函数：解析那半段由
`tests/test_tracking.py` 用真 Atom 字节守着，这一层只管"接口有没有把链路接对"。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.routers import tracking as tracking_routes
from app.tools.arxiv import ArxivEntry, SearchResult
from app.tracking import TrackStore


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ.setdefault("PAPER_STORE_DIR", str(tmp_path_factory.mktemp("papers_track")))
    from app.api.main import app

    return TestClient(app)


@pytest.fixture()
def store(tmp_path, monkeypatch) -> TrackStore:
    """每个用例一个干净的订阅表：这张表是落盘的，共用会把上一条用例的订阅带进来。"""
    fresh = TrackStore(tmp_path / "tracking.json")
    monkeypatch.setattr(tracking_routes, "STORE", fresh)
    return fresh


def _entry(arxiv_id: str, title: str = "") -> ArxivEntry:
    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title or f"Long context paper {arxiv_id}",
        authors=["A. Author"],
        published="2026-11-12",
        abstract="We study long context models.",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _ok(*entries) -> SearchResult:
    return SearchResult(query="cat:cs.CL", results=list(entries), degraded=False)


def _patch_latest(monkeypatch, result: SearchResult) -> list[str]:
    calls: list[str] = []

    def fake(categories=None, limit=20, *, fetch=None):
        calls.append(",".join(categories or []))
        return result

    monkeypatch.setattr("app.tracking.arxiv_tool.latest", fake)
    return calls


def test_create_then_list(client, store):
    r = client.post("/api/tracking", json={"query": "  long   context  ", "categories": "cs.CL, cs.LG"})
    assert r.status_code == 201, r.text
    track = r.json()
    assert track["query"] == "long context"          # 空白归一
    assert track["categories"] == ["cs.CL", "cs.LG"]
    assert track["interval_minutes"] == 360
    assert track["due"] is True                      # 没跑过 = 立刻到期
    assert track["items"] == [] and track["new_count"] == 0

    body = client.get("/api/tracking").json()
    assert [t["id"] for t in body["tracks"]] == [track["id"]]
    assert body["scheduler"]["tick_seconds"] >= 10
    assert body["note"] == ""


def test_empty_query_is_400(client, store):
    assert client.post("/api/tracking", json={"query": "   "}).status_code == 400


def test_default_categories_match_the_recommend_column(client, store):
    track = client.post("/api/tracking", json={"query": "attention"}).json()
    assert track["categories"] == ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]


def test_run_reports_a_baseline_then_only_real_additions(client, store, monkeypatch):
    track_id = client.post("/api/tracking", json={"query": "long context"}).json()["id"]
    _patch_latest(monkeypatch, _ok(_entry("2601.00001"), _entry("2601.00002")))

    first = client.post(f"/api/tracking/{track_id}/run").json()
    assert first["last_status"] == "baseline"
    assert first["new_count"] == 0
    assert first["seen_count"] == 2
    assert [item["arxiv_id"] for item in first["items"]] == ["2601.00001", "2601.00002"]
    assert first["last_note"].startswith("首次抓取")

    _patch_latest(
        monkeypatch, _ok(_entry("2601.00003"), _entry("2601.00002"), _entry("2601.00001"))
    )
    second = client.post(f"/api/tracking/{track_id}/run").json()
    assert second["last_status"] == "ok"
    assert second["new_count"] == 1
    assert second["seen_count"] == 3
    assert second["items"][0]["arxiv_id"] == "2601.00003"
    assert second["items"][0]["is_new"] is True
    assert second["items"][1]["is_new"] is False


def test_get_does_not_clear_the_new_badge(client, store, monkeypatch):
    track_id = client.post("/api/tracking", json={"query": "long context"}).json()["id"]
    _patch_latest(monkeypatch, _ok(_entry("2601.00001")))
    client.post(f"/api/tracking/{track_id}/run")          # 基线
    _patch_latest(monkeypatch, _ok(_entry("2601.00002"), _entry("2601.00001")))
    client.post(f"/api/tracking/{track_id}/run")

    for _ in range(2):                                     # 连读两次都一样
        track = client.get("/api/tracking").json()["tracks"][0]
        assert track["new_count"] == 1
    assert track["items"][0]["is_new"] is True


def test_ack_clears_the_badge(client, store, monkeypatch):
    track_id = client.post("/api/tracking", json={"query": "long context"}).json()["id"]
    _patch_latest(monkeypatch, _ok(_entry("2601.00001")))
    client.post(f"/api/tracking/{track_id}/run")
    _patch_latest(monkeypatch, _ok(_entry("2601.00002"), _entry("2601.00001")))
    client.post(f"/api/tracking/{track_id}/run")

    body = client.post("/api/tracking/ack", json={"ids": [track_id]}).json()
    assert body["tracks"][0]["new_count"] == 0
    assert all(not item["is_new"] for item in body["tracks"][0]["items"])
    # 条目本身一条没少 —— 清掉的是徽标，不是内容
    assert len(body["tracks"][0]["items"]) == 2


def test_degraded_fetch_is_a_200_with_a_reason_not_a_500(client, store, monkeypatch):
    track_id = client.post("/api/tracking", json={"query": "long context"}).json()["id"]
    _patch_latest(
        monkeypatch,
        SearchResult(query="cat:cs.CL", results=[], degraded=True, note="arXiv 追踪失败：429"),
    )
    r = client.post(f"/api/tracking/{track_id}/run")
    assert r.status_code == 200
    track = r.json()
    assert track["last_status"] == "degraded"
    assert "429" in track["last_note"]
    assert track["items"] == []
    # 失败也把这次跑记下来了，否则调度器会在限流期每分钟重试
    assert track["due"] is False


def test_delete_and_unknown_ids(client, store):
    track_id = client.post("/api/tracking", json={"query": "attention"}).json()["id"]
    assert client.delete(f"/api/tracking/{track_id}").status_code == 204
    assert client.delete(f"/api/tracking/{track_id}").status_code == 404
    assert client.post("/api/tracking/nope/run").status_code == 404
    assert client.post("/api/tracking/run").json()["tracks"] == []


def test_items_are_capped_but_the_total_is_reported(client, store, monkeypatch):
    track_id = client.post("/api/tracking", json={"query": "long context"}).json()["id"]
    _patch_latest(monkeypatch, _ok(*[_entry(f"2601.{i:05d}") for i in range(35)]))
    track = client.post(f"/api/tracking/{track_id}/run").json()
    assert len(track["items"]) == tracking_routes.RETURN_ITEMS
    assert track["items_total"] == 35
    assert track["seen_count"] == 35
