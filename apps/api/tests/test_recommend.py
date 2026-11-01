"""追踪与推荐的测试（全程离线）。

守四件事：
1. 相关性算的是"和库里的哪篇像"，而且**解释得出来**（命中哪几篇、重合多少）；
2. 开源代码地址能从摘要和 `<arxiv:comment>` 里找出来，坏链接不算；
3. 推荐理由里的每一个数字都来自真实计算，没有一句是编的；
4. 点赞/踩能投、能取消，按用户分开存，写盘不会留下半截文件。
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import recommender
from app.recommender import (
    FeedbackStore,
    LibraryPaper,
    extract_code_url,
    keywords,
    rank,
    similarity,
)
from app.tools.arxiv import ArxivEntry

TODAY = date.today().isoformat()


def _entry(arxiv_id="2601.00001", title="", abstract="", comment="", published=None):
    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title,
        authors=["A. Author"],
        published=published if published is not None else TODAY,
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        comment=comment,
    )


LIBRARY = [
    LibraryPaper(
        id="lib1",
        title="Attention Is All You Need",
        tokens=keywords(
            "attention transformer sequence transduction recurrence self attention "
            "encoder decoder machine translation bleu"
        ),
    ),
    LibraryPaper(
        id="lib2",
        title="BERT Pre-training",
        tokens=keywords("bert pre-training bidirectional encoder representations masked language model"),
    ),
]


# ---------------------------------------------------------------- 词面相关性


def test_keywords_drop_stopwords_and_short_tokens():
    tokens = keywords("We propose a new method based on attention")
    assert "attention" in tokens
    assert "propose" not in tokens and "the" not in tokens
    assert all(len(t) > 2 for t in tokens)


def test_similarity_is_cosine_over_shared_words():
    a = keywords("self attention transformer encoder")
    b = keywords("self attention transformer decoder")
    assert 0 < similarity(a, b) < 1
    assert similarity(a, keywords("protein folding cryo em")) == 0.0
    assert similarity(frozenset(), a) == 0.0


# ---------------------------------------------------------------- 开源代码


def test_code_url_found_in_abstract_and_comment():
    assert extract_code_url("see https://github.com/google-research/t5") == \
        "https://github.com/google-research/t5"
    # 作者常把代码只写在 comment 里
    assert extract_code_url("", "code at github.com/karpathy/minGPT.") == \
        "https://github.com/karpathy/minGPT"
    assert extract_code_url("", "") == ""
    # 不是代码仓库的链接不能冒充
    assert extract_code_url("blog: https://example.com/post") == ""


# ---------------------------------------------------------------- 打分与理由


def test_rank_prefers_relevant_and_open_source():
    entries = [
        _entry("2601.00001", "Protein Folding with Cryo-EM", "cryo em protein structure"),
        _entry("2601.00002", "Efficient Self Attention for Long Documents",
               "self attention transformer encoder long documents"),
        _entry("2601.00003", "Self Attention Revisited",
               "self attention transformer encoder efficiency",
               comment="Code: github.com/example/attention"),
    ]
    ranked = rank(entries, LIBRARY, today=TODAY, limit=3)

    assert [r.entry.arxiv_id for r in ranked][:2] == ["2601.00003", "2601.00002"]
    assert ranked[-1].entry.arxiv_id == "2601.00001"

    top = ranked[0]
    assert top.matches and top.matches[0].paper_id == "lib1"
    assert "词面重合" in top.reason
    assert "github.com/example/attention" in top.reason
    assert "今天刚提交" in top.reason
    assert top.code_url.startswith("https://github.com/")


def test_relevance_matches_are_ordered_and_capped():
    entries = [_entry(title="BERT attention transformer encoder", abstract="bert attention")]
    ranked = rank(entries, LIBRARY, today=TODAY, limit=5)
    sims = [m.similarity for m in ranked[0].matches]
    assert sims == sorted(sims, reverse=True)
    assert len(sims) <= 3


def test_empty_library_still_recommends_and_says_why():
    entries = [_entry(title="A New Attention Variant", abstract="attention")]
    ranked = rank(entries, [], today=TODAY, limit=5)
    assert ranked and "文献库还是空的" in ranked[0].reason


def test_unknown_date_does_not_invent_freshness():
    ranked = rank([_entry(published="")], LIBRARY, today=TODAY, limit=1)
    assert "提交" not in ranked[0].reason


# ---------------------------------------------------------------- 点赞 / 踩


def test_feedback_store_counts_per_user(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")
    assert store.counts("1706.03762") == (0, 0, "")

    assert store.cast("1706.03762", "alice", "up") == (1, 0, "up")
    assert store.cast("1706.03762", "bob", "up") == (2, 0, "up")
    assert store.cast("1706.03762", "bob", "down") == (1, 1, "down")   # 改票
    assert store.cast("1706.03762", "bob", "") == (1, 0, "")           # 取消

    # 重新打开：落盘了
    again = FeedbackStore(tmp_path / "feedback.json")
    assert again.counts("1706.03762", "alice") == (1, 0, "up")
    assert not (tmp_path / "feedback.json.tmp").exists()


def test_feedback_store_rejects_unknown_vote(tmp_path):
    with pytest.raises(ValueError):
        FeedbackStore(tmp_path / "f.json").cast("1706.03762", "me", "meh")


# ---------------------------------------------------------------- 端点


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from app.api import deps
    from app.api.registry import PaperRegistry
    from app.api.routers import recommend as recommend_routes

    monkeypatch.setattr(deps, "registry", PaperRegistry(tmp_path / "papers"))
    monkeypatch.setattr(
        recommend_routes, "_feedback", FeedbackStore(tmp_path / "feedback.json")
    )
    recommend_routes._CACHE.clear()

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(recommend_routes.router)
    return TestClient(app)


def test_recommend_endpoint_ranks_and_explains(client, monkeypatch):
    from app.api.routers import recommend as recommend_routes
    from app.tools import arxiv as arxiv_tool

    feed = arxiv_tool.SearchResult(
        query="cat:cs.CL",
        results=[
            _entry("2601.00009", "Protein Folding", "cryo em structure prediction"),
            _entry("2601.00010", "Self Attention for Long Documents",
                   "self attention transformer encoder long documents",
                   comment="code: github.com/example/longformer"),
        ],
        degraded=False,
    )
    monkeypatch.setattr(recommend_routes.arxiv_tool, "latest", lambda *a, **k: feed)

    body = client.get("/api/recommend", params={"categories": "cs.CL"}).json()
    assert body["degraded"] is False
    assert len(body["items"]) == 2
    assert body["items"][0]["arxiv_id"] == "2601.00010"
    assert body["items"][0]["code_url"].endswith("/example/longformer")
    assert body["items"][0]["reason"]
    assert body["library_size"] == 0

    # 第二次命中缓存，不再出网
    second = client.get("/api/recommend", params={"categories": "cs.CL"}).json()
    assert second["cached"] is True


def test_recommend_endpoint_degrades_without_network(client, monkeypatch):
    from app.api.routers import recommend as recommend_routes
    from app.tools import arxiv as arxiv_tool

    feed = arxiv_tool.SearchResult(
        query="cat:cs.CL", results=[], degraded=True, note="arXiv 追踪失败：超时"
    )
    monkeypatch.setattr(recommend_routes.arxiv_tool, "latest", lambda *a, **k: feed)

    response = client.get("/api/recommend")
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True and "追踪失败" in body["note"]
    assert body["items"] == []


def test_vote_endpoint_round_trip(client):
    first = client.post("/api/recommend/vote",
                        json={"arxiv_id": "1706.03762v7", "vote": "up"}).json()
    assert first == {"arxiv_id": "1706.03762", "votes_up": 1,
                     "votes_down": 0, "my_vote": "up"}
    cancel = client.post("/api/recommend/vote",
                         json={"arxiv_id": "1706.03762", "vote": ""}).json()
    assert cancel["votes_up"] == 0 and cancel["my_vote"] == ""


def test_vote_endpoint_rejects_bad_input(client):
    assert client.post("/api/recommend/vote",
                       json={"arxiv_id": "not-an-id", "vote": "up"}).status_code == 400
    assert client.post("/api/recommend/vote",
                       json={"arxiv_id": "1706.03762", "vote": "meh"}).status_code == 400


def test_failed_refresh_falls_back_to_last_good_batch(client, monkeypatch):
    from app.api.routers import recommend as recommend_routes
    from app.tools import arxiv as arxiv_tool

    good = arxiv_tool.SearchResult(
        query="cat:cs.CL", degraded=False,
        results=[_entry("2601.00011", "Self Attention", "self attention transformer")],
    )
    bad = arxiv_tool.SearchResult(
        query="cat:cs.CL", results=[], degraded=True, note="arXiv 追踪失败：限流（HTTP 429）"
    )
    monkeypatch.setattr(recommend_routes.arxiv_tool, "latest", lambda *a, **k: good)
    assert len(client.get("/api/recommend").json()["items"]) == 1

    # 强制刷新时 arXiv 挂了：不该整页空白，要用上一次那批并说明是旧的
    monkeypatch.setattr(recommend_routes.arxiv_tool, "latest", lambda *a, **k: bad)
    body = client.get("/api/recommend", params={"fresh": True}).json()
    assert body["degraded"] is True
    assert len(body["items"]) == 1
    assert "分钟前刷到" in body["note"]


def test_tracking_feed_sorts_by_date_and_does_not_retry():
    from app.tools import arxiv as arxiv_tool

    calls: list[str] = []

    def failing(url: str) -> bytes:
        calls.append(url)
        raise OSError("The read operation timed out")

    result = arxiv_tool.latest(["cs.CL", "cs.LG"], limit=10, fetch=failing)
    assert result.degraded is True
    assert "追踪失败" in result.note
    # 被限流时重试只是让页面多挂十几秒，追踪只打一次
    assert len(calls) == 1
    assert "sortBy=submittedDate" in calls[0] and "sortOrder=descending" in calls[0]
    assert "cat%3Acs.CL%20OR%20cat%3Acs.LG" in calls[0]


def test_rank_limit_cuts_after_sorting():
    entries = [_entry(f"2601.000{i:02d}", f"Attention paper {i}", "attention transformer")
               for i in range(10)]
    ranked = rank(entries, LIBRARY, today=TODAY, limit=4)
    assert len(ranked) == 4
    assert ranked == sorted(ranked, key=lambda r: r.score, reverse=True)
    assert recommender.W_CODE + recommender.W_FRESH + recommender.W_RELEVANCE == pytest.approx(1.0)
