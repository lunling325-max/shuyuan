"""arXiv 工具的测试（全程离线）。

重点不是"arXiv 今天返回了什么"，而是四条边界：
1. Atom 解析认得命名空间、多 entry 和版本号，并且把换行压成单行；
2. 网络失败只是降级（200 + degraded），不是 500；
3. 非法 id / 非 PDF 内容在入库前就被拦下，返回 400；
4. 同一份 PDF 重复 ingest 落到同一个内容哈希 id。

所有网络访问都换成假 fetch —— 测试里一个网络包都不许发。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.tools import arxiv

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"

ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: search_query=all:attention</title>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2023-08-02T00:00:00Z</published>
    <title>Attention Is
      All You Need</title>
    <summary>  The dominant sequence transduction models
      are based on complex recurrent or convolutional neural
      networks.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1810.04805v3</id>
    <published>2019-05-24T17:00:00Z</published>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce a new language representation model.</summary>
    <author><name>Jacob Devlin</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1810.04805v3" rel="related" type="application/pdf"/>
    <link href="http://arxiv.org/abs/1810.04805v3" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""

EMPTY_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
</feed>
"""


def _fake_fetch(body: bytes, calls: list[str] | None = None):
    def fetch(url: str) -> bytes:
        if calls is not None:
            calls.append(url)
        return body

    return fetch


@pytest.fixture(scope="module")
def client():
    # 只挂工具路由，不依赖 main.py 的挂载顺序；路径本身就是接口契约的一部分
    from fastapi import FastAPI
    from app.api.routers.tools import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def store(monkeypatch, tmp_path):
    """每个入库测试用独立注册表，别把 PDF 写进别的测试共用的资料库。"""
    from app.api import deps
    from app.api.registry import PaperRegistry

    registry = PaperRegistry(tmp_path / "papers")
    monkeypatch.setattr(deps, "registry", registry)
    return registry


# ------------------------------------------------------------ Atom 解析


def test_parse_atom_handles_multiple_entries_and_versions():
    calls: list[str] = []
    result = arxiv.search_arxiv("attention", limit=5, fetch=_fake_fetch(ATOM_FEED, calls))

    assert calls, "必须走注入的 fetch"
    assert "max_results=5" in calls[0]
    assert result.degraded is False
    assert result.note == ""

    # 版本号被抹掉；多条 entry 都要返回
    assert [entry.arxiv_id for entry in result.results] == ["1706.03762", "1810.04805"]

    first = result.results[0]
    assert first.title == "Attention Is All You Need", "标题里的换行要压平"
    assert first.abstract == (
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks."
    ), "摘要里的换行要压平"
    assert first.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert first.published == "2023-08-02"
    assert first.pdf_url == "http://arxiv.org/pdf/1706.03762v7"
    assert first.abs_url == "http://arxiv.org/abs/1706.03762v7"


def test_normalize_arxiv_id_strips_version_and_url_noise():
    assert arxiv.normalize_arxiv_id("1706.03762v7") == "1706.03762"
    assert arxiv.normalize_arxiv_id("1706.03762") == "1706.03762"
    assert arxiv.normalize_arxiv_id("arXiv:1706.03762v7") == "1706.03762"
    assert arxiv.normalize_arxiv_id("https://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert arxiv.normalize_arxiv_id("hep-th/9901001v2") == "hep-th/9901001"

    with pytest.raises(arxiv.InvalidArxivId):
        arxiv.normalize_arxiv_id("not-an-arxiv-id")


# ------------------------------------------------------------ 检索降级


def test_search_degrades_when_network_is_down(client, monkeypatch):
    def down(url: str) -> bytes:
        raise OSError("网络不可用")

    monkeypatch.setattr(arxiv, "default_fetch", down)

    r = client.get("/api/tools/arxiv/search", params={"q": "attention", "limit": 5})

    assert r.status_code == 200, "断网不能 500，页面还要能继续演示"
    body = r.json()
    assert body["query"] == "attention"
    assert body["results"] == []
    assert body["degraded"] is True
    assert "失败" in body["note"]


def test_search_limit_is_clamped_into_1_20(client, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(arxiv, "default_fetch", _fake_fetch(EMPTY_FEED, calls))

    r = client.get("/api/tools/arxiv/search", params={"q": "nlp", "limit": 99})
    assert r.status_code == 200
    assert "max_results=20" in calls[-1]

    r = client.get("/api/tools/arxiv/search", params={"q": "nlp", "limit": 0})
    assert r.status_code == 200
    assert "max_results=1" in calls[-1]

    r = client.get("/api/tools/arxiv/search", params={"q": "nlp", "limit": -7})
    assert r.status_code == 200
    assert "max_results=1" in calls[-1]
    assert r.json()["degraded"] is False


# ------------------------------------------------------------ 入库


def test_invalid_arxiv_id_returns_400_without_fetching(client, monkeypatch):
    calls: list[str] = []

    def must_not_be_called(url: str) -> bytes:
        calls.append(url)
        raise AssertionError("非法 id 不该发起下载")

    monkeypatch.setattr(arxiv, "default_fetch", must_not_be_called)

    r = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "2023-9999"})

    assert r.status_code == 400
    assert calls == []


def test_ingest_rejects_non_pdf_payload(client, store, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        arxiv,
        "default_fetch",
        _fake_fetch(b"<html><body>rate limited</body></html>", calls),
    )

    r = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762v7"})

    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]
    # 下载前已经归一化：v7 不该出现在 URL 里
    assert calls == ["https://arxiv.org/pdf/1706.03762"]


def test_ingest_with_no_extractable_text_returns_422(client, store, monkeypatch):
    # 文件头合法、PyMuPDF 也能打开，但一页都没有 —— 正是纯扫描件/坏 PDF 的形态
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type/Pages/Kids[]/Count 0>>\nendobj\n"
        b"trailer\n<</Root 1 0 R>>\n%%EOF"
    )
    monkeypatch.setattr(arxiv, "default_fetch", _fake_fetch(empty_pdf))

    r = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762"})

    assert r.status_code == 422


def test_ingest_download_failure_returns_502(client, store, monkeypatch):
    def down(url: str) -> bytes:
        raise OSError("连接超时")

    monkeypatch.setattr(arxiv, "default_fetch", down)

    r = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762"})

    assert r.status_code == 502


def test_repeated_ingest_lands_on_same_content_hash_id(client, store, monkeypatch):
    data = PDF_V7.read_bytes()
    calls: list[str] = []
    monkeypatch.setattr(arxiv, "default_fetch", _fake_fetch(data, calls))

    first = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762v7"})
    second = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762"})

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text

    first_body = first.json()
    second_body = second.json()
    assert first_body["paper"]["id"] == second_body["paper"]["id"], "同内容必须同 id"
    assert first_body["paper"]["chunks"] > 0
    assert first_body["fetched_bytes"] == len(data)
    assert second_body["fetched_bytes"] == len(data)
    assert len(store.all()) == 1, "重复入库不能在资料库里堆两个副本"
    assert calls == [
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/1706.03762",
    ]
