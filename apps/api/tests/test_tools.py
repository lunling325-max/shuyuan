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


# ------------------------------------------------------------ 下载重试


def _pdf_bytes() -> bytes:
    return PDF_V7.read_bytes()


def test_download_retries_after_a_read_timeout(monkeypatch):
    """一次读超时不该让用户看到失败：arXiv 这条链路实测会间歇性卡住。"""
    calls: list[str] = []
    data = _pdf_bytes()

    def flaky(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return data

    monkeypatch.setattr(arxiv, "default_fetch", flaky)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    assert arxiv.download_pdf("1706.03762") == data
    assert len(calls) == 2, "第二次就该成功"


def test_download_gives_up_after_three_attempts_and_says_so(monkeypatch):
    calls: list[str] = []

    def down(url: str) -> bytes:
        calls.append(url)
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(arxiv, "default_fetch", down)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    with pytest.raises(arxiv.FetchError) as excinfo:
        arxiv.download_pdf("1706.03762")

    assert len(calls) == arxiv.DOWNLOAD_ATTEMPTS
    message = str(excinfo.value)
    assert "3 次" in message, "错误里要写明试了几次"
    assert "拖进文献库" in message, "要给一条不依赖 arXiv 的绕行办法"


def test_non_pdf_content_is_not_retried(monkeypatch):
    """内容不对重试也没用，别让用户白等三倍时间。"""
    calls: list[str] = []
    monkeypatch.setattr(
        arxiv, "default_fetch", _fake_fetch(b"<html>rate limited</html>", calls)
    )

    with pytest.raises(arxiv.NotPdfError):
        arxiv.download_pdf("1706.03762")

    assert len(calls) == 1


def test_ingest_survives_one_timeout(client, store, monkeypatch):
    """端到端：第一次超时、第二次成功，用户拿到的是 201 而不是 502。"""
    calls: list[str] = []
    data = _pdf_bytes()

    def flaky(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return data

    monkeypatch.setattr(arxiv, "default_fetch", flaky)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    r = client.post("/api/tools/arxiv/ingest", json={"arxiv_id": "1706.03762"})

    assert r.status_code == 201, r.text
    assert r.json()["fetched_bytes"] == len(data)


def test_truncated_pdf_is_retried_then_succeeds(monkeypatch):
    """断流时 read() 不抛异常，只回半截 —— 必须靠 %%EOF 认出来并重下。

    这条是实测里真踩到的：拿到 815KB/2.1MB 的半截 PDF，
    PyMuPDF 报 "Failed to open file … as type pdf"，而下载那一步"成功"了。
    """
    calls: list[str] = []
    full = _pdf_bytes()
    truncated = full[: len(full) // 3]

    def flaky(url: str) -> bytes:
        calls.append(url)
        return truncated if len(calls) == 1 else full

    monkeypatch.setattr(arxiv, "default_fetch", flaky)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    assert arxiv.download_pdf("1706.03762") == full
    assert len(calls) == 2, "半截内容要触发重下，且第二次就该拿到完整文件"


def test_truncated_pdf_fails_after_all_attempts(monkeypatch):
    calls: list[str] = []
    truncated = _pdf_bytes()[: 100_000]

    def always_short(url: str) -> bytes:
        calls.append(url)
        return truncated

    monkeypatch.setattr(arxiv, "default_fetch", always_short)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    with pytest.raises(arxiv.FetchError) as excinfo:
        arxiv.download_pdf("1706.03762")

    assert len(calls) == arxiv.DOWNLOAD_ATTEMPTS
    assert "EOF" in str(excinfo.value), "要说清是半截文件，不是别的网络错误"


def test_complete_check_ignores_non_pdf_payloads():
    # HTML 限流页没有 %PDF 头，不该被当成"没下完"
    assert arxiv._is_complete_pdf(b"<html>rate limited</html>") is True
    assert arxiv._is_complete_pdf(_pdf_bytes()) is True
    assert arxiv._is_complete_pdf(_pdf_bytes()[:1000]) is False


# ------------------------------------------------------------ 检索式构造


def test_multi_word_query_is_sent_as_a_phrase(monkeypatch):
    """arXiv 的 all: 只绑定一个词，整句必须加引号，否则那篇经典根本排不进来。"""
    calls: list[str] = []
    monkeypatch.setattr(arxiv, "default_fetch", _fake_fetch(ATOM_FEED, calls))

    arxiv.search_arxiv("attention is all you need", limit=5)

    assert len(calls) == 1, "整句命中了就不用再按词搜"
    assert "search_query=all%3A%22attention%20is%20all%20you%20need%22" in calls[0]


def test_phrase_miss_falls_back_to_loose_words(monkeypatch):
    """"我只有一个主题、没有标题"时，整句搜不到要退回去按词搜。"""
    calls: list[str] = []
    feeds = [EMPTY_FEED, ATOM_FEED]

    def fetch(url: str) -> bytes:
        calls.append(url)
        return feeds[len(calls) - 1]

    monkeypatch.setattr(arxiv, "default_fetch", fetch)

    result = arxiv.search_arxiv("sparse attention long document retrieval")

    assert len(calls) == 2
    assert "%22" in calls[0] and "%22" not in calls[1], "第一次整句带引号，第二次按词搜"
    assert result.results, "退回去的那次有结果就要用"
    assert "精确匹配" in result.note, "要说清这批不是精确匹配"


def test_search_retries_once_after_handshake_timeout(monkeypatch):
    """握手卡死是实测过的常态，隔一下再发就能通，别让用户看到"不可达"。"""
    calls: list[str] = []

    def flaky(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("The handshake operation timed out")
        return ATOM_FEED

    monkeypatch.setattr(arxiv, "default_fetch", flaky)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _s: None)

    result = arxiv.search_arxiv("attention", limit=5)

    assert result.degraded is False
    assert len(calls) == 2
    assert result.results


def test_quotes_in_user_query_do_not_break_the_syntax():
    candidates = arxiv._search_candidates('a "quoted" (weird) query')
    assert candidates[0] == 'all:"a quoted (weird) query"'
