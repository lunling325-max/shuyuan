"""多来源检索的测试（全程离线）。

这组测试守的不是"各站今天返回了什么"，而是四件事：

1. **一个站挂了不影响其余站** —— 每站各自降级，最终仍是 200；
2. **同一篇论文不会出现两遍** —— 各站的身份标识不同，去重要认全；
3. **能不能入库要说真话** —— 只有解得出 arXiv id 的条目才给 can_ingest；
4. **认得出 arXiv 的各种写法** —— pdf 链接、abs 链接、10.48550/arxiv.* 的 DOI。

所有请求都走注入的假 fetch，一个网络包都不发。

另有第五件事（E-040 起的「引用含金量」）：**被引次数不能凭空出现，也不能凭空消失**。
各站这个字段的名字各不相同，取不到时必须是"没有数"而不是"0 次"。
"""

from __future__ import annotations

import json
import time

import pytest

from app.tools import sources
from app.tools.sources import (
    Hit,
    _arxiv_id_from,
    _cited,
    _dedupe_keys,
    _merge,
    search_sources,
)

ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2023-08-02T00:00:00Z</published>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on RNNs.</summary>
    <author><name>Ashish Vaswani</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""

EMPTY_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""

OPENALEX_SAME_PAPER = {
    "results": [
        {
            "id": "https://openalex.org/W2963403868",
            "display_name": "Attention Is All You Need",
            "publication_year": 2017,
            "doi": "https://doi.org/10.48550/arxiv.1706.03762",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Noam Shazeer"}},
            ],
            "primary_location": {
                "source": {"display_name": "arXiv"},
                "landing_page_url": "https://arxiv.org/abs/1706.03762",
            },
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762"},
        }
    ]
}

OPENALEX_MIXED = {
    "results": [
        {
            "id": "https://openalex.org/W2963403868",
            "display_name": "Attention Is All You Need",
            "publication_year": 2017,
            "doi": "https://doi.org/10.48550/arxiv.1706.03762",
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
            "primary_location": {"source": {"display_name": "arXiv"}},
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762"},
        },
        {
            "id": "https://openalex.org/W2043403838",
            "display_name": "Deep Residual Learning for Image Recognition",
            "publication_year": 2016,
            "doi": "https://doi.org/10.1109/cvpr.2016.90",
            "authorships": [{"author": {"display_name": "Kaiming He"}}],
            "primary_location": {
                "source": {"display_name": "CVPR"},
                "landing_page_url": "https://doi.org/10.1109/cvpr.2016.90",
            },
            "best_oa_location": {"pdf_url": "https://openaccess.thecvf.com/resnet.pdf"},
        },
    ]
}

CROSSREF_SAME_TITLE = {
    "message": {
        "items": [
            {
                "DOI": "10.5555/3295222.3295349",
                "title": ["Attention is All you Need"],
                "author": [{"given": "Ashish", "family": "Vaswani"}],
                "issued": {"date-parts": [[2017]]},
                "container-title": ["NeurIPS"],
                "link": [],
            }
        ]
    }
}


ZENODO_RECORDS = {
    "hits": {
        "hits": [
            {
                "id": 14571411,
                "doi": "10.5281/zenodo.14571411",
                "links": {"self_html": "https://zenodo.org/records/14571411"},
                "metadata": {
                    "title": "Generative AI Unveiled: Mechanisms and Biases",
                    "creators": [
                        {"name": "Thotapally, Saiprapul"},
                        {"name": "Rao, Meera"},
                    ],
                    "publication_date": "2024-12-29",
                    "resource_type": {"title": "Journal article"},
                    "description": "<p>This article explores <b>biases</b> in generative models.</p>",
                },
                "files": [
                    {
                        "key": "Generative AI Unveiled.pdf",
                        "links": {
                            "self": "https://zenodo.org/api/records/14571411/files/paper.pdf/content"
                        },
                    }
                ],
            }
        ]
    }
}


# 三站对"被引次数"各有各的叫法：OpenAlex 是 cited_by_count，Crossref 是
# is-referenced-by-count，S2 是 citationCount。标题故意各不相同，免得被去重并成一条。
OPENALEX_CITED = {
    "results": [
        {
            "id": "https://openalex.org/W2963403868",
            "display_name": "Attention Is All You Need",
            "publication_year": 2017,
            "doi": "https://doi.org/10.48550/arxiv.1706.03762",
            "cited_by_count": 136007,
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
            "primary_location": {
                "source": {"display_name": "arXiv"},
                "landing_page_url": "https://arxiv.org/abs/1706.03762",
            },
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762"},
        }
    ]
}

CROSSREF_CITED = {
    "message": {
        "items": [
            {
                "DOI": "10.1109/cvpr.2016.90",
                "title": ["Deep Residual Learning for Image Recognition"],
                "author": [{"given": "Kaiming", "family": "He"}],
                "issued": {"date-parts": [[2016]]},
                "container-title": ["CVPR"],
                "link": [],
                "is-referenced-by-count": 214800,
            }
        ]
    }
}

S2_CITED = {
    "data": [
        {
            "paperId": "abc123",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2019,
            "authors": [{"name": "Jacob Devlin"}],
            "externalIds": {"ArXiv": "1810.04805"},
            "openAccessPdf": {"url": "https://arxiv.org/pdf/1810.04805"},
            "venue": "NAACL",
            "citationCount": 99000,
        }
    ]
}


def _json(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _dispatcher(table: dict[str, bytes]):
    """按 URL 里的域名分派响应；表里没有的域名一律当断网。"""

    def fetch(url: str) -> bytes:
        for marker, body in table.items():
            if marker in url:
                return body
        raise OSError(f"没有为 {url} 准备响应")

    return fetch


# ------------------------------------------------------------------ 身份识别


def test_arxiv_id_is_found_in_every_common_form():
    assert _arxiv_id_from("https://arxiv.org/pdf/1706.03762v7") == "1706.03762"
    assert _arxiv_id_from("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert _arxiv_id_from("https://doi.org/10.48550/arXiv.1706.03762") == "1706.03762"
    assert _arxiv_id_from("10.1109/cvpr.2016.90") == ""
    assert _arxiv_id_from("") == ""


def test_dedupe_keys_cover_arxiv_doi_and_title():
    keys = _dedupe_keys(
        Hit(source="openalex", source_id="W2963403868",
            title="Attention Is All You Need!", arxiv_id="1706.03762")
    )
    assert "arxiv:1706.03762" in keys
    assert "title:attentionisallyouneed" in keys

    doi_hit = Hit(source="crossref", source_id="10.5555/3295222.3295349",
                  title="Attention is All you Need")
    # 只靠 DOI 撞不上 arXiv 那条，还要靠抹平后的标题 —— 这就是它存在的理由
    assert "doi:10.5555/3295222.3295349" in _dedupe_keys(doi_hit)
    assert "title:attentionisallyouneed" in _dedupe_keys(doi_hit)
    assert set(keys) & set(_dedupe_keys(doi_hit)) == {"title:attentionisallyouneed"}


# ------------------------------------------------------------------ 汇总


def test_same_paper_from_two_sources_appears_once():
    fetch = _dispatcher({
        "export.arxiv.org": ARXIV_FEED,
        "api.openalex.org": _json(OPENALEX_SAME_PAPER),
        "api.crossref.org": _json(CROSSREF_SAME_TITLE),
    })
    result = search_sources("attention is all you need", limit=10, fetch=fetch)

    assert len(result.hits) == 1
    hit = result.hits[0]
    # 保留先出现的来源（arXiv），其余站只是给它补字段
    assert hit.source == "arxiv"
    assert hit.arxiv_id == "1706.03762"

    reports = {r.name: r for r in result.sources}
    assert reports["arxiv"].count == 1
    assert reports["arxiv"].ok
    assert reports["crossref"].ok is False or reports["crossref"].count == 1


def test_one_source_down_does_not_take_the_others_with_it():
    fetch = _dispatcher({
        "export.arxiv.org": ARXIV_FEED,
        "api.crossref.org": _json(CROSSREF_SAME_TITLE),
        # 没有 openalex：它的请求会抛 OSError，模拟断网
    })
    result = search_sources("attention", limit=10, fetch=fetch)

    reports = {r.name: r for r in result.sources}
    assert reports["openalex"].ok is False
    assert reports["openalex"].degraded is True
    assert "没有为" in reports["openalex"].note
    assert reports["arxiv"].ok is True
    assert any(hit.arxiv_id == "1706.03762" for hit in result.hits)


def test_semantic_scholar_429_is_reported_as_rate_limit():
    def fetch(url: str) -> bytes:
        if "semanticscholar" in url:
            raise OSError("HTTP Error 429: Too Many Requests")
        if "export.arxiv.org" in url:
            return ARXIV_FEED
        raise OSError("没有为 %s 准备响应" % url)

    result = search_sources("attention", limit=5, fetch=fetch)
    s2 = next(r for r in result.sources if r.name == "semanticscholar")
    assert s2.ok is False
    assert "限流" in s2.note


def test_hits_keep_ingest_flag_honest():
    fetch = _dispatcher({
        "export.arxiv.org": EMPTY_FEED,
        "api.openalex.org": _json(OPENALEX_MIXED),
    })
    result = search_sources("resnet", limit=10,
                            sources=["arxiv", "openalex"], fetch=fetch)

    by_title = {hit.title: hit for hit in result.hits}
    attention = by_title["Attention Is All You Need"]
    resnet = by_title["Deep Residual Learning for Image Recognition"]
    assert attention.can_ingest is True and attention.arxiv_id == "1706.03762"
    # CVPR 那篇有 PDF 但不在 arXiv 上 —— 只能给原文链接，不能承诺入库
    assert resnet.can_ingest is False
    assert resnet.pdf_url.endswith("resnet.pdf")


def test_zenodo_hits_carry_pdf_and_plain_text_abstract():
    fetch = _dispatcher({"zenodo.org": _json(ZENODO_RECORDS)})
    result = search_sources("generative ai", limit=10, sources=["zenodo"], fetch=fetch)

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.source == "zenodo"
    assert hit.year == "2024"
    assert hit.venue == "Journal article"
    assert hit.authors == ["Thotapally, Saiprapul", "Rao, Meera"]
    assert hit.pdf_url.endswith("paper.pdf/content")
    assert hit.abs_url == "https://zenodo.org/records/14571411"
    # description 是 HTML，卡片里要显示纯文本
    assert "<" not in hit.abstract and "biases" in hit.abstract
    assert hit.can_ingest is False     # Zenodo 的 PDF 进不了 arXiv 通道


def test_slow_source_times_out_without_holding_the_whole_search(monkeypatch):
    monkeypatch.setattr(sources, "DEADLINE_SECONDS", 0.2)

    def fetch(url: str) -> bytes:
        if "export.arxiv.org" in url:
            return ARXIV_FEED
        time.sleep(1.0)                 # 比总时限慢，模拟卡住的站
        return _json(OPENALEX_MIXED)

    result = search_sources("attention", limit=5,
                            sources=["arxiv", "openalex"], fetch=fetch)
    reports = {r.name: r for r in result.sources}
    assert reports["arxiv"].ok is True
    assert reports["openalex"].ok is False
    assert "超过" in reports["openalex"].note
    assert result.hits                   # 快的那一站的结果照发，不陪着一起等


def test_unknown_source_names_fall_back_to_known_ones():
    fetch = _dispatcher({"export.arxiv.org": ARXIV_FEED})
    result = search_sources("attention", limit=5, sources=["arxiv", "acl"], fetch=fetch)
    assert [r.name for r in result.sources] == ["arxiv"]
    assert result.hits


def test_empty_query_returns_nothing_without_calling_out():
    def fetch(url: str) -> bytes:
        raise AssertionError("空查询不该发任何请求")

    result = search_sources("   ", limit=5, fetch=fetch)
    assert result.hits == []
    assert all(not r.ok for r in result.sources)


# ------------------------------------------------------------------ 引用含金量


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, (12, "openalex")),
        ("12", (12, "openalex")),      # Crossref 偶尔把数字写成字符串
        (7.9, (7, "openalex")),        # 小数向下取整，不四舍五入成 8
        (0, (0, "")),
        (None, (0, "")),
        ("", (0, "")),
        ("N/A", (0, "")),
        (-3, (0, "")),
    ],
)
def test_citation_value_of_junk_means_unknown_not_zero(value, expected):
    """`0` 的语义是"这个站没给数"，所以出处必须一起空掉。

    界面上只要看见 `citations_source` 为空就不印"被引 0" —— 印了的话，
    一篇高被引论文会显得无人问津，反过来一篇真没人引的也会显得是"取不到"。
    两个含义必须能分开。
    """
    assert _cited(value, "openalex") == expected


def test_each_source_citation_field_is_read_under_its_own_name():
    fetch = _dispatcher({
        "export.arxiv.org": EMPTY_FEED,
        "api.openalex.org": _json(OPENALEX_CITED),
        "api.crossref.org": _json(CROSSREF_CITED),
        "api.semanticscholar.org": _json(S2_CITED),
    })
    result = search_sources("x", limit=5,
                            sources=["openalex", "crossref", "semanticscholar"], fetch=fetch)

    got = {(h.source, h.citations, h.citations_source) for h in result.hits}
    assert got == {
        ("openalex", 136007, "openalex"),
        ("crossref", 214800, "crossref"),
        ("semanticscholar", 99000, "semanticscholar"),
    }


def test_arxiv_and_zenodo_hits_carry_no_citation_count():
    """这两站的接口里根本没有这个字段 —— 必须是"没有"，不能补一个 0。"""
    fetch = _dispatcher({
        "export.arxiv.org": ARXIV_FEED,
        "zenodo.org": _json(ZENODO_RECORDS),
    })
    result = search_sources("x", limit=5, sources=["arxiv", "zenodo"], fetch=fetch)

    assert result.hits
    for hit in result.hits:
        assert hit.citations == 0
        assert hit.citations_source == ""


def test_merge_keeps_the_larger_count_and_the_source_that_reported_it():
    arxiv_side = Hit(source="arxiv", source_id="1706.03762", title="Attention Is All You Need")
    openalex_side = Hit(source="openalex", source_id="W1", title="Attention Is All You Need",
                        citations=136007, citations_source="openalex")

    merged = _merge(arxiv_side, openalex_side)
    assert merged.source == "arxiv"                 # 元数据仍以先到的站为准
    assert (merged.citations, merged.citations_source) == (136007, "openalex")

    # 后面那条数更小 —— 不覆盖，出处也不跟着换
    smaller = _merge(merged, Hit(source="crossref", source_id="10.1/x", title="x",
                                 citations=12, citations_source="crossref"))
    assert (smaller.citations, smaller.citations_source) == (136007, "openalex")

    # 打平时保留先出现的来源，这样同一批结果两次刷新不会跳动
    tie = _merge(
        Hit(source="openalex", source_id="W1", title="t", citations=5,
            citations_source="openalex"),
        Hit(source="crossref", source_id="10.1/x", title="t", citations=5,
            citations_source="crossref"),
    )
    assert (tie.citations, tie.citations_source) == (5, "openalex")


# ------------------------------------------------------------------ 端点


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from app.api.routers.tools import router

    app = FastAPI()
    app.include_router(router)
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_search_endpoint_returns_hits_and_per_source_reports(client, monkeypatch):
    fetch = _dispatcher({
        "export.arxiv.org": ARXIV_FEED,
        "api.openalex.org": _json(OPENALEX_MIXED),
        "api.crossref.org": _json(CROSSREF_SAME_TITLE),
    })
    # 路由走的是 net.fetch_bytes，替掉它就等于断网
    monkeypatch.setattr(sources.net, "fetch_bytes",
                        lambda url, **kwargs: fetch(url))

    response = client.get("/api/tools/search", params={"q": "attention is all you need"})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "attention is all you need"
    assert {r["name"] for r in body["sources"]} >= {"arxiv", "openalex", "crossref"}
    titles = [hit["title"] for hit in body["hits"]]
    assert titles.count("Attention Is All You Need") == 1

    attention = next(h for h in body["hits"] if h["arxiv_id"] == "1706.03762")
    assert attention["can_ingest"] is True


def test_search_endpoint_reports_partial_failure_without_500(client, monkeypatch):
    monkeypatch.setattr(sources.net, "fetch_bytes",
                        lambda url, **kwargs: _json(OPENALEX_MIXED))

    response = client.get("/api/tools/search", params={"q": "resnet"})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True          # 有的站挂了、有的站有结果
    assert "暂不可用" in body["note"]
    assert body["hits"]


def test_search_endpoint_accepts_bogus_source_list(client, monkeypatch):
    monkeypatch.setattr(sources.net, "fetch_bytes",
                        lambda url, **kwargs: _json(OPENALEX_MIXED))
    response = client.get("/api/tools/search", params={"q": "resnet", "sources": "acl,foo"})
    assert response.status_code == 200
    # 一个都不认识 → 退回查全部，而不是返回空列表
    assert len(response.json()["sources"]) == len(sources.SOURCES)


def test_search_endpoint_passes_citation_fields_through(client, monkeypatch):
    """序列化漏一行，界面上的筛选就永远是"没有一条带被引数"。

    `SearchHitOut(**asdict(hit))` 是全量展开，字段漏在模型里会直接抛
    ValidationError；这里从响应那头验一遍，而不是相信它自动接上了。
    """
    monkeypatch.setattr(sources.net, "fetch_bytes",
                        lambda url, **kwargs: _json(OPENALEX_CITED))

    response = client.get("/api/tools/search",
                          params={"q": "attention", "sources": "openalex"})
    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["citations"] == 136007
    assert hit["citations_source"] == "openalex"


def test_search_endpoint_without_network_still_answers(client, monkeypatch):
    def boom(url: str, **kwargs):
        raise OSError("网络不可用")

    monkeypatch.setattr(sources.net, "fetch_bytes", boom)
    response = client.get("/api/tools/search", params={"q": "attention"})
    assert response.status_code == 200
    body = response.json()
    assert body["hits"] == []
    assert all(r["ok"] is False for r in body["sources"])
