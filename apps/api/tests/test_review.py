"""领域综述接口的测试（全程离线）。

重点不是"综述写得好不好"，而是四条底线：
1. 模型编出来的证据编号一律丢弃；
2. 引用文本来自证据块原文，不是模型生成的；
3. future 的 open/solved 判断原样透传，编号同样要校验；
4. 模型不可用仍返回 200，references 保留真实文献；库空才是 400。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

TOPIC = "attention"

VALID_SECTIONS = [
    {
        "heading": "研究现状",
        "points": [
            {"text": "注意力机制成为主流建模方式", "evidence": [1, 999]},
            {"text": "需要报告效率与效果的权衡", "evidence": []},
        ],
    },
    {
        "heading": "方法脉络",
        "points": [{"text": "从循环网络走向自注意力", "evidence": [1, 1]}],
    },
    {"heading": "研究空白", "points": [{"text": "长文本效率仍缺统一评测", "evidence": []}]},
]


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """直接往 registry 里灌两篇真实 PDF，不依赖未挂载的上传路由。"""
    from app.api.registry import PaperRegistry

    reg = PaperRegistry(tmp_path_factory.mktemp("papers_review"))
    for path in (PDF_V7, PDF_BERT):
        reg.add(path.read_bytes(), path.name)
    return reg


@pytest.fixture()
def client(registry, monkeypatch):
    """main.py 由调用方挂载；测试里单独装一个小 app，只测本路由。"""
    from app.api import deps
    from app.api.routers.review import router

    monkeypatch.setattr(deps, "registry", registry)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _fake_llm(sections: list | None = None, future: list | None = None):
    payload = {"sections": sections or [], "future": future or []}

    def call(user, *, system=None, **kwargs):
        return payload

    return call


def _fake_paper(i: int):
    """轻量假论文：只提供 collect_evidence / 年份推导用到的字段。"""
    chunk = SimpleNamespace(
        index=0,
        text=f"attention based paper {i}",
        section="3 Method",
        page_start=1,
        page_end=1,
    )
    hits = [SimpleNamespace(score=1.0 - i * 0.01, chunk=chunk)]
    return SimpleNamespace(
        id=f"p{i:02d}",
        title=f"Paper {i}",
        created_at="2020-01-01",
        arxiv_id="",
        uploaded_at="2024-01-01T00:00:00+00:00",
        chunks=[chunk],
        bm25=SimpleNamespace(search=lambda q, k=2: hits),
    )


def test_normal_review_backfills_real_evidence_and_quotes(client, registry, monkeypatch):
    future = [
        {"idea": "稀疏注意力", "status": "open", "evidence": [2], "why": "尚无统一方案"},
        {"idea": "长文本评测基准", "status": "solved", "evidence": [1], "why": "后续论文已覆盖"},
    ]
    monkeypatch.setattr(
        "app.llm.client.chat_json", _fake_llm(VALID_SECTIONS, future)
    )

    r = client.post("/api/review", json={"topic": TOPIC, "max_papers": 2})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["topic"] == TOPIC
    assert body["degraded"] is False

    # papers 是本次实际用到的真实论文，且按相关度排
    library_ids = {p.id for p in registry.all()}
    assert body["papers"]
    assert {p["id"] for p in body["papers"]} <= library_ids
    assert all(p["title"] and p["year"] for p in body["papers"])

    # 999 不存在 -> 丢掉；1 存在 -> 保留；重复编号去重
    assert body["sections"][0]["points"][0]["evidence"] == [1]
    assert body["sections"][0]["points"][1]["evidence"] == []
    assert body["sections"][1]["points"][0]["evidence"] == [1]
    assert body["sections"][2]["heading"] == "研究空白"

    # references 覆盖 sections 与 future 引用的编号，按编号去重
    refs = body["references"]
    assert [ref["n"] for ref in refs] == [1, 2]
    for ref in refs:
        assert 0 < len(ref["quote"]) <= 200
        assert "\n" not in ref["quote"], "quote 必须压成单行"
        paper = registry.get(ref["paper_id"])
        flattened = [" ".join(c.text.split()) for c in paper.chunks]
        assert any(ref["quote"] in text for text in flattened), "引用必须是原文片段"


def test_invented_evidence_numbers_are_dropped(client, monkeypatch):
    sections = [
        {
            "heading": "研究现状",
            "points": [{"text": "完全没有依据的要点", "evidence": [42, 999]}],
        },
        {"heading": "方法脉络", "points": [{"text": "另一条无依据要点", "evidence": [999]}]},
        {"heading": "研究空白", "points": [{"text": "空白", "evidence": [42]}]},
    ]
    future = [{"idea": "编出来的方向", "status": "open", "evidence": [999], "why": "无"}]
    monkeypatch.setattr(
        "app.llm.client.chat_json", _fake_llm(sections, future)
    )

    body = client.post("/api/review", json={"topic": TOPIC}).json()

    assert body["degraded"] is False
    assert all(
        point["evidence"] == []
        for section in body["sections"]
        for point in section["points"]
    )
    assert body["future"][0]["evidence"] == []
    assert body["references"] == [], "没有任何真实编号被引用时 references 应为空"


def test_llm_outage_degrades_but_keeps_references_and_raw_future(client, monkeypatch):
    from app.llm.client import LLMError

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    r = client.post("/api/review", json={"topic": TOPIC})

    assert r.status_code == 200, "模型不可用不能 500"
    body = r.json()
    assert body["degraded"] is True
    assert body["sections"] == []
    assert body["references"], "断网时仍要能展示库里可引用的原文"
    assert body["future"], "future 退化成原文摘录，而不是空白"
    for idea in body["future"]:
        assert idea["status"] == "open"
        assert idea["evidence"] == []
    assert body["note"], "必须说明这是未经模型判断的原始摘录"


def test_empty_library_returns_400(client, monkeypatch, tmp_path):
    from app.api import deps
    from app.api.registry import PaperRegistry

    monkeypatch.setattr(deps, "registry", PaperRegistry(tmp_path / "empty"))

    r = client.post("/api/review", json={"topic": TOPIC})

    assert r.status_code == 400
    assert "上传" in r.json()["detail"]


def test_endpoint_clamps_out_of_range_max_papers(client, monkeypatch):
    monkeypatch.setattr(
        "app.llm.client.chat_json", _fake_llm(VALID_SECTIONS, [])
    )

    body = client.post(
        "/api/review", json={"topic": TOPIC, "max_papers": 0}
    ).json()

    assert len(body["papers"]) == 1, "max_papers=0 应夹到 1，而不是 422"


def test_build_review_clamps_max_papers_to_upper_bound(monkeypatch):
    from app.rag.review import MAX_PAPERS, build_review

    papers = [_fake_paper(i) for i in range(15)]
    llm = _fake_llm(VALID_SECTIONS, [])

    high = build_review("attention", papers, max_papers=999, llm_call=llm)
    low = build_review("attention", papers, max_papers=0, llm_call=llm)

    assert len(high.papers) == MAX_PAPERS == 12
    assert len(low.papers) == 1


def test_zero_recall_falls_back_to_body_chunk_not_page_one():
    """0 召回时退到"第一篇正文块"，而不是第 0 块。

    这是实测踩出来的坑：中文主题 × 英文库召回 0 条，退回 chunks[0] 之后
    所有"引用原文"都变成了第 1 页的作者名单和 MIT Press 版权声明 ——
    技术上是原文，作为证据却毫无信息量。
    """
    from app.rag import review as review_mod

    front = SimpleNamespace(
        index=0,
        text="Ashish Vaswani, Noam Shazeer. Copyright in this work has been licensed.",
        section="",
        page_start=1,
        page_end=1,
        n_tokens=12,
    )
    body = SimpleNamespace(
        index=1,
        text="The Transformer uses self-attention. It dispenses with recurrence entirely.",
        section="3 Model Architecture",
        page_start=2,
        page_end=2,
        n_tokens=60,
    )
    paper = SimpleNamespace(
        id="p1",
        title="Attention Is All You Need",
        created_at="2017-06-12",
        arxiv_id="",
        uploaded_at="",
        chunks=[front, body],
        bm25=SimpleNamespace(search=lambda q, k=2: []),
    )

    evidence, selected = review_mod.collect_evidence([paper], "稀疏注意力在长文档里的作用")

    assert [e.n for e in evidence] == [1]
    assert evidence[0].text == body.text
    assert selected == [paper]


def test_search_merges_queries_and_keeps_best_score_per_chunk():
    from app.rag.review import _search

    c0, c1 = SimpleNamespace(index=0), SimpleNamespace(index=1)
    table = {
        "self-attention": [
            SimpleNamespace(score=1.0, chunk=c0),
            SimpleNamespace(score=0.4, chunk=c1),
        ],
        "long document": [SimpleNamespace(score=3.0, chunk=c1)],
    }
    paper = SimpleNamespace(
        bm25=SimpleNamespace(search=lambda q, k=2: table.get(q, []))
    )

    got = _search(paper, ["self-attention", "long document"])

    assert [h.chunk.index for h in got] == [1, 0], "按最高分排序，且同块只留一条"
    assert got[0].score == 3.0, "同块保留分数更高的那次命中"


def test_reference_list_chunks_are_never_used_as_evidence():
    """参考文献块命中也要丢掉。

    实测里 GPT-4 报告的 References 块被选成了证据，前端引用栏显示
    "[23] Xi Chen, Xiao Wang, ..." —— 原文没错，但当论据毫无意义。
    """
    from app.rag.review import _search

    refs = SimpleNamespace(
        index=0,
        section="References",
        page_start=19,
        page_end=19,
        text="[23] Xi Chen, Xiao Wang, Soravit Changpinyo. [24] A. Author. [25] B.",
    )
    body = SimpleNamespace(
        index=1,
        section="4 Results",
        page_start=8,
        page_end=8,
        text="GPT-4 is a Transformer-style model pre-trained to predict the next token.",
    )
    paper = SimpleNamespace(
        bm25=SimpleNamespace(
            search=lambda q, k=2: [
                SimpleNamespace(score=2.0, chunk=refs),
                SimpleNamespace(score=1.0, chunk=body),
            ]
        )
    )

    got = _search(paper, ["gpt"])

    assert [h.chunk.index for h in got] == [1]


def test_future_fallback_skips_a_trailing_reference_list():
    """Future Work 找不到时退到最后一块正文，不能退到参考文献。"""
    from app.rag.review import _future_excerpts

    body = SimpleNamespace(
        index=0, section="Conclusion", text="Limitations remain open for future study."
    )
    refs = SimpleNamespace(
        index=1, section="References", text="[1] A. Author. [2] B. Author. [3] C. Author."
    )
    paper = SimpleNamespace(id="p1", title="T", chunks=[body, refs])

    excerpts = _future_excerpts([paper])

    assert [chunk.index for _paper, chunk in excerpts] == [0]


def test_search_survives_a_broken_query():
    """一条检索式把索引查崩了，不能连累其它检索式的结果。"""
    from app.rag.review import _search

    chunk = SimpleNamespace(index=7)
    table = {"ok": [SimpleNamespace(score=0.9, chunk=chunk)]}

    def search(q, k=2):
        if q == "boom":
            raise RuntimeError("index corrupted")
        return table.get(q, [])

    got = _search(SimpleNamespace(bm25=SimpleNamespace(search=search)), ["boom", "ok"])

    assert [h.chunk.index for h in got] == [7]


def test_build_review_expands_topic_before_searching():
    """中文主题要先扩展成英文检索式，再拿扩展结果去查库。

    不扩展就等于拿中文字符串去查英文 BM25 —— 0 召回，只能吃兜底块，
    综述里每条引用都会变成版权声明。
    """
    from app.rag.review import build_review

    body = SimpleNamespace(
        index=3,
        text="Scaled dot-product attention divides by sqrt(dk).",
        section="3.2.1",
        page_start=3,
        page_end=3,
        n_tokens=40,
    )
    table = {"scaled dot-product attention": [SimpleNamespace(score=2.0, chunk=body)]}
    paper = SimpleNamespace(
        id="p1",
        title="Attention Is All You Need",
        created_at="2017-06-12",
        arxiv_id="",
        uploaded_at="",
        chunks=[body],
        bm25=SimpleNamespace(search=lambda q, k=2: table.get(q, [])),
    )

    asked: list[str] = []

    def llm(user, *, system=None, **kwargs):
        if user.startswith("Question:"):
            asked.append(user)
            return {"queries": ["scaled dot-product attention"]}
        return {"sections": VALID_SECTIONS, "future": []}

    result = build_review("稀疏注意力", [paper], llm_call=llm)

    assert asked, "必须先做检索词扩展"
    assert result.references, "扩展后的英文检索式应召回正文块"
    assert result.references[0].text == body.text


def test_future_open_solved_status_is_passed_through(client, monkeypatch):
    future = [
        {"idea": "方向 A", "status": "open", "evidence": [1], "why": "还没人做"},
        {"idea": "方向 B", "status": "solved", "evidence": [1], "why": "已被后续工作解决"},
    ]
    monkeypatch.setattr(
        "app.llm.client.chat_json", _fake_llm(VALID_SECTIONS, future)
    )

    body = client.post("/api/review", json={"topic": TOPIC}).json()

    assert [(item["idea"], item["status"]) for item in body["future"]] == [
        ("方向 A", "open"),
        ("方向 B", "solved"),
    ]
    assert body["future"][0]["why"] == "还没人做"
    assert body["future"][1]["why"] == "已被后续工作解决"


def _versioned(i: int, *, title: str, ver: int, scanned: bool, score: float):
    """同一篇的某一版：id / 版本号 / 是否扫描件 / 检索分都能单独摆。"""
    chunk = SimpleNamespace(
        index=0, text=f"body of version {ver}", section="3 Method",
        page_start=i, page_end=i,
    )
    hits = [SimpleNamespace(score=score, chunk=chunk)]
    return SimpleNamespace(
        id=f"p{i:02d}",
        title=title,
        created_at="",
        arxiv_id="1706.03762",
        uploaded_at="2024-01-01T00:00:00+00:00",
        arxiv_version=ver,
        scanned=scanned,
        chunks=[chunk],
        bm25=SimpleNamespace(search=lambda q, k=2: hits),
    )


def test_one_version_per_work_prefers_text_layer_over_ocr():
    """扫描件 v1 的分再高也让位给有文字层的 v7。

    OCR 会把 l 认成 1、0 认成 O；把它的识别结果当"作者原话"引用，
    溯源就成了反向的 —— 越像原文越错。
    """
    from app.rag.works import one_version_per_work

    scanned = _versioned(1, title="Attention Is All You Need", ver=1, scanned=True, score=9.0)
    clean = _versioned(2, title="Attention Is All You Need", ver=7, scanned=False, score=1.0)

    assert one_version_per_work([scanned, clean]) == [clean]


def test_collect_evidence_uses_one_slot_per_work():
    """两版同一篇只占一个名额，页数才等于"用了几篇工作"。"""
    from app.rag import review as review_mod

    scanned = _versioned(1, title="Attention Is All You Need", ver=1, scanned=True, score=9.0)
    clean = _versioned(2, title="Attention Is All You Need", ver=7, scanned=False, score=3.0)
    other = _versioned(3, title="BERT: Pre-training of Deep Bidirectional Transformers",
                       ver=2, scanned=False, score=1.0)

    evidence, selected = review_mod.collect_evidence(
        [scanned, clean, other], "attention", max_papers=2
    )

    assert [p.id for p in selected] == ["p02", "p03"]
    assert {e.paper_id for e in evidence} == {"p02", "p03"}
    assert evidence[0].version == 7
    assert evidence[0].scanned is False


def test_reference_carries_version_so_quote_can_be_traced(monkeypatch):
    """引用要能看出出自哪一版：v7 和 v1 扫描件的同一页，文字并不一样。"""
    from app.api import deps
    from app.api.routers.review import router

    scanned = _versioned(1, title="Attention Is All You Need", ver=1, scanned=True, score=9.0)
    clean = _versioned(2, title="Attention Is All You Need", ver=7, scanned=False, score=3.0)
    monkeypatch.setattr(deps, "registry", SimpleNamespace(all=lambda: [scanned, clean]))
    monkeypatch.setattr("app.llm.client.chat_json", _fake_llm(VALID_SECTIONS, []))

    app = FastAPI()
    app.include_router(router)
    body = TestClient(app).post("/api/review", json={"topic": TOPIC}).json()

    ref = body["references"][0]
    assert ref["paper_id"] == "p02"
    assert ref["version"] == 7
    assert ref["scanned"] is False
