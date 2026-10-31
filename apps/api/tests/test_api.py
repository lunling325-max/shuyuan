"""HTTP 层的测试。

原则：**测试里绝不打真接口**。DeepSeek 要花钱、会限流、还会让测试结果随模型情绪波动。
所以这里 monkeypatch `app.llm.client.chat_json`，让"大模型"变成一个可预测的桩，
但**保留完整的真实链路** —— 检索、RRF、守卫校验都是真跑的。

这样测的是"接口有没有把链路接对"，而不是"模型答得好不好"。
后者属于评测（docs/EXPERIMENTS.md），不该混进单测。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

# 论文 3.2.1 节里的原句（逐字，含合并进来的脚注编号前的部分）
VERBATIM = (
    "We suspect that for large values of dk, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # 必须在导入 app 之前设好，模块级 STORE_DIR 是导入时求值的
    import os

    os.environ["PAPER_STORE_DIR"] = str(tmp_path_factory.mktemp("papers"))
    from app.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def paper_id(client) -> str:
    data = PDF_V7.read_bytes()
    r = client.post(
        "/api/papers", files={"file": (PDF_V7.name, data, "application/pdf")}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- 健康检查


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------------------------------------------------------------- 上传


def test_upload_rejects_non_pdf(client):
    r = client.post(
        "/api/papers", files={"file": ("evil.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")}
    )
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_upload_returns_parsed_metadata(client, paper_id):
    r = client.get(f"/api/papers/{paper_id}")
    body = r.json()
    assert r.status_code == 200
    assert body["pages"] == 15
    assert body["chunks"] > 0
    assert body["title"]


def test_title_comes_from_the_body_not_the_metadata(client, paper_id):
    """回归测试：这份 PDF 的元数据标题是 "scholarly works."（版权声明），
    直接信它会让文献库里一半的论文同名且全错。"""
    body = client.get(f"/api/papers/{paper_id}").json()
    assert body["title"].lower().startswith("attention is all you need"), body["title"]
    assert "scholarly works" not in body["title"].lower()


def test_uploading_same_file_twice_reuses_one_record(client, paper_id):
    """内容哈希做 id：传两次必须落到同一条记录，库里不能堆两个副本。"""
    data = PDF_V7.read_bytes()
    r = client.post(
        "/api/papers", files={"file": ("改了名字但内容一样.pdf", data, "application/pdf")}
    )
    assert r.status_code == 201
    assert r.json()["id"] == paper_id
    assert len(client.get("/api/papers").json()) == 1


def test_unknown_paper_is_404(client):
    assert client.get("/api/papers/deadbeef").status_code == 404
    assert client.post("/api/papers/deadbeef/qa", json={"question": "hi"}).status_code == 404


# ---------------------------------------------------------------- 切块


def test_chunks_carry_bbox_for_highlighting(client, paper_id):
    chunks = client.get(f"/api/papers/{paper_id}/chunks").json()
    assert len(chunks) > 0
    sections = {c["section"] for c in chunks}
    assert any("3.2" in s for s in sections)
    # 没有坐标就没法在前端画高亮框，溯源会退化成一个页码
    for c in chunks:
        assert c["rects"], f"块 {c['index']} 没有坐标"
        page, *box = c["rects"][0]
        assert page >= 1 and len(box) == 4
        # 阅读器要把原文整段渲染出来做高亮，只有 preview 是不够的
        assert c["text"] and len(c["text"]) >= len(c["preview"].rstrip("…"))


# ---------------------------------------------------------------- 问答


def test_retrieve_mode_survives_the_model_being_down(client, paper_id, monkeypatch):
    """大模型整个挂掉时，retrieve 模式仍要给出证据。

    注意：它**会**试着做检索词扩展（中文问英文库，不扩展就是 0 召回），
    只是扩展失败能降级到原始问题。所以这里断言的是"接得住"，
    而不是"根本没调用" —— 后者是句谎话。
    """
    from app.llm.client import LLMError

    def down(*a, **k):
        raise LLMError("网络不可达")

    monkeypatch.setattr("app.llm.client.chat_json", down)
    r = client.post(
        f"/api/papers/{paper_id}/qa",
        json={"question": "为什么要除以 sqrt(dk)", "mode": "retrieve"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "retrieve"
    assert body["claims"] == []
    assert body["evidence"], "连证据都没召回，说明检索没接上"
    assert body["evidence"][0]["rank"] == 1


def test_answer_mode_returns_anchored_claims(client, paper_id, monkeypatch):
    def fake_llm(user, *, system=None, **kwargs):
        if system and "search queries" in system:
            return {"queries": ["scaled dot product attention sqrt dk"]}
        return {"claims": [{"text": "大 dk 时点积过大会把 softmax 推进梯度极小区", "quotes": [VERBATIM]}]}

    monkeypatch.setattr("app.llm.client.chat_json", fake_llm)
    r = client.post(
        f"/api/papers/{paper_id}/qa", json={"question": "为什么除以 sqrt(dk)？", "k": 6}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "answer"
    claim = body["claims"][0]
    assert claim["status"] == "supported"
    anchor = claim["anchors"][0]
    assert anchor["exact"] is True
    assert anchor["page"] == 4
    assert len(anchor["bbox"]) == 4
    assert body["stats"]["unsupported"] == 0


def test_fabricated_quote_is_dropped_not_softened(client, paper_id, monkeypatch):
    """模型编了一句原文里不存在的话 —— 结论必须被丢弃，且原因要能看见。

    这正是"抗幻觉"该被测的地方：不是期望模型不编，而是期望系统接得住。
    """

    def fake_llm(user, *, system=None, **kwargs):
        if system and "search queries" in system:
            return {"queries": ["scaled dot product attention sqrt dk"]}
        return {
            "claims": [
                {
                    "text": "编造出来的结论",
                    "quotes": [
                        "Attention is clearly the most important invention of the "
                        "twenty-first century and every modern system depends on it "
                        "in ways that were never described in this paper at all"
                    ],
                }
            ]
        }

    monkeypatch.setattr("app.llm.client.chat_json", fake_llm)
    r = client.post(
        f"/api/papers/{paper_id}/qa", json={"question": "为什么除以 sqrt(dk)？"}
    )
    body = r.json()
    assert body["claims"][0]["status"] == "unsupported"
    assert body["claims"][0]["reason"], "被拦下时必须说明原因，否则用户无从判断"
    assert body["claims"][0]["anchors"] == []
    assert body["stats"]["unsupported"] == 1


def test_llm_outage_degrades_to_503_with_a_readable_message(client, paper_id, monkeypatch):
    from app.llm.client import LLMError

    def down(*a, **k):
        raise LLMError("HTTP 402: insufficient balance")

    monkeypatch.setattr("app.llm.client.chat_json", down)
    r = client.post(f"/api/papers/{paper_id}/qa", json={"question": "为什么除以 sqrt(dk)？"})
    assert r.status_code == 503
    assert "大模型" in r.json()["detail"]


def test_empty_question_is_rejected_by_schema(client, paper_id):
    r = client.post(f"/api/papers/{paper_id}/qa", json={"question": ""})
    assert r.status_code == 422


def test_library_survives_a_restart(tmp_path):
    """重启服务后资料库不能空 —— 内存注册表一重启就丢，演示时最难解释。"""
    from app.api.registry import PaperRegistry

    data = PDF_V7.read_bytes()
    first = PaperRegistry(tmp_path)
    before = first.add(data, PDF_V7.name)

    second = PaperRegistry(tmp_path)  # 模拟进程重启
    after = second.get(before.id)
    assert after.title == before.title
    assert after.filename == PDF_V7.name
    assert len(after.chunks) == len(before.chunks)
    assert after.uploaded_at == before.uploaded_at


def test_qa_response_is_json_serializable(client, paper_id, monkeypatch):
    """坐标是浮点数元组，序列化写不好会漏出 NaN 或 numpy 类型。"""

    def fake_llm(user, *, system=None, **kwargs):
        if system and "search queries" in system:
            return {"queries": ["scaled dot product attention"]}
        return {"claims": [{"text": "结论", "quotes": [VERBATIM]}]}

    monkeypatch.setattr("app.llm.client.chat_json", fake_llm)
    r = client.post(f"/api/papers/{paper_id}/qa", json={"question": "为什么除以 sqrt(dk)？"})
    json.dumps(r.json(), allow_nan=False)


# ---------------------------------------------------------------- 版本谱系
#
# 上传顺序是**先 v7 后 v1**，故意反过来：接口若按上传顺序判新旧，
# 这几条测试就会把 v7 当成旧版。判新旧只能看 PDF 元数据里的生成日期。


@pytest.fixture(scope="module")
def v1_upload(client) -> dict:
    """保留整个上传响应 —— 重复识别只在上传那一下发生，事后再查列表是查不到的。"""
    data = PDF_V1.read_bytes()
    r = client.post("/api/papers", files={"file": (PDF_V1.name, data, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="module")
def paper_v1(v1_upload) -> str:
    return v1_upload["id"]


def test_upload_detects_an_existing_version_by_content(paper_id, v1_upload):
    """官方用例 3.1：把 v7 传进去，系统得自己认出"这跟库里那篇是同一篇的不同版本"。

    注意这里比的是**正文指纹**。文件名 `1706.03762v1.pdf` 也带着版号，
    但拿文件名做判断的系统换个命名就瞎了（"论文_终稿_真的终稿.pdf"）。
    """
    assert v1_upload["relation"] == "new_version"
    assert v1_upload["duplicate_of"] == paper_id
    assert v1_upload["similarity"] and v1_upload["similarity"] > 0.3
    assert v1_upload["duplicate_title"] and "attention" in v1_upload["duplicate_title"].lower()


def test_title_ignores_the_arxiv_stamp_and_matches_across_versions(client, paper_id, v1_upload):
    """首页字号最大的是那行旋转的 arXiv 投稿戳（20pt），标题才 17.2pt。

    认字号不剔戳，五篇论文会全叫 "arXiv:1706.03762vN [cs.CL] ..."。
    而两版标题必须一致 —— 版本谱系要在界面上并排显示它们。
    """
    v7 = client.get(f"/api/papers/{paper_id}").json()
    assert v1_upload["title"] == v7["title"] == "Attention Is All You Need"


def test_duplicate_fields_are_only_filled_on_upload(client, paper_id):
    """列表接口不该给每篇都挂一份"重复提示" —— 那是上传时的动作，不是论文的属性。"""
    listed = next(p for p in client.get("/api/papers").json() if p["id"] == paper_id)
    assert listed["duplicate_of"] is None and listed["relation"] is None


def test_compare_orders_old_and_new_by_pdf_creation_date(client, paper_id, paper_v1):
    """v1 后传，但它才是旧版。按上传顺序排会把整份 diff 的方向搞反。"""
    r = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}?summarize=false")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["old"]["id"] == paper_v1
    assert body["new"]["id"] == paper_id
    assert body["relation"] == "new_version"


def test_compare_returns_section_labeled_changes(client, paper_id, paper_v1):
    body = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}?summarize=false").json()

    assert body["changes"], "v1 到 v7 之间不可能没有差异"
    assert body["stats"]["trivial_dropped"] > 20, "降噪没生效，噪声会原样喂给大模型"
    for c in body["changes"]:
        assert c["kind"] in ("insert", "delete", "replace")
        assert c["page"].startswith("p.")
        assert isinstance(c["reference_churn"], bool)
    # 读者要的是"3.2 改了"，不是"第 4 页改了"
    assert sum(1 for c in body["changes"] if c["section"]) >= 0.8 * len(body["changes"])


def test_compare_summary_uses_the_model_and_keeps_sections(client, paper_id, paper_v1, monkeypatch):
    seen = {}

    def fake_llm(user, *, system=None, **kwargs):
        seen["prompt"] = user
        seen["system"] = system or ""
        return {"changes": [{"section": "3.2.2", "kind": "reworded", "what": "补充了缩放因子的解释"}]}

    monkeypatch.setattr("app.llm.client.chat_json", fake_llm)
    body = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}").json()

    assert body["summary_ok"] is True
    assert body["summary"][0]["section"] == "3.2.2"
    # 提示词里必须带着新旧句子 —— 只给"这里变了"，模型只能编
    assert "OLD:" in seen["prompt"] and "NEW:" in seen["prompt"]
    # 参考文献重排不算修订，不许出现在喂给模型的材料里
    assert "References" not in seen["prompt"]


def test_compare_survives_the_model_being_down(client, paper_id, paper_v1, monkeypatch):
    """模型挂了也要给出改动列表，只是没有中文摘要 —— 空摘要必须能被前端识别。"""
    from app.llm.client import LLMError

    def down(*a, **k):
        raise LLMError("HTTP 502")

    monkeypatch.setattr("app.llm.client.chat_json", down)
    r = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}")
    assert r.status_code == 200
    body = r.json()
    assert body["summary_ok"] is False
    assert body["summary"] == []
    assert body["changes"], "摘要失败不该连改动列表一起吞掉"


def test_compare_with_itself_is_rejected(client, paper_id):
    assert client.get(f"/api/papers/{paper_id}/compare/{paper_id}").status_code == 400


def test_compare_unknown_paper_is_404(client, paper_id):
    assert client.get(f"/api/papers/{paper_id}/compare/deadbeef").status_code == 404
    assert client.get(f"/api/papers/deadbeef/compare/{paper_id}").status_code == 404


# ---------------------------------------------------------------- 引用图谱


def papers_of(body: dict) -> list[dict]:
    return [n for n in body["nodes"] if n["kind"] == "paper"]


@pytest.fixture(scope="module")
def bert_id(client, paper_id, paper_v1) -> str:
    """库里已经有 v7 和 v1（同一篇的两版），再加一篇 BERT —— 图上才有一条真边。

    只有两版 Transformer 的话，图是一颗孤零零的点，
    测不出"引用方向对不对"这种最容易错的东西。
    """
    data = PDF_BERT.read_bytes()
    r = client.post("/api/papers", files={"file": (PDF_BERT.name, data, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_graph_merges_two_versions_into_one_node(client, bert_id):
    """库里三个 PDF、两篇论文。v1 和 v7 必须画成一个点。

    不合并的话 PageRank 会把被引票数劈成两半，Transformer 会因为
    "自己传了一次自己"而虚高 —— 排名错得看不出来。
    """
    body = client.get("/api/graph").json()
    papers = [n for n in body["nodes"] if n["kind"] == "paper"]
    assert len(papers) == 2, [n["label"] for n in papers]
    assert body["stats"]["merges"] == 1

    transformer = next(n for n in papers if n["label"].lower().startswith("attention"))
    assert len(transformer["paper_ids"]) == 2
    assert transformer["versions"] == 2


def test_graph_points_the_arrow_at_the_cited_paper(client, bert_id):
    """BERT 引了 Transformer，不是反过来。方向反了图照样画得出来。"""
    body = client.get("/api/graph").json()
    by_id = {n["id"]: n for n in body["nodes"]}
    bert = next(n for n in body["nodes"] if n["label"].startswith("BERT"))
    transformer = next(n for n in body["nodes"] if n["label"].lower().startswith("attention"))

    edge = next(
        (e for e in body["edges"] if e["source"] == bert["id"] and e["target"] == transformer["id"]),
        None,
    )
    assert edge is not None, [f"{by_id[e['source']]['label']} -> {by_id[e['target']]['label']}" for e in body["edges"]]
    assert edge["via"] in ("arxiv", "title")
    assert edge["score"] > 0.7
    assert transformer["cited_by"] >= 1
    assert transformer["rank"] == max(n["rank"] for n in papers_of(body))


def test_graph_coordinates_are_normalized_and_stable(client, bert_id):
    """坐标由后端算好并归一化到 0..1 —— 前端乘以画布尺寸即可。

    同一个库连查两次必须一模一样，否则每次打开图都"跳"一下。
    """
    first = client.get("/api/graph").json()
    second = client.get("/api/graph").json()
    assert [(n["id"], n["x"], n["y"]) for n in first["nodes"]] == [
        (n["id"], n["x"], n["y"]) for n in second["nodes"]
    ]
    for n in first["nodes"]:
        assert 0.0 <= n["x"] <= 1.0 and 0.0 <= n["y"] <= 1.0
    # 最中间那个点是排名最高的库内论文
    top = max(papers_of(first), key=lambda n: n["rank"])
    assert (top["x"], top["y"]) == (0.5, 0.5)


def test_graph_can_be_stripped_down_for_a_small_canvas(client, bert_id):
    """max_external=0：只留库内论文。手机上横向放不下时用得上。"""
    body = client.get("/api/graph?max_external=0").json()
    assert all(n["kind"] == "paper" for n in body["nodes"])
    assert body["stats"]["external_shown"] == 0
    assert body["stats"]["references_parsed"] > 0, "解析出来的参考文献条数得如实汇报"


def test_graph_clamps_an_absurd_max_external(client, bert_id):
    """前端传个 9999 不该 500，也不该真往上塞一千个点。"""
    r = client.get("/api/graph?max_external=9999")
    assert r.status_code == 200
    assert r.json()["stats"]["external_shown"] <= 60


def test_graph_notes_explain_what_was_left_out(client, bert_id):
    """图上少画了东西必须写出来。悄悄省略等于让观众看错结论。"""
    body = client.get("/api/graph").json()
    assert body["notes"], "至少要说清统计口径"
    assert any("引用" in s for s in body["notes"])


# ---------------------------------------------------------------- 删除
# 放在文件末尾：删除会改动共享的注册表，不能让前面的用例跟着漂。


def test_delete_takes_the_files_with_it(client):
    """删一篇必须三处一起删：内存、sidecar、落盘副本。

    少删一处就会"删了又回来"：只删内存，重启时 sidecar 把它复活；
    只删 sidecar，裸 PDF 扫描那一步照样能把它捞回来。
    """
    from app.api.main import registry

    data = PDF_BERT.read_bytes()
    r = client.post("/api/papers", files={"file": (PDF_BERT.name, data, "application/pdf")})
    assert r.status_code in (201, 200), r.text
    pid = r.json()["id"]
    paper = registry.get(pid)
    pdf_path = paper.path
    meta_path = registry.store_dir / f"{pid}.json"
    assert pdf_path.exists(), "上传后应该有落盘副本"

    assert client.delete(f"/api/papers/{pid}").status_code == 204

    assert not pdf_path.exists(), "落盘副本必须一起删"
    assert not meta_path.exists(), "sidecar 必须一起删"
    assert client.get(f"/api/papers/{pid}").status_code == 404
    assert pid not in [p["id"] for p in client.get("/api/papers").json()]


def test_deleted_paper_does_not_come_back_after_a_restart(client):
    """删完之后重建一次注册表（等价于重启），它不许复活。"""
    from app.api.main import registry
    from app.api.registry import PaperRegistry

    data = PDF_BERT.read_bytes()
    pid = client.post(
        "/api/papers", files={"file": (PDF_BERT.name, data, "application/pdf")}
    ).json()["id"]
    assert client.delete(f"/api/papers/{pid}").status_code == 204

    reloaded = PaperRegistry(registry.store_dir)
    assert pid not in [p.id for p in reloaded.all()]


def test_delete_unknown_paper_is_404(client):
    assert client.delete("/api/papers/does-not-exist").status_code == 404


# ---------------------------------------------------------------- 删除的 Windows 坑
#
# 这些用例用独立的临时注册表，不碰上面那个共享的。


def test_unlink_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    """WinError 32 是偶发的（杀毒/索引器在文件刚落地时短暂持有句柄），必须重试。

    踩过的坑：真机上一次 DELETE 返回 500，内存里已经把这篇文章摘掉了，
    磁盘上的 PDF 还在 —— 重启时"裸 PDF 扫描"把它捞回来，就是"删了又回来"。
    """
    from pathlib import Path as _Path

    from app.api import registry as reg_mod

    target = tmp_path / "a.pdf"
    target.write_bytes(b"%PDF")
    calls = {"n": 0}
    real_unlink = _Path.unlink

    def flaky(self, missing_ok=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "另一个程序正在使用此文件")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(_Path, "unlink", flaky)
    monkeypatch.setattr(reg_mod.time, "sleep", lambda _s: None)

    reg_mod._unlink_retry(target)

    assert calls["n"] == 3
    assert not target.exists()


def test_remove_rolls_back_when_the_file_stays_locked(tmp_path, monkeypatch):
    """实在删不掉时必须**撤销内存里的删除**，并把 sidecar 留在原处。

    宁可整个操作失败、库里状态一致，也不要留下"内存没有、磁盘还在"的幽灵文件。
    """
    from app.api import registry as reg_mod
    from app.api.registry import PaperBusy, PaperRegistry

    reg = PaperRegistry(tmp_path / "store")
    paper = reg.add(PDF_V7.read_bytes(), PDF_V7.name)
    meta_path = reg.store_dir / f"{paper.id}.json"
    assert meta_path.exists()

    def always_busy(_path, **_kwargs):
        raise PaperBusy("locked")

    monkeypatch.setattr(reg_mod, "_unlink_retry", always_busy)

    with pytest.raises(PaperBusy):
        reg.remove(paper.id)

    assert reg.get(paper.id) is paper, "删失败要把内存里的那篇放回去"
    assert meta_path.exists(), "sidecar 不能先删 —— 否则元数据丢了、文件还在"
    assert paper.path.exists()

