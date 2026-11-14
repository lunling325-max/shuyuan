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


def test_upload_records_authors_and_affiliations(client, paper_id):
    """模块一要求"作者及机构信息"。这一条守的是**接口出口** ——
    `Paper` 里有、sidecar 里有，但 `PaperOut` 漏了，界面照样一片空白。"""
    body = client.get(f"/api/papers/{paper_id}").json()
    assert body["authors"][0] == "Ashish Vaswani"
    assert len(body["authors"]) == 8
    assert body["affiliations"] == ["Google Brain", "Google Research", "University of Toronto"]


def test_author_list_is_copied_not_handed_out(client, paper_id):
    """每次响应都必须是新列表。交出去的要是注册表里那一份，调用方（前端、
    Agent 工具）顺手 append 一下，下一批读者看到的就是被改过的作者名单。"""
    first = client.get(f"/api/papers/{paper_id}").json()
    first["authors"].append("Someone Else")
    assert "Someone Else" not in client.get(f"/api/papers/{paper_id}").json()["authors"]


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



def test_blank_page_upload_does_not_promise_ocr_that_already_ran(client):
    """全空白页上传失败时，不能告诉用户"解析时会自动识别"。

    实测原话是：「共 1 页，全部缺少文字层，判定为纯扫描件；本机 OCR 引擎（RapidOCR）
    可用，解析时会自动识别。」—— 可解析刚刚跑完、OCR 也逐页认过了，仍然什么都没抽到。
    用户照着这句话重传，只会再等一遍同样的失败。
    """
    import pymupdf

    blank = pymupdf.open()
    blank.new_page()  # 一页白纸：既没有文字层，OCR 也无字可认
    payload = blank.tobytes()
    blank.close()

    r = client.post("/api/papers", files={"file": ("blank.pdf", payload, "application/pdf")})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "解析时会自动识别" not in detail, "解析已经结束，这句话是假的"
    assert "已用 OCR" in detail or "缺少文字层" in detail


def test_empty_body_reason_distinguishes_the_three_kinds_of_empty():
    """三种"空"必须给三种话：没文字层且能 OCR / 没文字层且没引擎 / 有文字层却切不出块。"""
    from app.api.main import _empty_body_reason
    from app.parsing.scan import ScanReport

    tried = ScanReport(
        pages=2, readable_pages=0, scanned_pages=[1, 2], text_chars=0,
        ocr_available=True, note="判定为纯扫描件；本机 OCR 引擎（RapidOCR）可用，解析时会自动识别。",
    )
    assert "已用 OCR" in _empty_body_reason(tried)
    assert "解析时会自动识别" not in _empty_body_reason(tried)

    no_engine = ScanReport(
        pages=2, readable_pages=0, scanned_pages=[1, 2], text_chars=0,
        ocr_available=False, note="判定为纯扫描件；本机没有可用的 OCR 引擎。",
    )
    assert "没有可用的 OCR 引擎" in _empty_body_reason(no_engine)
    assert "uv sync" in _empty_body_reason(no_engine)

    weird = ScanReport(
        pages=1, readable_pages=1, scanned_pages=[], text_chars=500,
        ocr_available=True, note="第 1 页有文字层，无需 OCR。",
    )
    assert "未能切分出正文段落" in _empty_body_reason(weird)


def test_upload_filename_keeps_only_the_basename():
    """客户端送进来的文件名可能带路径，存库前必须只留最后一段。

    实测能传 `../../evil.pdf` 进来：它会原样存进 sidecar，再拼成
    `Content-Disposition: attachment; filename="../../evil.pdf"`。
    浏览器会自己剥路径，所以不是漏洞；但"库里躺着一个带路径的文件名"迟早被人当路径用，
    文献库里显示 `../../evil.pdf` 也说不通。
    """
    from app.api.registry import _clean_filename

    assert _clean_filename("../../evil.pdf") == "evil.pdf"
    assert _clean_filename(r"C:\Users\x\paper.pdf") == "paper.pdf"
    assert _clean_filename("论文 v1.pdf") == "论文 v1.pdf"
    # 只剩点号的名字不能留：. 和 .. 在文件系统里有特殊含义
    assert _clean_filename("..") == "untitled.pdf"
    assert _clean_filename("") == "untitled.pdf"
    assert _clean_filename("   ") == "untitled.pdf"
    # 超长文件名截断，别让它把 sidecar 撑爆
    assert len(_clean_filename("a" * 500 + ".pdf")) == 160


# ---------------------------------------------------------------- 扫描件的位置
#
# 这一批全部来自录演示视频时暴露的问题：上传一份 1706.03762v1 的扫描件，
# 文献库里它的名字是 `1706.03762v1_scan`、日期是"日期未知"，版本谱系默认
# 对成了「BERT ↔ 扫描件」，方向还反了（v1 被判成更新的那版）。


def test_compare_says_which_clue_decided_the_order(client, paper_id, paper_v1):
    """新旧判出来了还不够，得说清凭什么 —— 否则没人能复核。

    这两版的 PDF 生成日期也差得很远，但优先级最高的依据是 arXiv 版本号：
    日期只到"天"，同一天导出的两版它判不出来，扫描件更是连日期都没有。
    """
    body = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}?summarize=false").json()
    assert body["order_by"] == "arxiv_version"
    assert body["old"]["arxiv_version"] == 1
    assert body["new"]["arxiv_version"] == 7
    assert body["ocr_side"] == ""


def test_a_scanned_side_is_flagged_in_the_compare_response(
    client, paper_id, paper_v1, monkeypatch
):
    """扫描件那一侧的逐字差异里混着 OCR 误差，接口必须把它标出来。

    不标的话前端只能照常显示"删除 50 句"，读起来像作者把半篇论文删了。
    """
    from app.api.main import registry

    paper = registry.get(paper_v1)
    monkeypatch.setattr(paper, "scanned", True)
    body = client.get(f"/api/papers/{paper_id}/compare/{paper_v1}?summarize=false").json()
    assert body["ocr_side"] == "old"


def test_arxiv_version_beats_upload_time_for_the_same_paper():
    """同一个 arXiv 号、版本号都取得到时，vN 说了算，上传时间没有发言权。"""
    from pathlib import Path

    from app.api.registry import Paper, order_versions

    def make(pid, version, created, uploaded):
        return Paper(
            id=pid, filename=f"{pid}.pdf", title="Attention Is All You Need",
            pages=1, chunks=[], path=Path(f"{pid}.pdf"), uploaded_at=uploaded,
            created_at=created, arxiv_id="1706.03762", arxiv_version=version,
        )

    # 扫描件：没有生成日期，而且**后传**，但它才是 v1
    scan = make("scan", 1, "", "2026-11-01T07:34:43+00:00")
    text = make("text", 7, "2024-04-10", "2026-10-31T10:00:01+00:00")
    old, new, basis = order_versions(scan, text)
    assert (old.id, new.id, basis) == ("scan", "text", "arxiv_version")

    # 反过来传参也不能改结论
    assert order_versions(text, scan)[0].id == "scan"


def test_order_falls_back_to_the_date_then_admits_it_is_only_upload_time():
    """版本号取不到时按生成日期；两样都没有时只能说"按上传时间排"，且明说。"""
    from pathlib import Path

    from app.api.registry import Paper, order_versions

    def make(pid, created, uploaded):
        return Paper(
            id=pid, filename=f"{pid}.pdf", title=pid, pages=1, chunks=[],
            path=Path(f"{pid}.pdf"), uploaded_at=uploaded, created_at=created,
        )

    a = make("a", "2017-06-13", "2026-11-02T00:00:00+00:00")
    b = make("b", "2024-04-10", "2026-10-30T00:00:00+00:00")
    assert order_versions(a, b)[:2] == (a, b)
    assert order_versions(a, b)[2] == "date"

    # 两份都没有生成日期（典型：两份扫描件）→ 依据降级成上传时间，并如实标注
    c = make("c", "", "2026-11-02T00:00:00+00:00")
    d = make("d", "", "2026-10-30T00:00:00+00:00")
    old, new, basis = order_versions(c, d)
    assert (old.id, new.id, basis) == ("d", "c", "uploaded_at")


def test_arxiv_version_comes_from_the_stamp_and_from_the_filename(tmp_path):
    """有文字层读投稿戳，扫描件读文件名 —— 两条路都得走通。"""
    import pymupdf

    from app.api.registry import PaperRegistry

    stamped = tmp_path / "stamped.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023")
    doc.save(str(stamped))
    doc.close()
    assert PaperRegistry._arxiv_version(stamped, "whatever.pdf") == 7

    # 扫描件：首页一个字都抽不出来，只能从文件名认
    blank = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(blank))
    doc.close()
    assert PaperRegistry._arxiv_version(blank, "1706.03762v1_scan.pdf") == 1
    assert PaperRegistry._arxiv_version(blank, "no-version-here.pdf") == 0


def test_arxiv_stamp_date_is_read_as_a_separate_field(tmp_path):
    """投稿戳上的日期要单独取，不能拿去冒充 PDF 生成日期。

    arXiv 从 2025 年起用 GenPDF 导出，元数据里根本没有 creationDate ——
    FSA（2508.18224v3）实测两个日期字段都是空串。只写「生成日期未知」会让人
    以为解析坏了；戳上的日期是真的，但它说的是"这一版什么时候投的"。
    """
    import pymupdf

    from app.api.registry import PaperRegistry

    def stamp(name, text):
        path = tmp_path / name
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), text)
        doc.save(str(path))
        doc.close()
        return path

    assert PaperRegistry._arxiv_date(stamp("a.pdf", "arXiv:2508.18224v3  [cs.DC]  1 Jul 2026")) == "2026-07-01"
    assert PaperRegistry._arxiv_date(stamp("b.pdf", "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023")) == "2023-08-02"
    # 早期投稿戳不一定带版本号
    assert PaperRegistry._arxiv_date(stamp("c.pdf", "arXiv:1706.03762 [cs.CL] 17 Jun 2017")) == "2017-06-17"
    # 不是 arXiv 的 PDF（报告、自印稿）：没有戳就返回空串，不许瞎猜
    assert PaperRegistry._arxiv_date(stamp("d.pdf", "GPT-4 Technical Report")) == ""
    # 分类框里的字不能把日期吃掉，日期也不可能超出月份
    assert PaperRegistry._arxiv_date(stamp("e.pdf", "arXiv:2310.03294v2 [cs.LG] 31 Mar 2024")) == "2024-03-31"


def test_all_caps_title_takes_the_metadata_casing(tmp_path):
    """版面把标题排成全大写、元数据里是同样字句的正常写法时，用元数据的大小写。

    只换大小写、一个字都不换：两者压平后必须完全相同才动手，
    否则 1706.03762v7 那种元数据标题（"scholarly works."）就会顶掉真标题。
    """
    import pymupdf

    from app.api.registry import PaperRegistry

    def build(name, layout, meta_title):
        path = tmp_path / name
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 100), layout, fontsize=20)
        doc.set_metadata({"title": meta_title})
        doc.save(str(path))
        doc.close()
        return path

    caps = build(
        "caps.pdf",
        "FSA: AN ALTERNATIVE EFFICIENT IMPLEMENTATION",
        "FSA: An Alternative Efficient Implementation",
    )
    assert PaperRegistry._title(caps, "2508.18224.pdf") == "FSA: An Alternative Efficient Implementation"

    # 元数据是别的东西（真实坑：arXiv v7 的 metadata title 是版权声明）→ 保持版面标题
    other = build("other.pdf", "ATTENTION IS ALL YOU NEED", "scholarly works.")
    assert PaperRegistry._title(other, "1706.03762v7.pdf") == "ATTENTION IS ALL YOU NEED"


def test_a_title_split_into_many_spans_is_put_back_together():
    """排版器把一行标题拆成一堆 span 时，标题不能碎掉。

    2310.03294 的标题在 PDF 里就是这样：逐字调字距让 PyMuPDF 断成
    `D`,`F`,`A`,`:`,`Distributed`,`Memory-efficient`,`Attention`。
    以前只收"字号恰好等于全页最大值"的 span 再取前三个，拼出来是 `D F A`，
    短到被判定"不是标题"，标题就掉到 OCR 那一路 —— 而 OCR 吃掉了空格，
    文献库里于是出现 `DISTFLASHATTN:Distributed Memory-efficientAttention`。

    这里守的是**行内拼 span 不加分隔符**：空格有没有由排版器的字符决定，
    补一个会把 `Memory-efficient` 拆成 `Memory- efficient`。
    """
    from app.api.registry import PaperRegistry

    # 真实的那一行（2310.03294 首页）：大写首字母 14.5、小型大写字母 11.6，
    # 连空格都是单独的 span
    line = {
        "bbox": (108.4, 82.0, 504.0, 96.0),
        "spans": [
            {"text": "D", "size": 14.5},
            {"text": "IST", "size": 11.6},
            {"text": "F", "size": 14.5},
            {"text": "LASH", "size": 11.6},
            {"text": "A", "size": 14.5},
            {"text": "TTN", "size": 11.6},
            {"text": ":", "size": 14.5},
            {"text": " ", "size": 14.5},
            {"text": "Distributed", "size": 14.5},
            {"text": " ", "size": 14.5},
            {"text": "Memory-efficient", "size": 14.5},
            {"text": " ", "size": 14.5},
            {"text": "Attention", "size": 14.5},
        ],
    }
    assert PaperRegistry._line_of(line) == (14.5, "DISTFLASHATTN: Distributed Memory-efficient Attention")

    stamp = {"bbox": (0, 0, 1, 1), "spans": [{"text": "arXiv:2310.03294v2 [cs.LG] 31 Mar 2024", "size": 20.0}]}
    assert PaperRegistry._line_of(stamp) is None, "投稿戳不能当标题的一部分"


def test_the_second_title_line_survives_a_slightly_smaller_size(tmp_path):
    """标题折行时第二行常常小 0.2pt（实测 14.3 对 14.5），不能只取第一行。"""
    import pymupdf

    from app.api.registry import PaperRegistry

    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "DISTFLASHATTN: Distributed Memory-efficient Attention", fontsize=14.5)
    page.insert_text((72, 98), "for Long-context LLMs Training", fontsize=14.3)
    page.insert_text((72, 140), "Abstract", fontsize=12.0)
    page.insert_text((72, 160), "Body text that is clearly smaller than the title line.", fontsize=10.0)
    doc.save(str(path))
    doc.close()

    assert PaperRegistry._title_from_layout(path) == (
        "DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training"
    )


def test_an_old_sidecar_title_is_recomputed_after_a_rule_change(tmp_path):
    """标题识别规则修好了，库里那行字也得跟着改。

    否则最坏的情况是：代码修了、测试绿了，而服务器上那份 sidecar 里躺着的
    还是旧规则算出来的连体字 —— 用户看到的和修之前一模一样。
    """
    import json

    import pymupdf

    from app.api.registry import PaperRegistry

    pdf = tmp_path / "abc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "DISTFLASHATTN: Distributed Memory-efficient Attention", fontsize=14.5)
    page.insert_text((72, 98), "for Long-context LLMs Training", fontsize=14.3)
    page.insert_text((72, 140), "Abstract", fontsize=12.0)
    doc.save(str(pdf))
    doc.close()

    (tmp_path / "abc.json").write_text(json.dumps({
        "id": "abc",
        "filename": "2310.03294.pdf",
        "title": "DISTFLASHATTN:Distributed Memory-efficientAttention",  # 旧规则算坏的那行
        "pages": 1,
        "uploaded_at": "2026-11-07T00:00:00+00:00",
    }), encoding="utf-8")

    reg = PaperRegistry(tmp_path)
    assert reg.get("abc").title == (
        "DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training"
    )
    # 修好之后要落盘，不然每次启动都得重算一遍
    assert json.loads((tmp_path / "abc.json").read_text(encoding="utf-8"))["title_v"] >= 1


def test_a_page_without_a_text_layer_is_marked_scanned(tmp_path):
    import pymupdf

    from app.api.registry import PaperRegistry

    blank = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(blank))
    doc.close()
    assert PaperRegistry._is_scanned(blank) is True

    typed = tmp_path / "typed.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A real title line with enough characters to count")
    doc.save(str(typed))
    doc.close()
    assert PaperRegistry._is_scanned(typed) is False


def test_a_scanned_title_is_read_by_ocr_and_the_glued_words_are_split(
    tmp_path, monkeypatch
):
    """扫描件没有文字层，标题只能靠 OCR 认。

    两个坑都在这条用例里：
    1. **不能按 size 挑**。OCR 的 size 是行高像素当量，实测邮箱那行比标题还大；
       稳定的判据是版面宽度 —— 标题是首页上半部分最宽的那行。
    2. **空格会被吃掉**（AttentionIsAllYouNeed），要按驼峰补回来，
       否则文献库里那行标题永远是一串连体字。
    """
    import pymupdf

    from app.api.registry import PaperRegistry
    from app.parsing import scan

    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()  # A4 空页：没有文本层，_title_from_layout 只能返回 None
    doc.save(str(blank))
    doc.close()

    def line(text, box, size=15.0):
        return scan.OcrLine(text, box, [scan.OcrWord(text, box)], size)

    fake = [
        line("AttentionIsAllYouNeed", (212, 100, 392, 116), 16.0),
        line("AshishVaswani*", (132, 184, 200, 198), 14.0),
        # 这行的 size 比标题还大 —— 按字号挑就会挑出邮箱
        line("nikip@google.com", (320, 205, 408, 224), 19.0),
    ]
    monkeypatch.setattr(scan, "ocr_enabled", lambda: True)
    monkeypatch.setattr(scan, "ocr_lines", lambda page, dpi=None: fake)

    assert PaperRegistry._title(blank, "1706.03762v1_scan.pdf") == "Attention Is All You Need"


def test_a_scanned_title_never_falls_back_to_the_filename_when_ocr_is_off(
    tmp_path, monkeypatch
):
    """OCR 引擎不可用时就老老实实用文件名，不许编一个出来。"""
    import pymupdf

    from app.api.registry import PaperRegistry
    from app.parsing import scan

    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(blank))
    doc.close()

    monkeypatch.setattr(scan, "ocr_enabled", lambda: False)
    assert PaperRegistry._title_from_ocr(blank) is None
    assert PaperRegistry._title(blank, "1706.03762v1_scan.pdf") == "1706.03762v1_scan"


def test_qa_tells_the_reader_when_the_paper_is_a_scan(client, paper_id, monkeypatch):
    """扫描件的引文是从 OCR 文本里摘的，接口要带上这个标记。

    不带的话，用户对着纸面核对 "we use d d, dode/h 64" 这种识别残渣，
    只会觉得系统抄错了 —— 其实它抄的正是自己索引里的原文。
    """
    from app.api.main import registry

    paper = registry.get(paper_id)
    assert client.post(
        f"/api/papers/{paper_id}/qa", json={"question": "attention heads", "mode": "retrieve"}
    ).json()["scanned"] is False

    monkeypatch.setattr(paper, "scanned", True)
    body = client.post(
        f"/api/papers/{paper_id}/qa", json={"question": "attention heads", "mode": "retrieve"}
    ).json()
    assert body["scanned"] is True


def test_whitespace_only_question_is_rejected_before_the_pipeline(client, paper_id):
    """只写空格的问题以前能过 min_length=1，整条链路白跑一遍再返回 200 + 空结论。

    实测过一次：`{"question": "   "}` 返回 200、claims/evidence 全空、耗时 833 ms ——
    用户看到的是一张"什么都没答"的界面，而不是"问题不能为空"。
    同样的口径也用在综述主题、框架 idea、Agent 任务与跨文献对比的问题上。
    """
    for body in ({"question": "   "}, {"question": ""}, {"question": chr(9) + chr(10)}):
        r = client.post(f"/api/papers/{paper_id}/qa", json=body)
        assert r.status_code == 422, r.text

    assert client.post("/api/review", json={"topic": "   "}).status_code == 422
    assert client.post("/api/copilot/outline", json={"idea": "  "}).status_code == 422
    assert client.post("/api/agent/plan", json={"task": " "}).status_code == 422
    assert client.post(
        "/api/papers/compare", json={"paper_ids": [paper_id, paper_id], "question": "  "}
    ).status_code == 422
