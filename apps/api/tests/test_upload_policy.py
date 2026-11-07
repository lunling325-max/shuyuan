"""去重策略（warn / keep / replace）与批量上传的测试。

这是官方用例 3.1 的验收点：系统不仅要**认出**"库里已经有同一个东西"，
还要把决定权交给用户 —— "覆盖"还是"保留"，并且说得出哪一版更新。

每个用例都换一个全新的注册表：`replace` 会真的删文件，跑在共享注册表上
会把别的用例刚传进去的论文删掉，制造一堆互相牵连的失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

NOT_A_PDF = b"MZ\x90\x00 not a pdf"


@pytest.fixture()
def env(tmp_path, monkeypatch) -> tuple[TestClient, object]:
    """干净的资料库 + main 的路由。返回 (client, registry) 方便直接查库。"""
    import app.api.main as main
    from app.api.registry import PaperRegistry

    reg = PaperRegistry(tmp_path / "papers")
    monkeypatch.setattr(main, "registry", reg)
    return TestClient(main.app), reg


def _upload(client: TestClient, path: Path, **params):
    return client.post(
        "/api/papers",
        params=params,
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )


def _ids(reg) -> set[str]:
    return {p.id for p in reg.all()}


def test_default_policy_keeps_both_versions_and_flags_them(env):
    """默认 warn：两版都留下。默认策略必须是非破坏性的 —— 自动覆盖判错时，
    删掉的是用户手里唯一的那份原件。"""
    client, reg = env
    old = _upload(client, PDF_V1).json()
    r = _upload(client, PDF_V7)

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["duplicate_of"] == old["id"]
    assert body["relation"] == "new_version"
    assert _ids(reg) == {old["id"], body["id"]}


def test_keep_policy_rejects_and_stores_nothing(env):
    """on_duplicate=keep：库里那篇原样不动，新传的那份不能留在库里。"""
    client, reg = env
    old = _upload(client, PDF_V1).json()

    r = _upload(client, PDF_V7, on_duplicate="keep")

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["existing"]["id"] == old["id"]
    assert detail["uploaded"]["relation"] == "new_version"
    assert _ids(reg) == {old["id"]}, "被拒的那份必须从库里撤掉，不能只是不回包"


def test_replace_policy_swaps_the_older_version(env):
    client, reg = env
    old = _upload(client, PDF_V1).json()

    r = _upload(client, PDF_V7, on_duplicate="replace")

    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    assert new_id != old["id"]
    assert _ids(reg) == {new_id}, "覆盖之后库里只能剩新那版"


def test_reuploading_identical_bytes_never_deletes_the_stored_copy(env):
    """重传同一个文件必须是无操作。

    踩过的坑：`registry.add` 对内容哈希相同的输入直接返回**库里那个对象**，
    如果这时照常执行 keep 策略里的 `remove(paper.id)`，删掉的就是用户
    早就传好的那篇 —— 一次手滑重传，文献库少一篇。
    """
    client, reg = env
    first = _upload(client, PDF_V7).json()

    r = _upload(client, PDF_V7, on_duplicate="keep")

    assert r.status_code == 201, r.text
    assert r.json()["id"] == first["id"]
    assert first["id"] in _ids(reg)


def test_reuploading_identical_bytes_is_not_reported_as_a_duplicate(env):
    """同一个文件重传，不该回一句"发现相似论文，是不是要覆盖"去吓用户。"""
    client, _reg = env
    _upload(client, PDF_V7)

    body = _upload(client, PDF_V7).json()

    assert body["duplicate_of"] is None and body["relation"] is None


@pytest.fixture()
def newer_variant(tmp_path) -> Path:
    """正文一字不改、只把 PDF 生成日期改到未来的"更新版本"。

    官方数据集里只有 v1 和 v7 两份，没有第三份可比的文件；从网上下载新的又不行
    （任何下载都要先过来源评估/哈希/杀毒四关）。所以就地造一份：
    内容完全一样、元数据里的 creationDate 更晚 —— 正好用来检验"谁更新"
    看的是日期，而不是文件名里那个 v1/v7。
    """
    import pymupdf

    doc = pymupdf.open(str(PDF_V7))
    doc.set_metadata({**(doc.metadata or {}), "creationDate": "D:20990101000000Z"})
    out = tmp_path / "v7_newer.pdf"
    doc.save(str(out))
    doc.close()
    return out


def test_recommendation_says_the_newer_version_should_win(env, newer_variant):
    client, _reg = env
    _upload(client, PDF_V7)  # 库里是 2017 那版

    body = _upload(client, newer_variant).json()

    assert body["recommendation"] == "replace_existing"


def test_recommendation_says_keep_existing_when_upload_is_older(env, newer_variant):
    client, _reg = env
    _upload(client, newer_variant)  # 库里已经是最新的

    body = _upload(client, PDF_V7).json()

    assert body["recommendation"] == "keep_existing", "库中已有较新版本，应建议保留"


def test_unknown_policy_is_a_client_error(env):
    client, _reg = env

    r = _upload(client, PDF_V7, on_duplicate="覆盖")

    assert r.status_code == 400
    assert "on_duplicate" in r.json()["detail"]


# ---------------------------------------------------------------- 批量上传


def _batch(client: TestClient, paths: list[Path], **params):
    files = [
        ("files", (p.name, p.read_bytes(), "application/pdf")) for p in paths
    ]
    return client.post("/api/papers/batch", params=params, files=files)


def test_batch_upload_reports_every_file_separately(env):
    """一个坏文件不能拖垮另外几个：部分成功 + 逐条原因，才是批量该有的语义。"""
    client, reg = env
    bad = Path(PDF_V7).with_name("坏文件.pdf")

    files = [
        ("files", ("1706.03762v7.pdf", PDF_V7.read_bytes(), "application/pdf")),
        ("files", ("1810.04805_BERT.pdf", PDF_BERT.read_bytes(), "application/pdf")),
        ("files", (bad.name, NOT_A_PDF, "application/pdf")),
    ]
    r = client.post("/api/papers/batch", files=files)

    assert r.status_code == 201, r.text
    items = r.json()
    assert [it["ok"] for it in items] == [True, True, False]
    assert items[2]["error"]
    assert len(_ids(reg)) == 2, "坏文件不该入库，好文件不该被连坐"


def test_batch_respects_keep_policy(env):
    client, reg = env
    old = _upload(client, PDF_V1).json()

    r = _batch(client, [PDF_V7], on_duplicate="keep")

    assert r.status_code == 201, r.text
    item = r.json()[0]
    assert item["ok"] is False and item["skipped"] is True
    assert _ids(reg) == {old["id"]}


def test_batch_rejects_too_many_files_before_parsing(env):
    """数量校验在读文件之前就要拦下，别让 21 个大 PDF 先在内存里摊开。"""
    client, _reg = env
    files = [
        ("files", (f"x{i}.pdf", NOT_A_PDF, "application/pdf")) for i in range(21)
    ]

    r = client.post("/api/papers/batch", files=files)

    assert r.status_code == 413
    assert "20" in r.json()["detail"]


@pytest.fixture()
def v1_reexported(tmp_path) -> Path:
    """扫描件 v1，但 PDF 导出日期被推到 2099。

    文件名保留了 arXiv 号：扫描件首页没有文字层，arXiv 号只能从文件名认，
    改名成 "v1_reexported.pdf" 会让它拿不到 arXiv 号，比新旧就退回日期了 ——
    那样测的就不是这里要测的东西。
    """
    import pymupdf

    doc = pymupdf.open(str(PDF_V1))
    doc.set_metadata({**(doc.metadata or {}), "creationDate": "D:20990101000000Z"})
    out = tmp_path / "1706.03762v1_reexported.pdf"
    doc.save(str(out))
    doc.close()
    return out


def test_version_number_beats_pdf_export_date(env, v1_reexported):
    """v7 比 v1 新 —— 哪怕 v1 那份是后来重新导出的。

    `created_at` 的本义是"这份 PDF 什么时候导出"，不是"这篇论文什么时候发版"。
    只看日期的话，把 2017 年的 v1 在 2099 年重导一遍，系统就会建议用 v1 顶掉
    v7；而版本谱系页对同一对文件写的是"按 arXiv 版本号排序"。两处口径不一致，
    用户看到的就是自相矛盾的建议 —— 而且照做会删掉真正的新版。
    """
    client, _reg = env
    _upload(client, v1_reexported)

    body = _upload(client, PDF_V7).json()

    assert body["relation"] == "new_version"
    assert body["recommendation"] == "replace_existing", "v7 才是新版，日期更早不算旧"


def test_no_recommendation_when_only_upload_time_differs():
    """两份都取不到版本号和生成日期：不许凭"谁先传的"建议删任何一份。"""
    from types import SimpleNamespace

    from app.api.deps import recommend_version

    def paper(uploaded_at: str):
        return SimpleNamespace(
            arxiv_id="", arxiv_version=0, created_at="", uploaded_at=uploaded_at
        )

    old = paper("2020-01-01T00:00:00+00:00")
    new = paper("2024-01-01T00:00:00+00:00")

    assert recommend_version(new, old) == ("", "")
    assert recommend_version(old, new) == ("", "")


def test_recommendation_says_what_it_was_based_on(env, v1_reexported):
    """建议后面必须跟着依据：凭版本号判的和凭日期判的，界面文案不一样。"""
    client, _reg = env
    _upload(client, v1_reexported)

    body = _upload(client, PDF_V7).json()

    assert body["recommendation"] == "replace_existing"
    assert body["recommendation_by"] == "arxiv_version"
