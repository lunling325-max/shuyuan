"""公式端点的 HTTP 层测试：四条端点约定各守一件事。

- `/formulas`：认出来的编号能直接给前端用（label、页码、包围盒、缩略图地址）；
- `/formulas/{i}.png`：必须是真正的 PNG 字节，不是一段 200 的 JSON；
- 越界索引：404 且说清共有几个，不能让前端拿到一张空白 `<img>`；
- 一篇**没有**编号公式的论文：`count` 是 0，而 `note` 必须说"没有找到"，
  不能让人误读成"解析失败"。

和 `test_visuals_api.py` 不一样的地方：这里不自己造 PDF。造一份能被认成
"有编号公式"的合成 PDF，需要数学字体（CMMI/CMSY/CMEX），而 pyMuPDF 的
内置字体里没有这些 —— 为了让判据通过去借 Symbol 凑数，测的就不是真实的版面了。
所以直接用 `pdf/` 里的真文件；缺文件时 conftest 会把整个模块标成跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"   # 式 (1)(2)(3)
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"  # 一条编号公式都没有


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # 必须在导入 app 之前设好：`deps.STORE_DIR` 是导入时求值的
    import os

    os.environ["PAPER_STORE_DIR"] = str(tmp_path_factory.mktemp("papers_formula"))
    from app.api.main import app

    return TestClient(app)


def _upload(client, path: Path) -> str:
    r = client.post("/api/papers", files={"file": (path.name, path.read_bytes(), "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture(scope="module")
def transformer_id(client) -> str:
    return _upload(client, PDF_V7)


@pytest.fixture(scope="module")
def bert_id(client) -> str:
    return _upload(client, PDF_BERT)


def test_formulas_endpoint_lists_numbered_equations(client, transformer_id):
    r = client.get(f"/api/papers/{transformer_id}/formulas")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert [f["number"] for f in body["formulas"]] == ["1", "2", "3"]
    assert body["pages"] == sorted({f["page"] for f in body["formulas"]})
    first = body["formulas"][0]
    assert first["label"] == "式 (1)"
    assert first["thumb"] == f"/api/papers/{transformer_id}/formulas/0.png"
    assert len(first["bbox"]) == 4
    assert first["bbox"][0] < first["bbox"][2] and first["bbox"][1] < first["bbox"][3]
    assert 0.0 < first["math_ratio"] <= 1.0
    # 文本层是有损的，但必须**有**东西：前端靠它做检索的提示
    assert "Attention" in first["text"]
    assert body["note"].startswith("带编号的公式 3 个")


def test_formula_png_endpoint_returns_png_bytes(client, transformer_id):
    r = client.get(f"/api/papers/{transformer_id}/formulas/0.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "max-age" in r.headers.get("cache-control", "")


def test_formula_index_out_of_range_is_404_with_a_count(client, transformer_id):
    r = client.get(f"/api/papers/{transformer_id}/formulas/3.png")
    assert r.status_code == 404
    assert "3" in r.json()["detail"]


def test_paper_without_numbered_equations_says_none_not_failed(client, bert_id):
    r = client.get(f"/api/papers/{bert_id}/formulas")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["formulas"] == []
    assert "没有找到带编号的公式" in body["note"]


def test_unknown_paper_is_404(client):
    assert client.get("/api/papers/nope/formulas").status_code == 404
    assert client.get("/api/papers/nope/formulas/0.png").status_code == 404
