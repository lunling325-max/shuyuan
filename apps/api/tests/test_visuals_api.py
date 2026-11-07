"""图表端点的 HTTP 层测试：三条端点各守一件事。

- `/tables` 与 `/figures`：抽出来的内容要能直接被前端渲染（行、列、标题）；
- `/figures/{i}.png`：必须是真正的 PNG 字节，而不是一段 200 的 JSON；
- 越界索引：404 且说清有几幅图，不能让前端拿到一张空白 `<img>`。
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # 必须在导入 app 之前设好：`deps.STORE_DIR` 是导入时求值的
    os.environ.setdefault("PAPER_STORE_DIR", str(tmp_path_factory.mktemp("papers_visual")))
    from app.api.main import app

    return TestClient(app)


def _sample_pdf(path: Path) -> Path:
    """一页里放一张 3×3 的表和一幅带图注的插图。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Table 1: BLEU scores of our models.", fontsize=9)
    rows = [("Model", "EN-DE", "EN-FR"), ("GNMT", "24.6", "39.92"), ("Ours", "28.4", "41.8")]
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x, y = 72 + c * 120, 100 + r * 22
            page.draw_rect(pymupdf.Rect(x, y, x + 120, y + 22), color=(0, 0, 0), width=0.6)
            page.insert_text((x + 4, y + 15), text, fontsize=9)

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
    pix.set_rect(pix.irect, (30, 120, 200))
    page.insert_image(pymupdf.Rect(250, 300, 450, 433), pixmap=pix)
    page.insert_text((72, 450), "Figure 1: Our model, drawn to scale.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def paper_id(client, tmp_path_factory) -> str:
    path = _sample_pdf(tmp_path_factory.mktemp("pdf") / "visuals.pdf")
    r = client.post("/api/papers", files={"file": (path.name, path.read_bytes(), "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_tables_endpoint_returns_rows_and_markdown(client, paper_id):
    r = client.get(f"/api/papers/{paper_id}/tables")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    table = body["tables"][0]
    assert table["rows"][2] == ["Ours", "28.4", "41.8"]
    assert table["markdown"].splitlines()[1] == "|---|---|---|"
    assert table["label"] == "Table 1"


def test_figures_endpoint_points_at_a_thumbnail(client, paper_id):
    r = client.get(f"/api/papers/{paper_id}/figures")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    figure = body["figures"][0]
    assert figure["label"] == "Figure 1"
    assert figure["thumb"] == f"/api/papers/{paper_id}/figures/0.png"


def test_thumbnail_endpoint_returns_png_bytes(client, paper_id):
    r = client.get(f"/api/papers/{paper_id}/figures/0.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "max-age" in r.headers.get("cache-control", "")


def test_thumbnail_index_out_of_range_is_404_with_a_count(client, paper_id):
    r = client.get(f"/api/papers/{paper_id}/figures/7.png")
    assert r.status_code == 404
    assert "1" in r.json()["detail"]


def test_unknown_paper_is_404(client):
    assert client.get("/api/papers/nope/tables").status_code == 404
    assert client.get("/api/papers/nope/figures").status_code == 404
    assert client.get("/api/papers/nope/figures/0.png").status_code == 404
