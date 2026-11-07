"""图表路由：把论文里的表格与插图单独取出来。

为什么单独开一条路由，而不是塞进 `/api/papers/{id}`：解析要十几秒
（`find_tables()` 约 0.29 秒/页，40 页上限就是 12 秒）。挂在文献列表上，
每刷新一次列表都要等它，而大多数时候用户根本不看图表。

三条端点的分工：

- `/tables` 与 `/figures` 是**懒解析**：第一次请求才算，算完落盘，
  之后连进程重启都不用再算。
- `/figures/{index}.png` 按需渲染缩略图。为什么不返回嵌入的原图：
  原图可能是 CMYK、带遮罩、4000px 宽，浏览器要么显示不出来要么白白下载几 MB。

扫描件在这条路上是**空结果 + 说明**，不是错误：`find_tables()` 靠文字层定位，
扫描页没有文字层，它一律返回空。与其猜一张网格出来，不如说清"提不出来"。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import get_paper
from app.parsing.visuals import render_figure

router = APIRouter()


def _table_payload(table) -> dict:
    return {
        "page": table.page,
        "bbox": list(table.bbox),
        "label": table.label,
        "caption": table.caption,
        "rows": [list(row) for row in table.rows],
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "markdown": table.markdown(),
    }


@router.get("/api/papers/{paper_id}/tables")
def list_tables(paper_id: str) -> dict:
    paper = get_paper(paper_id)
    report = paper.visuals
    tables = [_table_payload(t) for t in report.tables]
    pages = sorted({t["page"] for t in tables})
    return {
        "tables": tables,
        "count": len(tables),
        "pages": pages,
        "note": report.note,
        "analyzed_pages": report.analyzed_pages,
        "cached": report.cached,
    }


@router.get("/api/papers/{paper_id}/figures")
def list_figures(paper_id: str) -> dict:
    paper = get_paper(paper_id)
    report = paper.visuals
    figures = [
        {
            "index": i,
            "page": f.page,
            "bbox": list(f.bbox),
            "label": f.label,
            "caption": f.caption,
            "width": f.width,
            "height": f.height,
            "thumb": f"/api/papers/{paper.id}/figures/{i}.png",
        }
        for i, f in enumerate(report.figures)
    ]
    return {
        "figures": figures,
        "count": len(figures),
        "note": report.note,
        "analyzed_pages": report.analyzed_pages,
        "cached": report.cached,
    }


@router.get("/api/papers/{paper_id}/figures/{index}.png")
def figure_png(paper_id: str, index: int) -> Response:
    paper = get_paper(paper_id)
    figures = paper.visuals.figures
    if not (0 <= index < len(figures)):
        raise HTTPException(404, f"该文献共 {len(figures)} 幅插图，不存在第 {index + 1} 幅。")
    figure = figures[index]
    png = render_figure(paper.path, figure.page, figure.bbox)
    if not png:
        raise HTTPException(404, "该插图无法渲染（PDF 可能已损坏）。")
    # 图和文献一样不会变，缓存一天足够；浏览器来回切标签时不用重新渲染
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
