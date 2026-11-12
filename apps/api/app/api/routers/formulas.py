"""公式路由：把论文里**有编号的**独立公式取出来。

为什么不塞进 `/api/papers/{id}`：公式要整篇走一遍版面（40 页上限），
而绝大多数请求（列表、问答、跳转）根本不看公式。和图表一样懒解析：
第一次请求才算，算完落盘。

三个字段是刻意分开的：

- `label`（式 (1)）—— 给人引用用的，作者写了编号我们才敢这么叫；
- `text` —— 版面文本，**有损**（上标会被摊平、根号读成乱码），
  接口里就叫 `text` 而不是 `latex`，免得到处有人以为能直接拿去排版；
- `thumb` —— 按 `bbox` 裁出来的图。要看公式长什么样，看这张，别看 `text`。

`note` 负责说清"为什么是 0 个"：这份 PDF 没有编号公式，和这份 PDF 解析失败，
是两件事。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import get_paper
from app.parsing.visuals import render_figure

router = APIRouter()


@router.get("/api/papers/{paper_id}/formulas")
def list_formulas(paper_id: str) -> dict:
    paper = get_paper(paper_id)
    report = paper.formulas
    formulas = [
        {
            "index": i,
            "page": f.page,
            "bbox": list(f.bbox),
            "number": f.number,
            "label": f.label,
            "text": f.text,
            "math_ratio": f.math_ratio,
            "thumb": f"/api/papers/{paper.id}/formulas/{i}.png",
        }
        for i, f in enumerate(report.formulas)
    ]
    return {
        "formulas": formulas,
        "count": len(formulas),
        "pages": sorted({f["page"] for f in formulas}),
        "note": report.note,
        "analyzed_pages": report.analyzed_pages,
        "cached": report.cached,
    }


@router.get("/api/papers/{paper_id}/formulas/{index}.png")
def formula_png(paper_id: str, index: int) -> Response:
    paper = get_paper(paper_id)
    formulas = paper.formulas.formulas
    if not (0 <= index < len(formulas)):
        raise HTTPException(404, f"该文献共 {len(formulas)} 个编号公式，不存在第 {index + 1} 个。")
    formula = formulas[index]
    # 四周留一点白：公式的包围盒只框字形，紧贴着裁会把根号顶到边上。
    # 200 dpi 不是随手写的：公式的字号只有正文一半，110 dpi 放到栏宽后
    # 上下标糊成一团（实机截图确认过）；而这些裁片本来就窄，多出的字节可以忽略。
    x0, y0, x1, y1 = formula.bbox
    png = render_figure(paper.path, formula.page, (x0 - 3, y0 - 4, x1 + 3, y1 + 4), dpi=200)
    if not png:
        raise HTTPException(404, "该公式无法渲染（PDF 可能已损坏）。")
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
