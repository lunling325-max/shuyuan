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


@router.post("/api/papers/{paper_id}/formulas/{index}/latex")
def formula_latex(paper_id: str, index: int) -> dict:
    """把一条公式图识别成 LaTeX。

    **按需触发**，不进入库流程：单条 0.7~1.4 秒（CPU），541 页的书可能几百条公式，
    放进入库会让上传卡成几分钟。结果按 `公式索引 → LaTeX` 落盘缓存，
    重开页面不再重算。

    模型没装时返回 503 并说清原因 —— 这是"这个能力没启用"，不是"识别失败"。
    """
    from app.parsing import formula_latex as ocr

    paper = get_paper(paper_id)
    formulas = paper.formulas.formulas
    if not (0 <= index < len(formulas)):
        raise HTTPException(404, f"该文献共 {len(formulas)} 个编号公式，不存在第 {index + 1} 个。")

    cache = ocr.load_cache(paper.path.parent, paper.id)
    key = str(index)
    if key in cache:
        return {"index": index, "latex": cache[key], "cached": True}

    if not ocr.available():
        raise HTTPException(
            503,
            "公式识别模型未就位：这台服务没有装上 OCR 模型（models/mfr），"
            "其余功能不受影响。",
        )

    formula = formulas[index]
    x0, y0, x1, y1 = formula.bbox
    png = render_figure(paper.path, formula.page, (x0 - 3, y0 - 4, x1 + 3, y1 + 4), dpi=200)
    if not png:
        raise HTTPException(404, "该公式无法渲染（PDF 可能已损坏）。")

    try:
        latex = ocr.recognise(png)
    except ocr.OcrUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 —— 单条识别失败不该 500
        raise HTTPException(422, f"识别失败：{type(exc).__name__}") from exc

    cache[key] = latex
    ocr.save_cache(paper.path.parent, paper.id, cache)
    return {"index": index, "latex": latex, "cached": False}
