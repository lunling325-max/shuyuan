"""首页版式的底层读法：把 PyMuPDF 的 dict 输出还原成"一行一行带字号的字"。

标题识别（`registry._title_from_layout`）和首屏元信息识别（`frontmatter`）
都要同一套东西：怎么把一行拼成字符串、怎么认出并剔掉 arXiv 的投稿戳。
这两件事以前只写在注册表里，加个作者/机构识别就要么复制一份、要么就地改注册表 ——
复制一份意味着两边以后会慢慢走岔（E-038 刚教过一次：同一段逻辑写两遍，
改一处漏一处，症状是"部分引用校验失败"，很难查）。所以提到这里共用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# arXiv 左侧那行旋转的投稿戳，字号 20 比标题还大，必须先剔掉。
# 它自带版本号和日期，两版必然不同 —— 留着会让每份论文的标题都变成
# `arXiv:1706.03762v7 [cs.CL] …`，也会被误当成作者块里的一行。
STAMP = re.compile(r"^arxiv:\S+\s*\[", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Line:
    """首页的一行：字号、包围盒、文本。坐标单位是 PDF 点（1/72 英寸）。"""

    size: float
    x0: float
    y0: float
    y1: float
    text: str


def line_of(line: dict) -> tuple[float, str] | None:
    """把 PyMuPDF 的一行还原成 (字号, 文本)。

    **行内拼 span 不加分隔符**：排版器省掉的空格本来就不在 span 文本里，
    补一个空格会造出 `Memory- efficient Attention` 这种原文没有的东西。
    字号取行内最大的那个 span —— 一行里混着主标题和小型大写字母是常态。

    整行是 arXiv 投稿戳时返回 None。
    """
    parts: list[str] = []
    size = 0.0
    for span in line.get("spans", []):
        text = span.get("text") or ""
        if not text:
            continue
        if STAMP.match(" ".join(text.split())):
            return None
        parts.append(text)
        size = max(size, round(float(span["size"]), 1))
    text = " ".join("".join(parts).split())
    return (size, text) if text else None


def page_lines(path: str | Path, page_no: int = 0) -> list[Line]:
    """某一页的全部文本行，按 PDF 给出的顺序（多数排版器按纵序）。"""
    doc = pymupdf.open(str(path))
    try:
        if page_no >= doc.page_count:
            return []
        page = doc[page_no]
        out: list[Line] = []
        for block in page.get_text("dict").get("blocks", []):
            for raw in block.get("lines", []):
                got = line_of(raw)
                if got is None:
                    continue
                size, text = got
                box = raw["bbox"]
                out.append(Line(size, float(box[0]), float(box[1]), float(box[3]), text))
        return out
    finally:
        doc.close()
