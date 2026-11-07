"""量一下扫描件 OCR 的还原质量：识别得对不对，空格接得回来没有。

为什么要单独量：OCR 出来"看着像英文"不代表能用。下游是 BM25 检索和逐字引用，
所以真正要盯的是两件事 ——
1. **字符识别准确率**：认错的字会让引用永远对不上原文；
2. **词边界还原率**：PP-OCR 会吃掉词间空格（"Thedominantsequence"），
   粘起来的词在索引里根本不存在，检索和引用一起失效。

真值不用另找：`pdf/1706.03762v1.pdf` 有文字层，把它当标准答案，
和 `pdf/1706.03762v1_scan.pdf`（同一份被渲染成像素）逐词比。

用法（项目根目录）：
    PYTHONUTF8=1 uv run --directory apps/api python ../../scripts/eval_ocr.py 3
"""

from __future__ import annotations

import difflib
import re
import sys
import time
from pathlib import Path

import pymupdf

from app.parsing import scan

ROOT = Path(__file__).resolve().parents[1]
TRUE_PDF = ROOT / "pdf" / "1706.03762v1.pdf"
SCAN_PDF = ROOT / "pdf" / "1706.03762v1_scan.pdf"

WORD = re.compile(r"[A-Za-z0-9]+")


def words_of(text: str) -> list[str]:
    return WORD.findall(text.lower())


def main() -> int:
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else scan.DEFAULT_DPI
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    true_doc = pymupdf.open(str(TRUE_PDF))
    scan_doc = pymupdf.open(str(SCAN_PDF))

    total_true = matched = merged = wrong = 0
    ratios: list[float] = []
    f1s: list[float] = []
    spent = 0.0

    for page_no in range(start, start + pages):
        truth_text = true_doc[page_no - 1].get_text("text")
        truth_words = words_of(truth_text)
        truth_set = set(truth_words)

        t0 = time.perf_counter()
        lines = scan.ocr_lines(scan_doc[page_no - 1], dpi=dpi)
        spent += time.perf_counter() - t0

        got = [w.text for line in lines for w in line.words]
        got_words = [w for raw in got for w in words_of(raw)]

        ok = sum(1 for w in got_words if w in truth_set)
        # 粘连词：不在真值词表里，但能拆成两个以上真值词 → 空格没接回来
        glue = 0
        for w in got_words:
            if w in truth_set or len(w) < 6:
                continue
            for cut in range(3, len(w) - 2):
                if w[:cut] in truth_set and w[cut:] in truth_set:
                    glue += 1
                    break
        merged += glue
        matched += ok
        total_true += len(truth_words)
        wrong += len(got_words) - ok - glue
        ratios.append(
            difflib.SequenceMatcher(
                None, "".join(truth_words), "".join(got_words), autojunk=False
            ).ratio()
        )
        # 词级 F1：按词序列对齐算，丢一个词和错一个词都扣分，比字符相似度更贴近检索。
        f1 = difflib.SequenceMatcher(None, truth_words, got_words, autojunk=False).ratio()
        f1s.append(f1)
        print(
            f"  第 {page_no:2d} 页 · {len(lines):3d} 行 · {len(got_words):4d} 词 · "
            f"命中 {ok / max(len(got_words), 1):5.1%} · 粘连 {glue:3d} · "
            f"词级 F1 {f1:.3f} · 字符相似度 {ratios[-1]:.3f}"
        )

    print()
    print(f"识别 {total_true} 个标准词，OCR 给出 {matched + merged + wrong} 个词")
    print(f"  完全命中      {matched / max(matched + merged + wrong, 1):6.1%}")
    # 召回：真值里有百分之多少的词在 OCR 结果里出现过。
    # 只看"命中率"会被漏字骗过去 —— 少认一半词、剩下的全认对，命中率反而更高。
    print(f"  召回          {matched / max(total_true, 1):6.1%}")
    print(f"  粘连（丢空格）{merged / max(matched + merged + wrong, 1):6.1%}")
    print(f"  认错          {wrong / max(matched + merged + wrong, 1):6.1%}")
    print(f"  词级 F1       {sum(f1s) / len(f1s):.3f}")
    print(f"  字符相似度    {sum(ratios) / len(ratios):.3f}")
    if spent / pages < 1.0:
        print(f"  平均 {spent / pages:.2f}s/页 —— 全是缓存命中，这个时间不算数")
    else:
        print(f"  平均 {spent / pages:.1f}s/页（dpi={dpi}）")

    true_doc.close()
    scan_doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
