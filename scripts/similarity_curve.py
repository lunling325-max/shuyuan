"""诊断：v1 vs v7 的相似度为何随 n-gram 阶数崩塌。

若 unigram 高而 5-gram 低 → 是提取顺序/换行问题，不是内容变化。
用法: python scripts/similarity_curve.py <a.pdf> <b.pdf>
"""
import re
import sys
from collections import Counter

import pymupdf


def tokens(path: str) -> list[str]:
    doc = pymupdf.open(path)
    return re.findall(r"[a-z0-9]+", " ".join(pg.get_text() for pg in doc).lower())


def grams(ts: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(ts[i : i + k]) for i in range(len(ts) - k + 1)}


def jaccard(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if (x or y) else 1.0


def main() -> None:
    a, b = tokens(sys.argv[1]), tokens(sys.argv[2])
    print(f"v1 tokens={len(a)}  v7 tokens={len(b)}")
    print("\nn-gram Jaccard 曲线:")
    for k in (1, 2, 3, 5, 8):
        print(f"  k={k}: {jaccard(grams(a, k), grams(b, k)):.4f}")

    ca, cb = Counter(a), Counter(b)
    bag = sum((ca & cb).values()) / sum((ca | cb).values())
    print(f"\n词袋重合（完全忽略顺序）: {bag:.4f}")

    diff = sorted(((ca[w] - cb[w], w) for w in set(ca) | set(cb)), reverse=True)
    print("\nv1 独有高频词:", [f"{w}(+{d})" for d, w in diff[:6]])
    print("v7 独有高频词:", [f"{w}(+{-d})" for d, w in diff[-6:]])


if __name__ == "__main__":
    main()
