"""诊断二：连字/断词归一化能救回多少相似度。

假设：v1 的 5-gram 跌到 0.68，主因是 U+FB00-FB06 连字字符 + 行尾连字符，
而非内容真实变化。归一化后相似度应显著回升。

用法: python scripts/normalize_probe.py <a.pdf> <b.pdf>
"""
import re
import sys
import unicodedata
from collections import Counter

import pymupdf

LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"


def raw_text(path: str) -> str:
    doc = pymupdf.open(path)
    return "\n".join(pg.get_text() for pg in doc)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)      # ﬁ -> fi, ﬀ -> ff ...
    text = re.sub(r"-\s*\n\s*", "", text)           # 行尾断词: signi-\ncantly -> significantly
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


def grams(ts: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(ts[i : i + k]) for i in range(len(ts) - k + 1)}


def jaccard(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if (x or y) else 1.0


def main() -> None:
    ta, tb = raw_text(sys.argv[1]), raw_text(sys.argv[2])

    print("连字字符出现次数（U+FB00–FB06）:")
    for name, t in (("v1", ta), ("v7", tb)):
        cnt = Counter(c for c in t if c in LIGATURES)
        print(f"  {name}: {sum(cnt.values()):>4}  明细={dict(cnt) or '无'}")

    print("\n归一化前后对比（5-gram Jaccard）:")
    for label, fn in (("原始", lambda s: s), ("NFKC+断词修复", normalize)):
        a, b = tokens(fn(ta)), tokens(fn(tb))
        print(f"  {label:<16} k=1: {jaccard(grams(a,1),grams(b,1)):.4f}   "
              f"k=5: {jaccard(grams(a,5),grams(b,5)):.4f}   tokens: {len(a)}/{len(b)}")

    a, b = tokens(normalize(ta)), tokens(normalize(tb))
    ca, cb = Counter(a), Counter(b)
    print(f"\n归一化后词袋重合: {sum((ca & cb).values())/sum((ca | cb).values()):.4f}")


if __name__ == "__main__":
    main()
