"""可行性探针：验证「基于内容语义去重」在 arXiv v1 vs v7 上是否成立。

输出三件事：
1. 整体相似度（5-gram shingle Jaccard）—— 决定去重阈值定多少
2. 逐页相似度 —— 决定"改了哪几页"
3. 差异片段 —— 决定版本 diff 摘要喂给 LLM 的材料

用法: python scripts/dedup_probe.py <旧版.pdf> <新版.pdf>
"""
import re
import sys

import pymupdf


def page_texts(path: str) -> list[str]:
    doc = pymupdf.open(path)
    return [pg.get_text() for pg in doc]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(text))


def shingles(tokens: list[str], k: int = 5) -> set[tuple[str, ...]]:
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    old_path, new_path = sys.argv[1], sys.argv[2]
    old_pages, new_pages = page_texts(old_path), page_texts(new_path)

    old_all, new_all = " ".join(old_pages), " ".join(new_pages)
    ow, nw = words(old_all), words(new_all)

    doc_sim = jaccard(shingles(ow), shingles(nw))

    print("=" * 68)
    print(f"旧版: {old_path}  pages={len(old_pages)}  words={len(ow)}")
    print(f"新版: {new_path}  pages={len(new_pages)}  words={len(nw)}")
    print("=" * 68)
    print(f"整体 5-gram Jaccard 相似度 : {doc_sim:.4f}")
    print(f"字符级相似度(参考)         : {jaccard(set(norm(old_all)), set(norm(new_all))):.4f}")
    print()

    print("逐页相似度（<0.90 视为有实质改动）:")
    changed = []
    for i in range(max(len(old_pages), len(new_pages))):
        o = old_pages[i] if i < len(old_pages) else ""
        n = new_pages[i] if i < len(new_pages) else ""
        sim = jaccard(shingles(words(o)), shingles(words(n)))
        flag = "" if sim >= 0.90 else "  <-- 改动"
        if sim < 0.90:
            changed.append((i + 1, sim))
        print(f"  p{i + 1:>2}  sim={sim:.3f}{flag}")

    print()
    print(f"改动页: {[p for p, _ in changed] or '无'}")
    print()
    print("判定建议:")
    if doc_sim >= 0.95:
        print(f"  → 相似度 {doc_sim:.3f} 极高，判为同一篇的不同版本，阈值可定 0.90~0.95")
    elif doc_sim >= 0.80:
        print(f"  → 相似度 {doc_sim:.3f} 偏高，需配合 arXiv ID / 标题归一化再判")
    else:
        print(f"  → 相似度 {doc_sim:.3f} 偏低，纯文本相似度不足以判重，必须引入结构化特征")


if __name__ == "__main__":
    main()
