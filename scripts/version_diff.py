"""版本差异检测原型：v1 -> v7 改了哪些段落。

流程：PDF → 归一化（NFKC + 断词修复）→ 按页提取 → 句级切分
      → difflib.SequenceMatcher 对齐 → 输出增/删/改块及其页码

用法: python scripts/version_diff.py <旧版.pdf> <新版.pdf> [--json out.json]
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import unicodedata

import pymupdf

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[0-9])")
MIN_SENT_CHARS = 25


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_sentences(path: str) -> list[dict]:
    """返回 [{'page': n, 'text': s}, ...]，保持阅读顺序。"""
    doc = pymupdf.open(path)
    out: list[dict] = []
    for pno, page in enumerate(doc, start=1):
        text = normalize(page.get_text())
        for s in SENT_SPLIT.split(text):
            s = s.strip()
            if len(s) >= MIN_SENT_CHARS:
                out.append({"page": pno, "text": s})
    return out


def diff(old: list[dict], new: list[dict]) -> list[dict]:
    sm = difflib.SequenceMatcher(
        None, [s["text"] for s in old], [s["text"] for s in new], autojunk=False
    )
    changes: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changes.append(
            {
                "type": tag,  # replace | delete | insert
                "old_pages": sorted({s["page"] for s in old[i1:i2]}),
                "new_pages": sorted({s["page"] for s in new[j1:j2]}),
                "old": [s["text"] for s in old[i1:i2]],
                "new": [s["text"] for s in new[j1:j2]],
            }
        )
    return changes


def main() -> None:
    old_path, new_path = sys.argv[1], sys.argv[2]
    old, new = load_sentences(old_path), load_sentences(new_path)
    changes = diff(old, new)

    n_replace = sum(1 for c in changes if c["type"] == "replace")
    n_delete = sum(1 for c in changes if c["type"] == "delete")
    n_insert = sum(1 for c in changes if c["type"] == "insert")

    print(f"v1 句子数={len(old)}  v7 句子数={len(new)}")
    print(f"变更块={len(changes)}  (改写 {n_replace} / 删除 {n_delete} / 新增 {n_insert})")
    touched = sorted({*[p for c in changes for p in c["old_pages"]]})
    print(f"受影响的旧版页码: {touched}")

    print("\n前 5 个变更块:")
    for c in changes[:5]:
        print(f"\n  [{c['type']}] v1 p{c['old_pages']} -> v7 p{c['new_pages']}")
        if c["old"]:
            print(f"    - {c['old'][0][:160]}")
        if c["new"]:
            print(f"    + {c['new'][0][:160]}")

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(changes, f, ensure_ascii=False, indent=2)
        print(f"\n完整差异已写入 {out}")


if __name__ == "__main__":
    main()
