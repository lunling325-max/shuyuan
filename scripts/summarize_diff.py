"""把句级差异块细化到词级，并交给 LLM 生成版本修订摘要。

输入: version_diff.py --json 产出的差异文件
输出: 结构化变更列表 + LLM 生成的中文修订摘要

用法: python scripts/summarize_diff.py .cache/v1_v7_diff.json [--llm]
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import urllib.request

WORD = re.compile(r"\w+|[^\w\s]")


def word_diff(old: str, new: str) -> tuple[str, str, float]:
    """返回 (删掉的部分, 新增的部分, 相似度)。"""
    a, b = WORD.findall(old), WORD.findall(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    dels, inss = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            dels.append(" ".join(a[i1:i2]))
        if tag in ("insert", "replace"):
            inss.append(" ".join(b[j1:j2]))
    return " ".join(dels).strip(), " ".join(inss).strip(), sm.ratio()


def main() -> None:
    changes = json.load(open(sys.argv[1], encoding="utf-8"))

    refined = []
    noise = 0
    for c in changes:
        if c["type"] == "replace" and len(c["old"]) == 1 and len(c["new"]) == 1:
            d, i, ratio = word_diff(c["old"][0], c["new"][0])
            if ratio >= 0.97 or (not d and not i):
                noise += 1
                continue
            refined.append(
                {"page": c["old_pages"][0], "kind": "改", "old": d, "new": i, "ratio": round(ratio, 3)}
            )
        elif c["type"] == "insert":
            refined.append(
                {"page": c["new_pages"][0] if c["new_pages"] else "?", "kind": "增",
                 "old": "", "new": " ".join(c["new"])[:200], "ratio": 0.0}
            )
        elif c["type"] == "delete":
            refined.append(
                {"page": c["old_pages"][0] if c["old_pages"] else "?", "kind": "删",
                 "old": " ".join(c["old"])[:200], "new": "", "ratio": 0.0}
            )
        else:
            refined.append(
                {"page": c["old_pages"][0] if c["old_pages"] else "?", "kind": "块改写",
                 "old": " ".join(c["old"])[:200], "new": " ".join(c["new"])[:200], "ratio": 0.0}
            )

    print(f"原始变更块 {len(changes)} → 有效变更 {len(refined)}（过滤对齐噪声 {noise} 条）\n")
    for r in refined[:15]:
        print(f"  [p{r['page']:>2}·{r['kind']}] sim={r['ratio']}")
        if r["old"]:
            print(f"     - {r['old'][:140]}")
        if r["new"]:
            print(f"     + {r['new'][:140]}")

    out = ".cache/v1_v7_refined.json"
    json.dump(refined, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")

    if "--llm" not in sys.argv:
        return

    digest = "\n".join(
        f"[p{r['page']}][{r['kind']}] 旧:{r['old'][:150]} ||| 新:{r['new'][:150]}" for r in refined[:40]
    )
    prompt = (
        "下面是同一篇论文 arXiv v1 与 v7 两个版本的逐条差异（[页码][增/删/改] 旧:... ||| 新:...）。\n"
        "请用中文输出一段‘版本修订摘要’，要求：\n"
        "1. 归纳成 3-6 条实质性修订，按重要性排序，不要逐条罗列；\n"
        "2. 每条标注涉及的页码；\n"
        "3. 只写差异中确实体现的内容，不要猜测、不要编造；\n"
        "4. 纯文本，不要 markdown 标题。\n\n差异：\n" + digest
    )

    key = os.environ["DEEPSEEK_API_KEY"]
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    body = json.dumps(
        {
            "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    summary = data["choices"][0]["message"]["content"]
    open(".cache/v1_v7_summary.txt", "w", encoding="utf-8").write(summary)
    print("\n===== LLM 版本修订摘要 =====")
    print(summary)
    print(f"\n(usage: {data.get('usage')})")


if __name__ == "__main__":
    main()
