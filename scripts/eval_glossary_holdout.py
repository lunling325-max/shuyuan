"""离线术语表的**留出集**检验：换论文、换题，看它是不是只对调参那 18 题有效。

为什么需要这个脚本
------------------
`--offline` 在 `eval_retrieval.py` 的 18 题上把 Hit@3 从 61.1% 拉到 94.4%。
但那 18 题全是 Transformer 论文的题，而术语表也正是照着这类论文的词汇写的 ——
**同一批题既是开发集又是测试集，数字好看不说明问题**。

这里换三件事再测一遍：
1. **换论文**：BERT、GPT-3、FSA（都不是 Transformer 那篇）；
2. **换题型**：`scripts/eval_questions.json` 里的中文题，没有一道参与过术语表的编写；
3. **换指标**：不要求命中小节编号（换论文后编号体系不同），
   只要求**答案关键词出现在前 k 个候选块里** —— 这是检索层的下限，
   答不出来等于连材料都没找到。

用法：
    uv run python ../../scripts/eval_glossary_holdout.py
    ... --k 6 --no-expand   # 对照：完全不扩写
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

PAPERS: dict[str, Path] = {
    "1706.03762": ROOT / "pdf" / "1706.03762v7.pdf",
    "1810.04805": ROOT / "pdf" / "1810.04805_BERT.pdf",
    "2005.14165": ROOT / "pdf" / "2005.14165_GPT-3.pdf",
    "2508.18224": ROOT / "build" / "eval-cache" / "a3352442cd37.pdf",
}


def _hits_any(text: str, patterns: list[str]) -> bool:
    """`must_contain` 的每一项是「用 | 分隔的几种写法取 OR」，与 eval_qa 一致。

    （第一版直接把 `"96|九十六"` 整串去比对，于是每一道带别名的题都判成未命中 ——
    数字立刻从 5/13 变成该有的样子。）
    """
    low = text.lower()
    return any(
        alt.strip().lower() in low
        for p in patterns
        for alt in p.split("|")
        if alt.strip()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "scripts" / "eval_questions.json"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--no-expand", action="store_true", help="对照：不扩写，直接拿中文查")
    ap.add_argument("--llm", action="store_true", help="对照：用大模型改写检索式")
    ap.add_argument("--both", action="store_true", help="大模型改写 + 术语表，两路一起融合")
    args = ap.parse_args()

    if args.llm or args.both:  # 对照组要真调模型，先读 .env；离线组刻意不读 —— 它不该需要 key
        from ask import load_dotenv

        load_dotenv(ROOT / ".env")

    from app.parsing.chunker import chunk_pdf
    from app.rag.glossary import describe, expand_offline
    from app.rag.query import expand_query
    from app.retrieval import BM25, reciprocal_rank_fusion

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    # 只取该答的题：拒答题没有"正确答案所在块"，量不了检索
    cases = [q for q in spec if q["expect"] == "answer" and q["paper"] in PAPERS]

    indexes: dict[str, BM25] = {}
    total = ok = 0
    rows = []
    for case in cases:
        pid = case["paper"]
        if pid not in indexes:
            path = PAPERS[pid]
            if not path.exists():
                print(f"缺 {path}，跳过该论文的题")
                continue
            indexes[pid] = BM25(chunk_pdf(str(path)))
        retriever = indexes[pid]

        question = case["question"]
        if args.both:
            queries = expand_query(question)
            for extra in expand_offline(question):
                if extra not in queries:
                    queries.append(extra)
        elif args.llm:
            queries = expand_query(question)
        elif args.no_expand:
            queries = [question]
        else:
            queries = expand_offline(question) or [question]
        rankings = [r for r in (retriever.search(q, k=args.k) for q in queries) if r]
        hits = (
            rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings, limit=args.k)
        )
        found = _hits_any(
            " ".join(h.chunk.text for h in hits[: args.k]), case["must_contain"]
        )
        total += 1
        ok += found
        rows.append((pid, question, ";".join(queries)[:46], found))

    print(f"{'论文':<12} {'问题':<34} {'检索式':<46} 命中")
    print("-" * 108)
    for pid, question, queries, found in rows:
        print(f"{pid:<12} {question[:32]:<34} {queries:<46} {'✓' if found else '✗'}")
    print("-" * 108)
    print(f"答案关键词在前 {args.k} 块里：{ok}/{total} = {ok / total:.1%}")
    if not args.no_expand:
        print("\n术语表命中情况：")
        for case in cases:
            print(f"  {case['question'][:30]:<32} {describe(case['question'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
