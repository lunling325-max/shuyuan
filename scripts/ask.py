"""命令行问答演示：问一篇 PDF 一个问题，打印带页码的答案。

用法（在项目根目录）：
    PYTHONUTF8=1 uv run --with pymupdf python scripts/ask.py "为什么要除以 sqrt(dk)？"

默认论文是 Attention Is All You Need v7，可用 --pdf 换。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.config import load_dotenv  # noqa: E402 —— 必须在 sys.path 之后导入


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pdf", default=str(ROOT / "pdf" / "1706.03762v7.pdf"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--no-expand", action="store_true", help="关闭检索词扩展（单路 BM25）")
    ap.add_argument(
        "--dense",
        action="store_true",
        help="额外启用向量通道（实验性；实测在单篇论文上无净增益，见 E-004）",
    )
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    from app.parsing.chunker import chunk_pdf
    from app.parsing.text_index import TextIndex
    from app.rag import answer_question
    from app.rag.query import expand_query, retrieve
    from app.retrieval import BM25

    print(f"论文: {Path(args.pdf).name}")
    chunks = chunk_pdf(args.pdf)
    retriever = BM25(chunks)
    index = TextIndex(args.pdf)
    print(f"切块: {len(chunks)} 块 · 词级索引: {len(index)} token\n")

    queries = expand_query(args.question) if not args.no_expand else [args.question]
    if not args.no_expand:
        print("扩展检索式:", " / ".join(queries))

    dense = None
    if args.dense:
        from app.retrieval.dense import DenseIndex
        from app.retrieval.embeddings import load_embedder

        embedder = load_embedder()
        if embedder is None:  # 降级，不中断问答
            print("（找不到本地向量模型，本次只走词法通道）")
        else:
            dense = DenseIndex.build(chunks, embedder)
            print(f"向量通道: {dense.embeddings.shape[0]} 块 × {embedder.dim} 维")

    hits = retrieve(
        args.question,
        retriever=retriever,
        dense=dense,
        k=args.k,
        expand=not args.no_expand,
        queries=queries,  # 把打印出来的那套检索式原样交给检索，避免二次采样
    )
    print("检索命中:")
    for i, h in enumerate(hits, start=1):
        print(f"  [#{i}] rrf={h.score:.4f} p.{h.chunk.page_start} {h.chunk.section[:52]}")
    print()

    answer = answer_question(
        args.question, hits=hits, index=index, doc_id="paper"
    )

    print("答案:")
    print(answer.render())
    print()

    stats = answer.stats
    print(f"守卫统计: supported={stats['supported']} weak={stats['weak']} "
          f"unsupported={stats['unsupported']}")
    if answer.dropped:
        print("被拦下的结论（幻觉候选）:")
        for c in answer.dropped:
            print(f"  ✗ {c.claim.text}  ← {c.reason}")

    print("\n锚点明细:")
    for c in answer.kept:
        for a in c.anchors:
            x0, y0, x1, y1 = a.rect
            print(f"  p.{a.page} [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}] "
                  f"{'逐字' if a.exact else '模糊'} ← {a.source_text[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
