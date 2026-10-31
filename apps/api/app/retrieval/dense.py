"""向量检索：把块编码成向量，按余弦相似度召回。

当前实现在内存里做穷举（28 块、841 页的教材也就几千块），
毫秒级，不需要向量数据库。规模上去以后换 Qdrant —— `search_vector` 的签名
就是给它留的边界：上层只依赖"给我一个向量，还我 k 条命中"。

为什么不直接上 Qdrant？
这次先证明**语义通道能不能解决问题**（E-003 里那 5 道词法搞不定的题）。
基础设施的复杂度要等收益被证明之后再付 —— 现在证明的结果是：**不值得付**。

实测结论（docs/EXPERIMENTS.md E-004，18 题、检索式冻结）：
向量单路 66.7% / 88.9%，词法 83.3% / **100%**，双通道 RRF 融合反而掉到 61.1%。
换句话说，在这篇论文上它没有净增益，**因此不进默认检索路径**，
只作为 `--dense` / `--hybrid` 的可选消融开关保留。

保留而不是删除的理由：它的价值不在当前指标，而在下一步 ——
语料从 1 篇变成 10+ 篇、或用户用中文问英文库时，词法会整体失效，语义通道才是主力。
`search_vector` 的签名就是给那时的 Qdrant 留的边界：上层只依赖"给个向量，还我 k 条命中"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.parsing.chunker import Chunk
from app.retrieval.bm25 import Hit


@dataclass
class DenseIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray  # (n_chunks, dim)，已 L2 归一化
    embedder: object  # 只用到 encode_queries

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "DenseIndex":
        # 章节标题一起编码：正文里满是 "we" "the model"，标题才是主题信号
        texts = [f"{c.section}\n{c.text}" for c in chunks]
        return cls(chunks=chunks, embeddings=embedder.encode_documents(texts), embedder=embedder)

    def search_vector(self, vector: np.ndarray, k: int = 6) -> list[Hit]:
        if len(self.chunks) == 0:
            return []
        scores = self.embeddings @ np.asarray(vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(chunk=self.chunks[i], score=round(float(scores[i]), 4), source="dense")
            for i in order
            if scores[i] > 0
        ]

    def search(self, query: str, k: int = 6) -> list[Hit]:
        vec = self.embedder.encode_queries([query])[0]
        return self.search_vector(vec, k=k)
