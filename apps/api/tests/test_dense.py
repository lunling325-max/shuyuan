"""向量通道的测试。

模型是本地大文件（133MB），不能假设每台机器都有 ——
所以整组测试在模型缺席时 **skip 而不是 fail**：
代码要在没有模型的机器上照样能跑测试，降级路径才有意义。

注意：这组测试只验证"向量通道本身没写错"（池化、归一化、排序），
**不**验证"向量通道对本题库有用" —— 后者是实验结论，写在 docs/EXPERIMENTS.md E-004，
而结论恰恰是：在单篇 Transformer 论文上，它的净增益为负。两者不要混为一谈。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.embeddings import QUERY_PREFIX, load_embedder

embedder_available = pytest.mark.skipif(
    load_embedder() is None, reason="本地没有 ONNX 模型，跳过向量测试"
)

DOCS = [
    "We use eight attention heads in the multi-head attention layer.",
    "The optimizer uses Adam with beta1 equal to 0.9 and beta2 equal to 0.98.",
    "We apply label smoothing with epsilon of 0.1 during training.",
]


@pytest.fixture(scope="module")
def embedder():
    e = load_embedder()
    assert e is not None
    return e


@embedder_available
def test_output_shape_and_unit_norm(embedder):
    vecs = embedder.encode_documents(DOCS)
    assert vecs.shape == (len(DOCS), embedder.dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


@embedder_available
def test_identical_text_scores_one(embedder):
    """同一段文本，余弦相似度必须≈1 —— 池化或归一化写错时这条会立刻红。"""
    a, b = embedder.encode_documents([DOCS[0], DOCS[0]])
    assert float(a @ b) > 0.999


@embedder_available
def test_query_prefers_semantically_matching_doc(embedder):
    """对角线必须赢：查询向量与它对应的文档最相似。"""
    docs = embedder.encode_documents(DOCS)
    queries = embedder.encode_queries(
        ["how many attention heads", "what is the Adam beta2 value", "label smoothing epsilon"]
    )
    sims = queries @ docs.T
    assert np.array_equal(sims.argmax(axis=1), np.arange(len(DOCS)))


@embedder_available
def test_query_prefix_actually_changes_the_vector(embedder):
    """v1.5 的查询前缀不是装饰：加了前缀和不加，向量必须不同。"""
    with_prefix = embedder.encode_queries(["attention heads"])[0]
    without = embedder.encode_documents(["attention heads"])[0]
    assert float(with_prefix @ without) < 0.999
    assert QUERY_PREFIX.endswith("passages: ")


@embedder_available
def test_batches_match_single_pass(embedder):
    """batch 大小不能改变结果，否则线上和评测会对不上。"""
    batched = embedder.encode_documents(DOCS, batch_size=2)
    one_by_one = embedder.encode_documents(DOCS, batch_size=1)
    assert np.allclose(batched, one_by_one, atol=1e-5)


# ------------------------------------------------- 索引层（不依赖真模型）

class _FakeEmbedder:
    """把文本长度塞进第一维，方便人工推出期望排序。"""

    dim = 4

    def encode_documents(self, texts, **_):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0 if "alpha" in t else 0.0
            v[1] = 1.0 if "beta" in t else 0.0
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return np.vstack(out)

    def encode_queries(self, queries, **_):
        return self.encode_documents(queries)


def test_dense_index_ranks_by_similarity():
    from app.parsing.chunker import Chunk
    from app.retrieval.dense import DenseIndex

    chunks = [
        Chunk(index=0, text="alpha body", section="alpha", page_start=1, page_end=1,
              n_tokens=2, rects=[]),
        Chunk(index=1, text="beta body", section="beta", page_start=1, page_end=1,
              n_tokens=2, rects=[]),
    ]
    index = DenseIndex.build(chunks, _FakeEmbedder())
    hits = index.search("alpha", k=2)
    # 相似度为 0 的块被丢掉：正交就是"不相关"，不该拿来凑数
    assert [h.chunk.index for h in hits] == [0]
    assert hits[0].source == "dense"


@embedder_available
def test_batches_match_single_pass(embedder):
    """batch 大小不能改变结果，否则线上和评测会对不上。"""
    batched = embedder.encode_documents(DOCS, batch_size=2)
    one_by_one = embedder.encode_documents(DOCS, batch_size=1)
    assert np.allclose(batched, one_by_one, atol=1e-5)
