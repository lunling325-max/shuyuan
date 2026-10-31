"""文本嵌入：BGE（small/en）的 ONNX 推理实现。

为什么不用 sentence-transformers？
它会把 torch 一起拖进来（Windows 上又一个 200MB+ 的依赖），
而这里需要的只是"文本 → 384 维向量"。onnxruntime + tokenizers 加起来不到 20MB，
启动快，部署到没有 GPU 的机器上也不心疼 —— 比赛演示的环境不一定有显卡。

BGE 的正确用法（不是随便取个均值）：
- 池化用 **CLS**（第 0 个 token），不是 mean pooling；
- 查询侧要加指令前缀，文档侧不加 —— 这是 bge v1.5 的 s2p 检索设定；
- 向量做 L2 归一化后，余弦相似度退化成点积，省一次除法。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "bge-small-en-v1.5"

# bge v1.5 英文版官方建议的查询前缀（只加在 query 上）
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512


class EmbedderUnavailable(RuntimeError):
    """模型不可用。上层应当降级到纯词法检索，而不是整个挂掉。"""


class OnnxBgeEmbedder:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "onnx" / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tokenizer_path.exists():
            raise EmbedderUnavailable(
                f"找不到本地模型：{self.model_dir}（需要 onnx/model.onnx 与 tokenizer.json）"
            )

        # 依赖缺失也要走"不可用"这条路：装了模型没装 onnxruntime 的机器
        # 必须降级到纯词法，而不是在导入阶段就把整个服务打挂。
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise EmbedderUnavailable(f"缺少向量依赖：{exc}") from exc

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self._tok.enable_truncation(max_length=MAX_TOKENS)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

        # 输出维度从模型里读，不写死 —— 换模型时不会静默错位
        self.dim: int = int(self._session.get_outputs()[0].shape[-1])

    # ------------------------------------------------------------ 推理

    def _run(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        enc = self._tok.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in enc], dtype=np.int64
            )

        hidden = self._session.run(None, feed)[0]
        cls = hidden[:, 0, :]  # CLS pooling
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        return (cls / np.maximum(norms, 1e-9)).astype(np.float32)

    def encode_documents(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        out = [self._run(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode_queries(self, queries: list[str], *, batch_size: int = 32) -> np.ndarray:
        return self.encode_documents(
            [QUERY_PREFIX + q for q in queries], batch_size=batch_size
        )


def load_embedder(model_dir: str | Path | None = None) -> OnnxBgeEmbedder | None:
    """拿得到模型就用，拿不到就返回 None（由调用方降级到纯词法）。"""
    try:
        return OnnxBgeEmbedder(model_dir or DEFAULT_MODEL_DIR)
    except EmbedderUnavailable:
        return None
