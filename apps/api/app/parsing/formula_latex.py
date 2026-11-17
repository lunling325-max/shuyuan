"""公式图 → LaTeX（可选能力，模型不在就明确说不可用）。

为什么值得单独做这一层：PDF 里没有 LaTeX 源码，公式只是排好版的字形。
赛题模块一要求「细粒度抽取并结构化呈现 … LaTeX 公式」，只给公式裁图不算达标。
这里接的是 ONNX 版 pix2text-mfr（TrOCR 结构：DeiT 编码器 + Transformer 解码器，
词表 1200），只依赖 onnxruntime 与 tokenizers，**不引入 torch**。

三条设计约束：
1. **模型缺失不是崩溃**：`available()` 为假时调用方返回明确的"不可用"，不抛栈；
2. **按需识别**：单条 0.7~1.4 秒（CPU），一本 541 页的书可能有几百条公式，
   绝不能放进入库流程；
3. **识别结果落盘缓存**：同一份 PDF 重算没有意义，且缓存能让"重开页面"瞬间出结果。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

# 仓库根下的 models/mfr（models/ 已在 .gitignore 里）；容器里由 FORMULA_OCR_DIR 指过去
MODEL_DIR = Path(os.environ.get("FORMULA_OCR_DIR")
                 or Path(__file__).resolve().parents[4] / "models" / "mfr")
SIZE = 384           # preprocessor_config.json 里的 size
DECODER_START = 2    # </s> 作为解码起点
EOS = 2
MAX_NEW = 256

REQUIRED = ("encoder_model.onnx", "decoder_model.onnx", "tokenizer.json")


class OcrUnavailable(RuntimeError):
    """模型或运行时不在 —— 调用方据此降级，不要当成解析失败。"""


def available() -> bool:
    if not all((MODEL_DIR / n).exists() for n in REQUIRED):
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except Exception:  # noqa: BLE001 —— 缺依赖同样是"不可用"
        return False
    return True


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if not available():
        raise OcrUnavailable("公式识别模型未就位")
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    enc = ort.InferenceSession(str(MODEL_DIR / "encoder_model.onnx"),
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(MODEL_DIR / "decoder_model.onnx"),
                               providers=["CPUExecutionProvider"])
    return tok, enc, dec


def _prep(data: bytes):
    """按 preprocessor_config.json：RGB、384×384、/255 后按 0.5 均值方差归一。"""
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None, ...]


def _to_latex(pieces: list[str]) -> str:
    """把词元还原成 LaTeX。

    这个模型几乎每个字形都带前导 `Ġ`（空格标记），命令名（mathrm/cdot/operatorname）
    不带 —— 所以空格必须保留，否则 `\\cdot warmup` 会粘成不存在的命令 `\\cdotwarmup`。
    只把花括号组内部的空格挤掉；组外的单字母串（`l r a t e`）合起来。
    """
    text = "".join(p if not p.startswith("Ġ") else " " + p[1:] for p in pieces)
    text = " ".join(text.split())
    prev = None
    while prev != text:                       # 由内向外反复处理嵌套花括号
        prev = text
        text = re.sub(r"\{([^{}]*)\}", lambda m: "{" + m.group(1).replace(" ", "") + "}", text)
    text = re.sub(r"(?<![\\A-Za-z])((?:[A-Za-z] )+[A-Za-z])(?![A-Za-z])",
                  lambda m: m.group(1).replace(" ", ""), text)
    # 运算符与括号两侧不留空格：`d _ { \mathrm {m} }` → `d_{\mathrm{m}}`。
    # 只在这些字符旁边动手，所以 `\cdot warmup` 里那个空格（两侧都是字母）保得住。
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\s*([{}_^=,()+-])\s*", r"\1", text)
        text = re.sub(r"\s+(?=\\)", "", text)     # `step \_num` → `step\_num`
    return text


def recognise(data: bytes) -> str:
    """一张公式图 → LaTeX 字符串。模型不可用时抛 `OcrUnavailable`。"""
    import numpy as np

    tok, enc, dec = _session()
    hidden = enc.run(None, {"pixel_values": _prep(data)})[0]
    ids = [DECODER_START]
    for _ in range(MAX_NEW):
        logits = dec.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                "encoder_hidden_states": hidden})[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOS:
            break
        ids.append(nxt)
    return _to_latex([tok.id_to_token(i) for i in ids[1:]])


# ---------------------------------------------------------------- 缓存

def _cache_path(store_dir: Path, paper_id: str) -> Path:
    return Path(store_dir) / f"{paper_id}.latex.json"


def load_cache(store_dir: Path, paper_id: str) -> dict[str, str]:
    path = _cache_path(store_dir, paper_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(store_dir: Path, paper_id: str, cache: dict[str, str]) -> None:
    try:
        _cache_path(store_dir, paper_id).write_text(
            json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except OSError:
        pass       # 缓存写不进去不该让识别结果作废


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
