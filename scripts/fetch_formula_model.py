"""下载公式识别模型，并逐个核对**官方哈希**；对不上就让构建失败。

为什么核得这么细：这台机器连不上 HuggingFace，线上只能走镜像（hf-mirror.com），
而镜像的内容无法验签 —— 唯一能验证的就是"文件内容是否等于官方发布的那个"。
所以哈希值全部写死在这里，从 HuggingFace 元数据生成：

- `.onnx` 是 LFS 文件，官方 `lfs.oid` 就是**内容的 sha256**；
- 其余 JSON 是普通 git 文件，官方 `oid` 是 **git blob 的 sha1**，
  即 sha1(b"blob <长度>" + b"\\0" + 内容)。

用法：
    python fetch_formula_model.py <目标目录>
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://hf-mirror.com/breezedeus/pix2text-mfr/resolve/main/"

FILES = {
    "config.json": ("git-sha1", "c6f828ccd5f3e8781dc7c7a715bc4b8f80ff41bc"),
    "decoder_model.onnx": ("sha256",
                           "fd0f92d7a012f3dae41e1ac79421aea0ea888b5a66cb3f9a004e424f82f3daed"),
    "encoder_model.onnx": ("sha256",
                           "bd8d5c322792e9ec45793af5569e9748f82a3d728a9e00213dbfc56c1486f37d"),
    "generation_config.json": ("git-sha1", "a3d09b3add4319b3c2d0ca15011f3618109df47b"),
    "preprocessor_config.json": ("git-sha1", "c2bbec3a0dbefdd3ecce8a82458664790ce39b20"),
    "special_tokens_map.json": ("git-sha1", "b1879d702821e753ffe4245048eee415d54a9385"),
    "tokenizer.json": ("git-sha1", "c07aa39397f33b7822ef84e435e911a70a4ce303"),
    "tokenizer_config.json": ("git-sha1", "bdfa7cd6ca95d1662e765144547e7952869bed33"),
}

NULL = bytes([0])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha1(data: bytes) -> str:
    """git 对 blob 的哈希：先拼 `blob <字节数>` + NUL，再算 sha1。"""
    header = b"blob " + str(len(data)).encode() + NULL
    return hashlib.sha1(header + data).hexdigest()


def main(out_dir: str) -> int:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, (algo, want) in FILES.items():
        path = target / name
        if path.exists():
            data = path.read_bytes()
        else:
            print(f"  下载 {name} …", flush=True)
            with urllib.request.urlopen(BASE + name, timeout=900) as r:
                data = r.read()
            path.write_bytes(data)
        got = _sha256(data) if algo == "sha256" else _git_sha1(data)
        if got != want:
            print(f"  [x] {name}\n      期望 {want}\n      实际 {got}", flush=True)
            failed += 1
        else:
            print(f"  [ok] {name} ({len(data) // 1024} KB, {algo})", flush=True)
    if failed:
        print(f"{failed} 个文件哈希不一致，拒绝使用这批模型。", flush=True)
        return 1
    print("全部核对通过。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./models/mfr"))
