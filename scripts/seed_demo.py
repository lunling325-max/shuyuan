"""把一批 PDF 灌进**已经在跑**的服务里，省得演示时一篇篇手点。

    uv run python scripts/seed_demo.py                  # 灌仓库根 pdf/ 里的全部 PDF
    uv run python scripts/seed_demo.py docs/demo        # 换目录
    uv run python scripts/seed_demo.py --server http://127.0.0.1:8020

只走 HTTP（`POST /api/papers`），不直接写 `data/`：服务在内存里也存着一份索引，
绕过接口直接落盘会让两边对不上。同一份 PDF 传两次是幂等的（内容哈希做 id），
所以这个脚本可以反复跑。

只用标准库：演示机器上不需要额外装东西。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "pdf"
DEFAULT_SERVER = "http://127.0.0.1:8017"


def _safe_name(name: str) -> str:
    """multipart 头里只放 ASCII 文件名；中文名退化成 paper.pdf，内容不受影响。"""
    ascii_name = name.encode("ascii", "ignore").decode().strip()
    return ascii_name or "paper.pdf"


def _body(path: Path, boundary: str) -> bytes:
    ctype = mimetypes.guess_type(path.name)[0] or "application/pdf"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{_safe_name(path.name)}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + path.read_bytes() + tail


def upload(server: str, path: Path, timeout: float = 600.0) -> tuple[bool, str]:
    """返回 (是否成功, 一句话结果)。541 页的书要几十秒，超时给得宽一些。"""
    boundary = "----seed" + str(abs(hash(path.name)))
    req = urllib.request.Request(
        server.rstrip("/") + "/api/papers",
        data=_body(path, boundary),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return True, f"{data.get('pages')} 页 / {data.get('chunks')} 块 · 《{data.get('title')}》"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail") or detail
        except ValueError:
            pass
        return False, f"HTTP {exc.code}：{detail}"
    except urllib.error.URLError as exc:
        return False, f"连不上 {server}（服务起了吗？）：{exc.reason}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把一批 PDF 灌进正在运行的服务")
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_DIR), help="PDF 所在目录")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="服务地址")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"没有这个目录：{folder}", file=sys.stderr)
        return 2
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"{folder} 里没有 PDF。先把几篇论文放进去，或者用界面上的「＋ 上传 PDF」。")
        return 1

    failed = 0
    for path in pdfs:
        print(f"… {path.name}", flush=True)
        ok, message = upload(args.server, path)
        print(("  已入库 " if ok else "  失败   ") + message)
        failed += 0 if ok else 1

    print(f"\n共 {len(pdfs)} 篇，失败 {failed} 篇。打开 {args.server.rstrip('/')}/ 看看。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
