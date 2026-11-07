"""把一份矢量 PDF 渲染成"只剩像素"的扫描件，用来验 OCR 链路。

为什么自己造而不是去下：
赛题 3.2 点名要 `1706.03762v1_img.pdf`（官方给的图像扫描版），但那份文件
不在仓库里，而"扫描件"这个用例考的是**没有文字层时系统怎么办**，不是那张
特定的图片。把已有的 v1 逐页渲染成位图再拼回 PDF，就得到了一份形态完全相同
的样本：`get_text()` 拿到空串，页面靠 OCR 才能读出字。

区别写在文件名里（`_scan` 而不是 `_img`），免得和组委会那份混淆；
生成结果也记进 `pdf/SOURCES.md`。

用法（在项目根目录）：
    PYTHONUTF8=1 uv run --with pymupdf python scripts/make_scan_sample.py \
        pdf/1706.03762v1.pdf pdf/1706.03762v1_scan.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pymupdf

DEFAULT_DPI = 200
JPEG_QUALITY = 80


def render_to_scans(src: Path, dst: Path, *, dpi: int = DEFAULT_DPI) -> dict:
    """逐页渲染成 JPEG 再塞进一份新 PDF —— 新文件不带任何文字层。"""
    source = pymupdf.open(str(src))
    out = pymupdf.open()
    try:
        for page in source:
            pix = page.get_pixmap(dpi=dpi)
            img = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            # 新页的尺寸按原页 pt 值给，保证版面比例不变
            rect = pymupdf.Rect(0, 0, page.rect.width, page.rect.height)
            new_page = out.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=img)
        out.save(str(dst), deflate=True, garbage=3)
        return {"pages": source.page_count, "bytes": dst.stat().st_size}
    finally:
        out.close()
        source.close()


def text_chars(path: Path) -> int:
    """文字层里非空白字符的总数：扫描件应该接近 0。"""
    doc = pymupdf.open(str(path))
    try:
        return sum(len("".join(p.get_text("text").split())) for p in doc)
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成扫描件样本")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    if not args.src.exists():
        print(f"找不到源文件：{args.src}")
        return 1

    t0 = time.perf_counter()
    info = render_to_scans(args.src, args.dst, dpi=args.dpi)
    took = time.perf_counter() - t0

    digest = hashlib.sha256(args.dst.read_bytes()).hexdigest()
    before, after = text_chars(args.src), text_chars(args.dst)
    print(f"生成 {args.dst}")
    print(f"  页数 {info['pages']} · 体积 {info['bytes']:,} bytes · 耗时 {took:.1f}s · dpi {args.dpi}")
    print(f"  文字层字符数：源 {before:,} → 扫描件 {after}（{after == 0 and '没有文字层，符合预期' or '还有残留，需要检查'}）")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
