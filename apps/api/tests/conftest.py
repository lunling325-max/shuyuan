"""演示 PDF 不进仓库（`pdf/` 在 `.gitignore` 里），所以从零 clone 下来要先处理一件事：
用到这些 PDF 的用例要么老实跳过并说清缺哪个文件，要么整片变红。

选跳过。理由：那些用例真正依赖的是"有一份 15 页双栏论文"，不是这台机器的状态。
缺文件时让 100 多条用例同时 ERROR/FAILED，读的人分不清是"代码坏了"还是"少了资料"，
反而把真正该看的红灯埋掉。

跑全绿的办法：把 PDF 放进仓库根的 `pdf/`（或 `uv run python scripts/fetch_demo_pdfs.py`
从 arXiv 抓公开的那几份）。文件名照 README 里的清单，用例是按名字找的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

PDF_DIR = Path(__file__).resolve().parents[3] / "pdf"


def _missing_pdfs(module) -> list[Path]:
    """模块级常量里，指向 `pdf/` 但文件不在的那些（含 fixture 依赖的）。"""
    out: list[Path] = []
    for value in vars(module).values():
        if isinstance(value, Path) and PDF_DIR in value.parents and not value.exists():
            out.append(value)
    return out


def pytest_collection_modifyitems(config, items):
    checked: dict[str, list[Path]] = {}
    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        if module.__name__ not in checked:
            checked[module.__name__] = _missing_pdfs(module)
        missing = checked[module.__name__]
        if missing:
            names = "、".join(sorted({p.name for p in missing}))
            item.add_marker(
                pytest.mark.skip(reason=f"缺少演示 PDF：{names}（放进 pdf/ 就能跑，见 README「快速开始」）")
            )
