"""作者与机构从版面一路走到接口：注册表认得、sidecar 记得住、前端拿得到。

识别本身在 `test_frontmatter.py` 里测。这里只测"接线"有没有接上 ——
这类字段最常见的坏法不是认错，而是认对了却没落盘、或者落盘了却没出现在响应里，
界面上一片空白，你以为识别没生效，其实是序列化漏了一行。
"""

from __future__ import annotations

import json
from pathlib import Path

PDF_DIR = Path(__file__).resolve().parents[3] / "pdf"
BERT = PDF_DIR / "1810.04805_BERT.pdf"

BERT_AUTHORS = ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]


def _registry(tmp_path):
    """**在函数里导入，不在文件顶上导入。**

    `app.api.registry` 一被导入就会牵出整条 `app.api`，而 `deps.STORE_DIR`
    是导入时读环境变量求值的。谁先导入谁定调：本模块先导入，整条测试进程的
    资料库就指向真实目录，`test_api.py` 里几条"去重 / 版本"用例立刻假红
    （实测踩过，一次跑出 3 个红）。`test_api.py` 一直就是这个写法 ——
    在 fixture 里先设 `PAPER_STORE_DIR` 再导入。
    """
    from app.api.registry import PaperRegistry

    return PaperRegistry(tmp_path)


def test_upload_records_authors_and_affiliations(tmp_path):
    registry = _registry(tmp_path)
    paper = registry.add(BERT.read_bytes(), BERT.name)

    assert paper.authors == BERT_AUTHORS
    assert paper.affiliations == ["Google AI Language"]

    meta = json.loads((tmp_path / f"{paper.id}.json").read_text(encoding="utf-8"))
    assert meta["authors"] == BERT_AUTHORS
    assert meta["affiliations"] == ["Google AI Language"]


def test_old_sidecar_without_the_field_is_backfilled(tmp_path):
    """后加的字段：仓库里已有的 .json 里没有 authors/affiliations 这两栏。

    不补的话，升级之后老文献的作者永远空着 —— 除非用户重新上传一遍，
    而重新上传会撞上"内容重复"，`add` 直接返回库里那条，还是空的。
    所以启动时按"键在不在"补算一次，算完回写，下次启动就是读盘。
    """
    registry = _registry(tmp_path)
    paper = registry.add(BERT.read_bytes(), BERT.name)
    meta_path = tmp_path / f"{paper.id}.json"

    old = json.loads(meta_path.read_text(encoding="utf-8"))
    old.pop("authors")
    old.pop("affiliations")
    meta_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    reloaded = _registry(tmp_path)  # 模拟进程重启
    assert reloaded.get(paper.id).authors == BERT_AUTHORS
    assert "authors" in json.loads(meta_path.read_text(encoding="utf-8")), "算完要回写"


# 接口层的那两条（`PaperOut` 带着字段、返回的列表是副本）写在 `tests/test_api.py`。
# 这里刻意**不导入 `app.api.deps`**：它的 `STORE_DIR` 是导入时求值的，
# 在本模块先导入会把整个测试进程指到真实资料库上，`test_api.py` 里几条
# "去重 / 版本"用例立刻假红 —— 实测踩过，一次跑出 3 个红。
