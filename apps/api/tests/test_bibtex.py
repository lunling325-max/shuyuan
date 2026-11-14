"""BibTeX 导出：能编译、键不撞、缺的东西说出来。

守四件事：

1. **转义**：标题里带 `&`、`%`、`_` 的条目，导出后不会让 bibtex 报错；
2. **引用键**：名在前的排版（`Ashish Vaswani, …`）取姓，姓在前的（`Vaswani, Ashish`）也取姓，
   同一份文件里键不重复；
3. **不完整的条目照样导出**，并在 `note` 里说明字段不全 —— 少一条比给一条残的更难发现；
4. **端点**：给出 `text/plain` + `Content-Disposition`，前端 `<a download>` 才能直接存盘。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.bibtex import _bibtex_authors, escape, to_bibtex
from app.references import Reference


def _ref(key="1", authors="", title="", venue="", year="", arxiv_id="", raw=""):
    return Reference(
        key=key,
        raw=raw or f"{authors} {title}",
        authors=authors,
        title=title,
        venue=venue,
        year=year,
        arxiv_id=arxiv_id,
    )


# ------------------------------------------------------------------ 转义

def test_special_characters_are_escaped():
    assert escape("A & B") == r"A \& B"
    assert escape("100%") == r"100\%"
    assert escape("a_b") == r"a\_b"
    assert escape(r"back\slash") == r"back\textbackslash{}slash"
    assert escape("plain text") == "plain text"


def test_escape_runs_backslash_first_so_it_is_not_double_escaped():
    # 先把 \ 换成 \textbackslash{}，再处理其它字符；顺序反了会得到 \\textbackslash{}
    out = escape(r"& \ %")
    assert out.count("textbackslash") == 1
    assert r"\\" not in out


# ------------------------------------------------------------------ 引用键

def test_key_uses_the_surname_in_both_author_layouts():
    given_first = to_bibtex([_ref(authors="Ashish Vaswani, Noam Shazeer", title="Attention Is All You Need", year="2017")])
    last_first = to_bibtex([_ref(authors="Vaswani, Ashish; Shazeer, Noam", title="Attention Is All You Need", year="2017")])
    assert "@article{vaswani2017attention," in given_first or "@misc{vaswani2017attention," in given_first
    assert "@article{vaswani2017attention," in last_first or "@misc{vaswani2017attention," in last_first


def test_keys_do_not_collide_within_one_file():
    refs = [
        _ref(key="1", authors="Ashish Vaswani", title="Attention Is All You Need", year="2017"),
        _ref(key="2", authors="Ashish Vaswani", title="Attention Is All You Need", year="2017"),
    ]
    body = to_bibtex(refs)
    keys = [line.split("{", 1)[1].rstrip(",") for line in body.splitlines() if line.startswith("@")]
    assert len(keys) == 2 and keys[0] != keys[1]


def test_entry_without_author_or_title_still_gets_a_key():
    body = to_bibtex([_ref(key="7", raw="Some unparseable reference text")])
    assert "@misc{ref7," in body


# ------------------------------------------------------------------ 完整性

def test_unparsed_entry_is_exported_with_a_note_instead_of_dropped():
    refs = [
        _ref(key="1", authors="A. Author", title="A Real Title", year="2020"),
        _ref(key="2", raw="[2] 某某等. 一篇没能解析出来的文献. 2019."),
    ]
    body = to_bibtex(refs)
    assert "A Real Title" in body
    assert "note = {" in body and "字段可能不完整" in body
    assert "% 共 2 条" in body


def test_empty_reference_list_says_so_instead_of_returning_an_empty_file():
    body = to_bibtex([])
    assert "没有解析出可导出的参考文献条目" in body


def test_author_list_is_converted_to_bibtex_and_form():
    """`A, B, and C` 必须变成 `A and B and C`。

    不转的话 BibTeX 会把「逗号」当成"姓, 名"的分隔：整串读成两个作者，
    第一个的姓是 `Jimmy Lei Ba`、名是 `Jamie Ryan Kiros`。用 pandoc 的 citeproc
    实测渲染出 "Jimmy Lei Ba and Hinton 2016" —— 名字全乱。转完是 "Ba et al. 2016"。
    """
    assert _bibtex_authors("Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton") ==         "Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E Hinton"
    # 「姓, 名; 姓, 名」的排版：分号切完用 and 连，逗号要保留（那是姓名的分隔）
    assert _bibtex_authors("Vaswani, Ashish; Shazeer, Noam") == "Vaswani, Ashish and Shazeer, Noam"
    assert _bibtex_authors("Francois Chollet") == "Francois Chollet"
    assert _bibtex_authors("") == ""
    # "et al" 不是作者，丢掉
    assert _bibtex_authors("A. Author, B. Author, et al") == "A. Author and B. Author"


def test_arxiv_id_becomes_eprint_fields():
    body = to_bibtex([_ref(authors="Ashish Vaswani", title="Attention", year="2017", arxiv_id="1706.03762")])
    assert "eprint = {1706.03762}" in body
    assert "archiveprefix = {arXiv}" in body


def test_entry_type_is_guessed_from_the_venue():
    conf = to_bibtex([_ref(authors="X Y", title="T", year="2020", venue="Proceedings of NeurIPS")])
    journal = to_bibtex([_ref(authors="X Y", title="T", year="2020", venue="Nature Machine Intelligence")])
    other = to_bibtex([_ref(authors="X Y", title="T", year="2020", venue="Some Blog")])
    assert "@inproceedings{" in conf
    assert "@article{" in journal
    assert "@misc{" in other


# ------------------------------------------------------------------ 端点

def test_endpoints_send_a_downloadable_text_file(monkeypatch, tmp_path):
    """端点层面：`text/plain` + `Content-Disposition`，前端 <a download> 才能直接存盘。

    库换成空的临时注册表 —— 直接读本机真实资料的话，这条断言会随"这台机器上有没有文献"变化。
    注意要换 `app.api.main.registry`：main 里是 `from ... import registry`，
    换 `deps.registry` 影响不到它已绑定的名字。
    """
    from app.api import main as main_module
    from app.api.registry import PaperRegistry

    monkeypatch.setattr(main_module, "registry", PaperRegistry(tmp_path / "papers"))
    client = TestClient(main_module.app)

    r = client.get("/api/bibtex")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "library.bib" in r.headers.get("content-disposition", "")
    assert "没有解析出可导出的参考文献条目" in r.text

    assert client.get("/api/papers/nope/bibtex").status_code == 404
