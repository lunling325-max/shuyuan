"""版本谱系的测试。

这份文件测的其实是**降噪**，不是 diff 本身：`difflib` 会把两版之间的文本差异
一条条吐出来，那部分几乎不会错。真正的难点是"哪些差异根本不值得说"——
脚注编号、行尾连字符、参考文献重排，这些占 v1→v7 全部变更的六成以上。
把它们当成修订喂给大模型，模型就会一本正经地总结出并不存在的学术改动。

所以这里的断言大多形如"**这个不该出现**"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.versioning import (
    DISTINCT_THRESHOLD,
    MATCH_RATIO,
    SAME_VERSION_THRESHOLD,
    TRIVIAL_RATIO,
    Change,
    Sentence,
    _merge_adjacent,
    _refine,
    classify,
    jaccard,
    normalize,
    relate,
    shingles,
)

ROOT = Path(__file__).resolve().parents[3]
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

needs_pdfs = pytest.mark.skipif(
    not (PDF_V1.exists() and PDF_V7.exists()), reason="样本 PDF 不在仓库里"
)


# ---------------------------------------------------------------- 归一化


def test_normalize_repairs_pdf_hyphenation():
    """PDF 断词造成的伪差异必须先修掉。

    "signifi-" 换行 "cantly" 和 "significantly" 在字符层面完全不同，
    不修的话每一处跨行长词都会变成一条"改写"。
    """
    assert normalize("This is signifi-\ncantly faster.") == "this is significantly faster."


def test_normalize_preserves_numbers():
    """归一化绝不能削掉数字。

    曾经有过一条"剥上标脚注编号"的规则，在两份 PDF 上命中 91 处，
    真脚注不到 5 处 —— "BLEU 41.0" 变成 ".0"、"Figure 2" 变成 "figure"、
    "layer 5 of 6" 和 "layer 6 of 6" 归一化后完全一样。
    删掉它之后 diff 结果一字未变（41 处 / 79 条 trivial），误伤却是实打实的。
    这条测试就是防止它被"顺手"加回来。
    """
    assert normalize("We use 8 GPUs.") == "we use 8 gpus."
    assert normalize("the BLEU score of 41.0") == "the bleu score of 41.0"
    assert normalize("See Figure 2. Then") == "see figure 2. then"
    assert normalize("layer 5 of 6") != normalize("layer 6 of 6")


def test_normalize_folds_ligatures_and_case():
    assert normalize("eﬃcient") == "efficient"
    assert normalize("  Multi   Head\n") == "multi head"


# ---------------------------------------------------------------- 指纹 / 关系


def test_shingles_are_insensitive_to_whitespace_and_case():
    a = shingles("the quick brown fox jumps over the lazy dog")
    b = shingles("The quick\n brown fox   jumps over the lazy dog")
    assert a == b


def test_short_text_still_produces_a_shingle():
    """少于 k 个词的短句不能返回空集合 —— 空集合会让 jaccard 假报 0，
    两段完全一样的短文本会被判成"无关"。"""
    assert shingles("low rank") == {"low rank"}
    assert shingles("") == set()


def test_classify_uses_the_measured_thresholds():
    assert classify(SAME_VERSION_THRESHOLD).kind == "new_version"
    assert classify(SAME_VERSION_THRESHOLD - 1e-6).kind == "related"
    assert classify(DISTINCT_THRESHOLD).kind == "related"
    assert classify(DISTINCT_THRESHOLD - 1e-6).kind == "distinct"


def test_identical_text_is_reported_as_identical_not_just_similar():
    """"内容一样"和"很像"要分开：老版本库里会提示"确实是同一份，别重复上传"，
    新版本库里提示的是"要不要对比一下"。说成同一件事，提示就只剩一句废话。"""
    rel = relate("Attention is all you need.", "  attention IS all\nyou need.  ")
    assert rel.kind == "identical"
    assert rel.similarity == 1.0
    assert relate("one two three four five six", "seven eight nine ten eleven").kind == "distinct"


# ---------------------------------------------------------------- 句级降噪


def test_punctuation_only_edit_is_dropped_as_trivial():
    """只改了数字写法的句子不该被当成修订上报。"""
    old = [Sentence(1, "We use eight GPUs to train the model for twelve hours.", "3.1")]
    new = [Sentence(1, "We use eight GPUs to train the model for 12 hours.", "3.1")]
    changes, dropped = _refine(old, new, drop_trivial=True)
    assert changes == []
    assert dropped == 1

    kept, _ = _refine(old, new, drop_trivial=False)
    assert len(kept) == 1 and kept[0].kind == "replace"


def test_substantive_rewrite_is_kept_with_both_sides():
    old = [Sentence(1, "We train our models on the standard WMT 2014 English-German dataset.", "3.1")]
    new = [Sentence(1, "Our experiments use the WMT 2014 English-to-German translation benchmark.", "3.1")]
    changes, dropped = _refine(old, new, drop_trivial=True)
    assert dropped == 0
    assert len(changes) == 1
    c = changes[0]
    assert c.kind == "replace"
    # 摘要要能看着新旧两句说话，只给新句等于让它猜"原来是什么"
    assert c.old == [old[0].text]
    assert c.new == [new[0].text]
    assert c.section == "3.1"


def test_unrelated_sentences_become_insert_and_delete_not_replace():
    """配不上对的句子不是"改写"——说成改写会让读者以为原意还在。"""
    old = [Sentence(2, "Results were measured on the test set.")]
    new = [Sentence(2, "Dividend yields fell sharply during the third quarter.")]
    changes, _ = _refine(old, new, drop_trivial=True)
    kinds = sorted(c.kind for c in changes)
    assert kinds == ["delete", "insert"]
    assert 0.0 < MATCH_RATIO <= TRIVIAL_RATIO  # 分档顺序不能反


def test_refine_pairs_sentences_one_to_one():
    """贪婪配对必须一人一个 —— 同一句被配给两条新句会让改动数虚高一倍。"""
    old = [Sentence(1, "The encoder is composed of a stack of six identical layers.", "3.1")]
    new = [
        Sentence(1, "We replaced the encoder with a convolutional stack of nine layers.", "3.1"),
        Sentence(1, "We replaced the encoder with a convolutional stack of nine layers, again.", "3.1"),
    ]
    changes, _ = _refine(old, new, drop_trivial=True)
    paired = [c for c in changes if c.old]
    assert len(paired) == 1, "同一句旧句被配给了两条新句"


# ---------------------------------------------------------------- 合并


def test_adjacent_changes_in_the_same_section_merge():
    a = Change("insert", [], [4], [], ["first new sentence."], "3.2")
    b = Change("insert", [], [4], [], ["second new sentence."], "3.2")
    merged = _merge_adjacent([a, b])
    assert len(merged) == 1
    assert merged[0].new == ["first new sentence.", "second new sentence."]


def test_changes_never_merge_across_sections():
    """合并后那条只能写其中一节的 label，等于对读者撒谎。"""
    a = Change("insert", [], [4], [], ["x."], "3.2")
    b = Change("insert", [], [4], [], ["y."], "4.1")
    assert len(_merge_adjacent([a, b])) == 2


def test_adjacent_changes_on_different_pages_stay_separate():
    a = Change("insert", [], [4], [], ["x."], "3.2")
    b = Change("insert", [], [5], [], ["y."], "3.2")
    assert len(_merge_adjacent([a, b])) == 2


# ---------------------------------------------------------------- 真 PDF


@needs_pdfs
def test_v1_and_v7_are_classified_as_versions_of_one_paper():
    """E-001 标定过的关键数字：同一篇 0.70，不同论文 0.003，差两个数量级。"""
    from app.parsing.chunker import chunk_pdf

    def fp(p):
        return shingles(" ".join(c.text for c in chunk_pdf(p)))

    v1_v7 = jaccard(fp(PDF_V1), fp(PDF_V7))
    assert v1_v7 > SAME_VERSION_THRESHOLD
    assert classify(v1_v7).kind == "new_version"

    if PDF_BERT.exists():
        assert classify(jaccard(fp(PDF_V7), fp(PDF_BERT))).kind == "distinct"


@needs_pdfs
def test_real_diff_drops_most_noise_but_keeps_real_revisions():
    """v1→v7 的朴素 diff 有 60 多条，降噪后应当少得多，但不能少到 0。

    两头都要断言：全是噪声说明过滤过头（真改动被吞了），
    一条没滤说明归一化没起作用（行尾连字符还在制造伪差异）。
    """
    from app.versioning import diff_versions

    diff = diff_versions(PDF_V1, PDF_V7)
    assert diff.old_pages == 15 and diff.new_pages == 15
    assert diff.trivial_dropped > 20, "归一化基本没起作用"
    assert 5 < len(diff.changes) < diff.trivial_dropped + 5
    # "改了哪一节"必须标得出来，否则摘要只能说"第 4 页变了"，对读者没用
    labeled = [c for c in diff.changes if c.section]
    assert len(labeled) >= 0.8 * len(diff.changes)


@needs_pdfs
def test_reference_churn_is_flagged_so_summary_can_skip_it():
    from app.versioning import diff_versions

    diff = diff_versions(PDF_V1, PDF_V7)
    churn = [c for c in diff.changes if c.is_reference_churn]
    # 参考文献从 v1 的十几条涨到 v7 的四十多条，编号位移必然产生一堆变更
    assert churn, "参考文献节的差异没被标记出来"
    assert all(c.section.strip().lower().startswith("references") for c in churn)


@needs_pdfs
def test_arxiv_stamp_never_leaks_into_the_diff():
    """投稿戳自带版本号与日期，两版必然不同。

    留着它，每份论文对比都会凭空多出一条"删除 arXiv:1706.03762v1 [cs.CL] 12 Jun 2017"，
    而且它还会被当成章节名，让那条改动挂在 "arXiv:..." 名下。
    """
    from app.versioning import diff_versions

    from app.parsing.chunker import ARXIV_STAMP

    diff = diff_versions(PDF_V1, PDF_V7)
    for c in diff.changes:
        for s in [*c.old, *c.new]:
            assert not ARXIV_STAMP.search(s), s
        assert not ARXIV_STAMP.search(c.section), c.section
    # 但正文里真实引用 arXiv 预印本（"preprint arXiv:1511.06114, 2015."）不能删 ——
    # 那不是投稿戳，是参考文献
    blob = " ".join(s for c in diff.changes for s in [*c.old, *c.new]).lower()
    assert "arxiv:" in blob


@needs_pdfs
def test_diff_is_deterministic():
    """同一对文件跑两次必须一模一样 —— 演示时重跑给出不同结果最毁信任。"""
    from app.versioning import diff_versions

    a = diff_versions(PDF_V1, PDF_V7)
    b = diff_versions(PDF_V1, PDF_V7)
    assert a.stats == b.stats
    assert [(c.kind, c.new_pages, c.old) for c in a.changes] == [
        (c.kind, c.new_pages, c.old) for c in b.changes
    ]


# ---------------------------------------------------------------- 摘要降级


def test_summary_failure_is_reported_not_faked():
    """模型挂了就如实返回 ok=False。编一段"作者修订了第三节"是最坏的结果。"""
    from app.versioning import VersionDiff
    from app.versioning_summary import summarize_diff

    diff = VersionDiff(
        old_pages=6,
        new_pages=15,
        old_sentences=100,
        new_sentences=150,
        changes=[Change("replace", [4], [4], ["old one."], ["new one."], "3.2")],
    )

    def boom(*a, **k):
        raise RuntimeError("HTTP 502 Bad Gateway")

    out = summarize_diff(diff, llm_call=boom)
    assert out.ok is False
    assert out.items == []
    assert "502" in out.reason


def test_empty_diff_does_not_call_the_model():
    from app.versioning import VersionDiff
    from app.versioning_summary import summarize_diff

    called = []
    out = summarize_diff(
        VersionDiff(1, 1, 10, 10, changes=[]),
        llm_call=lambda *a, **k: called.append(1) or {"changes": []},
    )
    assert out.ok is True and out.items == []
    assert called == [], "没有差异还去问模型，纯属浪费 token 并制造幻觉机会"


def test_prompt_excludes_reference_churn():
    """参考文献改条号不是论文修订。喂进去，模型一定会说"作者大幅更新了文献综述"。"""
    from app.versioning import VersionDiff
    from app.versioning_summary import build_prompt

    diff = VersionDiff(
        old_pages=15,
        new_pages=15,
        old_sentences=10,
        new_sentences=10,
        changes=[
            Change("replace", [4], [4], ["old body sentence here."], ["new body sentence here."], "3.2"),
            Change("insert", [], [14], [], ["REFERENCE-CHURN-MARKER"], "References"),
        ],
    )
    prompt = build_prompt(diff)
    assert "new body sentence here." in prompt
    assert "REFERENCE-CHURN-MARKER" not in prompt
    assert "其中 1 处不在参考文献里" in prompt


@needs_pdfs
def test_prompt_trims_long_sentences():
    from app.versioning import diff_versions
    from app.versioning_summary import MAX_BODY_CHARS, build_prompt

    prompt = build_prompt(diff_versions(PDF_V1, PDF_V7), max_items=10)
    for line in prompt.splitlines():
        for tag in ("OLD: ", "NEW: "):
            if tag in line:
                assert len(line.split(tag, 1)[1]) <= MAX_BODY_CHARS + 1
