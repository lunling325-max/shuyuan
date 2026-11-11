"""端到端评测：引用能不能回原文、该拒答的时候有没有拒答。

为什么要有这个脚本：`docs/EXPERIMENTS.md` 里的记录大多是"改了什么 bug、改完什么样"，
那是**工程记录**，不是**方法评测**。评委真正会问的是两个数字 ——

1. **引用准确率**：系统说"这句出自第 3 页那个框"，那个框里到底有没有这句话？
2. **拒答正确率**：库里根本没有答案时，它有没有老实说"证据不足"？

题单是人工标注的（`scripts/eval_questions.json`，每题标 `expect=answer / refuse`），
本脚本只负责跑和核。核验是**独立**的：把服务器上的 PDF 下下来，用 pymupdf 自己做
版面分析 —— 先确认引文真在它说的那一页，再确认那个 bbox 盖住了引文的首词与尾词 ——
不采信接口自己报的 `exact` 字段。自己证明自己对，那不叫评测。

用法（项目根目录）：

    # 打线上服务（题单里的 PDF 会缓存到 build/eval-cache，只下不传）
    uv run --directory apps/api python ../../scripts/eval_qa.py \
        --base-url http://101.37.190.214:8017 --out build/eval-qa.md

    # 只跑其中几题
    uv run --directory apps/api python ../../scripts/eval_qa.py --only 3,7,11

退出码：全部通过 0，有失败项 1 —— 可以直接挂到 CI 或发布前检查。
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "scripts" / "eval_questions.json"


MAX_RETRY_ON_429 = 3


def _open(req, timeout: float):
    """发请求，撞上限流就按 `Retry-After` 等一等再试。

    服务端默认每分钟 30 次、突发 8 次（`app/api/ratelimit.py`）。整套评测 24 题
    是"人不可能这样点"的突发流量，会被正确地拦下来 —— 所以这里按规矩等，
    而不是去要求服务端为评测开后门。撞了一次限流会在日志里说一声，
    免得"等了 90 秒"被误读成"服务变慢了"。
    """
    for attempt in range(MAX_RETRY_ON_429 + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == MAX_RETRY_ON_429:
                raise
            wait = float(exc.headers.get("Retry-After") or 5)
            print(f"    撞上限流，等 {wait:.0f} 秒（第 {attempt + 1} 次）", flush=True)
            time.sleep(wait)


def _get(url: str, timeout: float = 120.0) -> bytes:
    with _open(url, timeout) as resp:
        return resp.read()


def _post_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with _open(req, timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _flatten(text: str) -> str:
    """压成单行小写：编码差异不该被算成"引用错了"。"""
    return " ".join((text or "").split()).lower()


@dataclass
class AnchorCheck:
    page: int
    verified: bool
    why: str
    in_page: bool
    box_ok: bool
    server_says_exact: bool
    quote: str


@dataclass
class Result:
    index: int
    paper: str
    question: str
    expect: str
    ok: bool
    why: str
    claims: int = 0
    supported: int = 0
    keywords: list[str] = field(default_factory=list)
    anchors: list[AnchorCheck] = field(default_factory=list)
    took_ms: int = 0
    error: str = ""


class Library:
    """把题单里的 arXiv 号翻译成接口 id，并保证本地有一份 PDF 用来独立核验。"""

    def __init__(self, base: str, cache: Path) -> None:
        self.base = base.rstrip("/")
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        papers = json.loads(_get(f"{self.base}/api/papers").decode("utf-8"))
        self.by_arxiv: dict[str, dict] = {}
        for paper in papers:
            key = paper.get("arxiv_id") or ""
            if not key:
                continue
            kept = self.by_arxiv.get(key)
            # 同一篇有多版时，优先用有文字层的那一版：扫描件的引文带 OCR 误差，
            # 拿它去核"引用准不准"会把 OCR 的错算成系统的错。
            rank = (0 if paper.get("scanned") else 1, paper.get("arxiv_version") or 0)
            if kept is None or rank > kept["_rank"]:
                paper["_rank"] = rank
                self.by_arxiv[key] = paper

    def paper(self, arxiv_id: str) -> dict:
        found = self.by_arxiv.get(arxiv_id)
        if found is None:
            raise SystemExit(f"题单里的 arXiv:{arxiv_id} 不在库里，先上传或修正题单")
        return found

    def pdf(self, paper: dict) -> Path:
        path = self.cache / f"{paper['id']}.pdf"
        if not path.exists():
            path.write_bytes(_get(f"{self.base}/api/papers/{paper['id']}/file"))
        return path


def _stream(text: str) -> str:
    """只留字母数字、转小写。

    为什么不按词比：pymupdf 的 `get_text("words")` 是按空白切的粗分词，
    `H=1024,` 是一整词、`con tains` 会被拆成两词、纯括号是一词。引文里
    却是 `H=1024` 这样干净的一段。两边各自切词再比，分母永远对不齐，
    于是"引文确实在页上"会被误判成"找不到"。压成字符流以后，标点、
    连字符断行、词被拆开这些差异全都不再影响定位。
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _page_stream(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect]]:
    """页面文字压成字符流，并逐字符记住它属于哪个词的框 —— 位置从这里回溯。"""
    chars: list[str] = []
    rects: list[pymupdf.Rect] = []
    for w in page.get_text("words"):
        rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
        for ch in w[4].lower():
            if ch.isalnum():
                chars.append(ch)
                rects.append(rect)
    return "".join(chars), rects


def check_anchors(doc: pymupdf.Document, anchors: list[dict]) -> list[AnchorCheck]:
    """每条引用都回 PDF 独立核一次，不采信接口自报的 exact。

    查两件事，和界面上那句"原文在第 N 页这个框里"一一对应：

    1. **引文真的在这一页**：引文的字符流能在本页字符流里原样找到；
    2. **框真的框住了它**：引文首字符和末字符所在的词，其框落在 bbox 内（容差 3pt）。

    为什么不用 "bbox 抠出的文字 == 引文"：一段话跨行、又不左对齐时，一个矩形
    必然兜进邻行的字；按矩形反抠会把"框得对"判成"框得错"。首尾落点才是这个框
    真正承诺的东西。
    """
    out: list[AnchorCheck] = []
    for anchor in anchors:
        page_no = int(anchor.get("page") or 0)
        quote = anchor.get("source_text") or ""
        bbox = anchor.get("bbox") or [0, 0, 0, 0]
        base = {
            "page": page_no,
            "server_says_exact": bool(anchor.get("exact")),
            "quote": " ".join(quote.split())[:90],
        }
        needle = _stream(quote)
        if not (1 <= page_no <= doc.page_count) or not needle:
            out.append(
                AnchorCheck(
                    **base,
                    verified=False,
                    why="页码越界或引文为空",
                    in_page=False,
                    box_ok=False,
                )
            )
            continue

        stream, rects = _page_stream(doc[page_no - 1])
        start = stream.find(needle)
        in_page = start >= 0
        box_ok = False
        if in_page:
            box = pymupdf.Rect(*bbox)
            box = pymupdf.Rect(box.x0 - 3, box.y0 - 3, box.x1 + 3, box.y1 + 3)
            first, last = rects[start], rects[start + len(needle) - 1]
            box_ok = box.contains(first) and box.contains(last)
            why = "引文在本页，框覆盖首尾" if box_ok else "引文在本页，但框没盖住首尾字"
        else:
            # 找不到时给一个"最像的一段占引文多少"的数字，方便分辨是引文换了页，
            # 还是这一页的字被抽得乱七八糟。不拿它当通过依据。
            hit = difflib.SequenceMatcher(None, needle, stream, autojunk=False)
            best = hit.find_longest_match(0, len(needle), 0, len(stream))
            why = f"本页找不到这句话（最长连续重合 {best.size}/{len(needle)} 字）"
        out.append(
            AnchorCheck(
                **base,
                verified=in_page and box_ok,
                why=why,
                in_page=in_page,
                box_ok=box_ok,
            )
        )
    return out


def negative_control(
    pool: list[tuple[pymupdf.Document, dict]], limit: int = 6
) -> list[tuple[str, bool, str]]:
    """反向对照：把真引用故意改坏，核验器必须判它没过。

    为什么非做不可：一个**永远通过**的核验器也能给出 100% 的引用准确率，
    而它什么也没证明。所以每轮评测都抽几条真引用，分别做两种破坏 ——

    - `shift`：框整体平移 300pt，引文还在这一页，但框盖的是别处的字；
    - `text`：引文尾部续编一句页面根本不存在的话。

    两种都该被判没过。若有一条被"改坏了还通过"，那这一轮的引用准确率不作数。
    """
    out: list[tuple[str, bool, str]] = []
    for doc, anchor in pool[:limit]:
        quote = anchor.get("source_text") or ""
        bbox = list(anchor.get("bbox") or [0, 0, 0, 0])

        shifted = dict(anchor)
        shifted["bbox"] = [bbox[0] + 300, bbox[1] + 300, bbox[2] + 300, bbox[3] + 300]
        got = check_anchors(doc, [shifted])[0]
        out.append(("shift", got.verified, got.why))

        faked = dict(anchor)
        faked["source_text"] = quote + " This sentence was never written in this paper."
        got = check_anchors(doc, [faked])[0]
        out.append(("text", got.verified, got.why))
    return out


def judge(result: Result, body: dict) -> None:
    """按题单标注判这一题过没过。"""
    claims = body.get("claims") or []
    supported = [c for c in claims if c.get("status") == "supported"]
    result.claims = len(claims)
    result.supported = len(supported)

    if result.expect == "answer":
        if not supported:
            result.ok, result.why = False, "该答却没给出可溯源结论"
            return
        text = _flatten(" ".join(c.get("text", "") for c in supported))
        # 题单里的关键词允许写 "8|八"：模型说"八个注意力头"和"8 个注意力头"都对，
        # 只认数字会把答对的题判成答错。
        missing = []
        for entry in result.keywords:
            forms = [_flatten(x) for x in entry.split("|") if x.strip()]
            if forms and not any(f in text for f in forms):
                missing.append(entry)
        if missing:
            result.ok, result.why = False, "结论里缺少关键词：" + "、".join(missing)
            return
        bad = [a for a in result.anchors if not a.verified]
        if bad:
            pages = "、".join(f"p.{a.page}（{a.why}）" for a in bad)
            result.ok, result.why = False, f"{len(bad)} 条引用未通过核验：{pages}"
            return
        result.ok = True
        result.why = f"{len(supported)} 条结论，{len(result.anchors)} 条引用全部落地"
    else:
        if supported:
            result.ok = False
            result.why = f"库里没有答案却给了 {len(supported)} 条可溯源结论"
            return
        result.ok = True
        result.why = "证据不足，未给结论" if claims else "未给出结论"


def main() -> int:
    # 逐题结果里有 ✓ / ✗ / 中文题面：Windows 控制台默认 GBK，编不出来就
    # UnicodeEncodeError，跑完一遍却什么都没打印、还把逐题循环从中途打断。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="端到端评测：引用溯源 + 拒答")
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--cache-dir", default=str(ROOT / "build" / "eval-cache"))
    parser.add_argument("--out", default="", help="把结果写成 Markdown 表格")
    parser.add_argument(
        "--json", dest="json_out", default="", help="把原始结果写成 JSON"
    )
    parser.add_argument("--only", default="", help="只跑指定题号，逗号分隔，如 1,3,9")
    args = parser.parse_args()

    # 相对路径按**仓库根**算，不按当时的工作目录算。README 里的用法是
    # `uv run --directory apps/api python ../../scripts/eval_qa.py --out build/eval-qa.md`，
    # 而 --directory 会把工作目录换成 apps/api —— 照工作目录走，文件会悄悄落到
    # apps/api/build/ 里，人按 README 去 build/ 找是空的。
    for flag in ("out", "json_out"):
        value = getattr(args, flag)
        if value and not Path(value).is_absolute():
            setattr(args, flag, str(ROOT / value))

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    items = list(
        enumerate(spec["questions"], start=1)
    )  # 题号按题单固定，--only 才不会错位
    if args.only:
        wanted = {int(x) for x in args.only.replace("，", ",").split(",") if x.strip()}
        items = [(i, q) for i, q in items if i in wanted]

    lib = Library(args.base_url, Path(args.cache_dir))
    docs: dict[str, pymupdf.Document] = {}
    results: list[Result] = []
    pool: list[tuple[pymupdf.Document, dict]] = []  # 核验通过的引用，留给反向对照

    for index, item in items:
        paper = lib.paper(item["paper"])
        result = Result(
            index=index,
            paper=f"arXiv:{item['paper']}",
            question=item["question"],
            expect=item["expect"],
            ok=False,
            why="",
        )
        result.keywords = item.get("must_contain") or []
        try:
            body = _post_json(
                f"{lib.base}/api/papers/{paper['id']}/qa",
                {"question": item["question"], "mode": "answer"},
            )
        except Exception as exc:  # noqa: BLE001 —— 一题挂了不影响整轮，但要记下来
            result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            print(
                f"[{index:>2}] ✗ {item['question'][:40]} —— 请求失败：{result.error[:60]}"
            )
            continue

        result.took_ms = int(body.get("took_ms") or 0)
        if paper["id"] not in docs:
            docs[paper["id"]] = pymupdf.open(str(lib.pdf(paper)))
        anchors = [
            a for c in (body.get("claims") or []) for a in (c.get("anchors") or [])
        ]
        result.anchors = check_anchors(docs[paper["id"]], anchors)
        pool += [
            (docs[paper["id"]], a)
            for a, chk in zip(anchors, result.anchors)
            if chk.verified
        ]
        judge(result, body)
        results.append(result)
        mark = "✓" if result.ok else "✗"
        print(f"[{index:>2}] {mark} {item['question'][:44]} —— {result.why}")

    # ---------------------------------------------------------------- 汇总
    checked = [r for r in results if not r.error]
    anchors = [a for r in results for a in r.anchors]
    verified = [a for a in anchors if a.verified]
    answered = [r for r in results if r.expect == "answer"]
    refused = [r for r in results if r.expect == "refuse"]
    latencies = sorted(r.took_ms for r in checked if r.took_ms)

    def pct(part: int, whole: int) -> str:
        return f"{part / whole * 100:.1f}%" if whole else "—"

    lines = [
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 题数（答 / 拒） | {len(results)}（{len(answered)} / {len(refused)}） |",
        f"| 通过 | {sum(1 for r in checked if r.ok)} / {len(checked)} |",
        f"| 引用准确率（回 PDF 独立核验） | {len(verified)} / {len(anchors)} = {pct(len(verified), len(anchors))} |",
        f"| ├ 引文确实在所指页 | {sum(1 for a in anchors if a.in_page)} / {len(anchors)} = {pct(sum(1 for a in anchors if a.in_page), len(anchors))} |",
        f"| └ bbox 覆盖引文首尾词 | {sum(1 for a in anchors if a.box_ok)} / {len(anchors)} = {pct(sum(1 for a in anchors if a.box_ok), len(anchors))} |",
        f"| 接口自报逐字命中（仅供参考） | {sum(1 for a in anchors if a.server_says_exact)} / {len(anchors)} |",
        f"| 该答的题给出可溯源结论 | {sum(1 for r in answered if r.ok)} / {len(answered)} |",
        f"| 该拒的题拒住了 | {sum(1 for r in refused if r.ok)} / {len(refused)} |",
    ]
    # 反向对照：从通过的引用里均匀抽几条，改坏它们，核验器必须拒绝
    sample = pool[:: max(1, len(pool) // 6)] if pool else []
    controls = negative_control(sample)
    if controls:
        caught = sum(1 for _kind, verified, _why in controls if not verified)
        lines.append(
            f"| 反向对照（改坏引用后应被拒） | {caught} / {len(controls)} 被拒 |"
            if caught == len(controls)
            else f"| 反向对照（改坏引用后应被拒） | {caught} / {len(controls)} —— **核验器失效，本轮引用数据不作数** |"
        )
    if latencies:
        lines += [
            f"| 单题耗时 中位数 / 最长 | {int(statistics.median(latencies))} ms / {latencies[-1]} ms |",
        ]
    summary = "\n".join(lines)
    print(summary)

    if args.out:
        detail = [
            "## 逐题结果",
            "",
            "| # | 论文 | 问题 | 期望 | 结果 | 引用 | 说明 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            good = "✓" if r.ok else "✗"
            refs = (
                f"{sum(1 for a in r.anchors if a.verified)}/{len(r.anchors)}"
                if r.anchors
                else "—"
            )
            why = r.error or r.why
            detail.append(
                f"| {r.index} | {r.paper} | {r.question} | {r.expect} | {good} | {refs} | {why} |"
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(detail) + "\n\n" + summary + "\n", encoding="utf-8")
        print(f"\n已写入 {out}")

    if args.json_out:
        payload = [
            {
                "index": r.index,
                "paper": r.paper,
                "question": r.question,
                "expect": r.expect,
                "ok": r.ok,
                "why": r.why,
                "took_ms": r.took_ms,
                "error": r.error,
                "anchors": [a.__dict__ for a in r.anchors],
            }
            for r in results
        ]
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
