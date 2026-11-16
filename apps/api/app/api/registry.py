"""文档注册表：上传一次，解析一次，之后所有请求共用同一份索引。

为什么要单独一层，而不是在路由里现算？
1. **切块与建索引是重活**。一篇 15 页论文建词级索引要几百毫秒，
   541 页教材要几十秒。每次问答都重建，演示时页面会明显卡住。
2. **id 用内容哈希，不是自增计数**。同一份 PDF 传两次必须落到同一个 id ——
   这是后面"版本谱系"能工作的前提，也顺手挡住重复上传。
3. **懒构建**。上传时只切块（要立刻把块数返回给前端），
   词级索引和 BM25 等到第一次真正检索时才建。

线程安全：FastAPI 的同步路由跑在线程池里，所以构建过程加锁。
锁的粒度是**每篇文档一把**，不是全局一把 —— 否则两篇文档的解析会互相排队。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from app.parsing.chunker import Chunk, chunk_pdf
from app.parsing.layout import STAMP, line_of, page_lines
from app.parsing.text_index import TextIndex
from app.retrieval.bm25 import BM25
from app.versioning import Relation, classify, jaccard, shingles


class PaperNotFound(KeyError):
    pass


class PaperBusy(OSError):
    """文件还被别的进程占着，这次删不掉。"""


def _unlink_retry(path: Path, *, attempts: int = 6) -> None:
    """Windows 上删文件会撞 WinError 32（文件被占用），而且是**偶发**的。

    实测：刚上传几秒的文件，删的时候报"另一个程序正在使用此文件"——
    占着它的不是我们自己的代码（`TextIndex` 读完就 close），而是系统的
    杀毒扫描/索引器在文件刚落地时短暂持有句柄。Linux 上删已打开的文件
    从不报错，所以这个坑只有 Windows 开发机会踩到。

    不重试的后果比报错更难看：内存里已经从库里摘掉了，磁盘上的 PDF 还在，
    下次启动"裸 PDF 扫描"又把它捞回来 —— 用户看到的就是"删了又回来"。
    """
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            if i == attempts - 1:
                raise PaperBusy(str(path)) from exc
            time.sleep(0.12 * (i + 1))


def _ocr_hook():
    """有 OCR 引擎就把页级钩子交给切块器，没有就返回 None。

    在注册表这一层做判断，而不是在切块器里：切块器若直接去找引擎，
    每跑一次单元测试都要依赖本机装没装 RapidOCR，测试结果就不可复现了。
    `ocr_enabled()` 带缓存，几百页的教材不会探几百次。
    """
    from app.parsing.scan import ocr_blocks_for_page, ocr_enabled

    return ocr_blocks_for_page if ocr_enabled() else None


def _chunk(path: Path) -> list[Chunk]:
    """先把扫描页并发识别掉，再交给切块器。

    顺序很重要：`extract_paras` 是逐页同步调 OCR 钩子的，串行识别 15 页扫描件
    要两三分钟，上传接口就那么干等着。`warm()` 先把这些页并发识别完，
    切块器随后逐页取用时全是缓存命中 —— 实测同一份 15 页扫描件上传
    2m39s → 1m42s。非扫描件这一趟什么都不做（只读几页文字层），可以忽略。
    """
    from app.parsing.scan import warm

    warm(str(path))
    return chunk_pdf(str(path), ocr=_ocr_hook())


_ABSTRACT_SECTION = re.compile(r"^(abstract|summary|摘要)", re.IGNORECASE)


def _strip_author_blob(text: str, authors, affiliations, title: str = "") -> str:
    """把摘要开头粘着的「作者 / 机构 / 邮箱 / 标题连写 / 角标」一层层剥掉。

    为什么是一轮循环而不是几步各跑一次：剥掉一个作者，后面常常又是下一个作者，
    再后面是邮箱块、又是机构 —— 实测 v1 那份有 8 个作者 + 4 个机构 + 一组邮箱，
    分步各跑一次只能剥掉最外面一层。

    判据全部是"确定的事实"（已知名字、已知机构、标题本身、含 @ 的 token），
    不做"这看起来像人名"这类猜测 —— 猜错会把摘要开头吃掉。
    """
    names = [n.strip() for n in (list(authors or []) + list(affiliations or []))
             if n and n.strip()]
    title_key = re.sub(r"\s+", "", (title or "")).lower()
    for _ in range(40):
        before = text
        for name in names:
            if text.startswith(name):
                text = text[len(name):].lstrip(" ,;·-∗†‡*")
        if title_key:
            head = text.split(" ", 1)[0]
            if head and re.sub(r"\s+", "", head).lower().startswith(title_key):
                text = text[len(head):].lstrip(" ,;·-∗†‡*")
        toks = text.split()
        if toks and any(m in toks[0] for m in ("@", "{", "}", "∗", "†", "‡", "*")):
            text = " ".join(toks[1:])
        text = text.lstrip("∗†‡*·,; {")
        if text == before:
            break
    return text


def _abstract_of(chunks: list[Chunk], authors: list[str] | None = None,
                 affiliations: list[str] | None = None, title: str = "") -> str:
    """从切块里取摘要正文；没有就返回空串，**不要从别处凑一段**。

    为什么从切块取而不是回版面再解析一遍：切块里的章节名已经由解析器认过，
    这里再认一次等于把同一件事做两遍，两份结果还可能不一致。

    `authors` 用来剥掉粘在摘要开头的人名：BERT 那份的版面把作者行并进了
    Abstract 这一块，抽出来是 `Jacob Devlin Ming-Wei Chang … Google AI Language
    We introduce a new language representation model…`。剥的判据只有一条 ——
    **开头连续出现已知作者名**，多一个字都不动。
    """
    for chunk in chunks:
        if _ABSTRACT_SECTION.match((chunk.section or "").strip()):
            text = " ".join(chunk.text.split())
            text = _strip_author_blob(text, authors, affiliations, title)
            return text[:600]
    return ""


def _front_matter(path: Path):
    """作者与机构。解析出错就当没认出来 —— 它是加分项，不该让上传失败。"""
    from app.parsing.frontmatter import FrontMatter, extract_front_matter

    try:
        return extract_front_matter(path)
    except Exception:  # noqa: BLE001
        return FrontMatter()


def _clean_filename(filename: str) -> str:
    """只留文件名本身：路径成分与不可打印字符一律去掉。

    上传时客户端送来什么就是什么 —— 实测能送进 `../../evil.pdf`，
    它会一路存进 sidecar，再原样拼成
    `Content-Disposition: attachment; filename="../../evil.pdf"`。
    浏览器自己会剥掉路径，所以这不是漏洞；但"库里存着一个带路径的文件名"
    迟早有人当成路径去用，而且文献库里显示 `../../evil.pdf` 也说不通。
    """
    name = Path(str(filename).replace("\\", "/")).name
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name or set(name) <= {"."}:
        return "untitled.pdf"
    return name[:160]


# 标题识别规则的版本号，写进 sidecar。改动识别逻辑就加一 —— 老 sidecar 里存着的
# 是旧规则算出来的标题（比如 2 版之前的 "DISTFLASHATTN:Distributed Memory-efficientAttention"），
# 不重算的话它会永远留在库里，代码修好了也白修。
#   1: 按 span 挑"字号恰好等于最大值"，行间用空格拼
#   2: 按行挑（容忍 0.8pt 字号差），行内拼 span 不加空格
#   3: 版面标题全大写、而元数据是逐字相同的正常写法时，采用元数据的大小写
TITLE_SCHEMA = 3


def _title_shaped(text: str) -> bool:
    """OCR 的一行字**形式上**像不像标题：够长、不是邮箱、不是符号堆。

    只做形式判断，位置和宽度由调用方负责 —— 这样这段逻辑在文本层那条路上
    也用得上，两边不必各写一套。
    """
    t = " ".join(text.split())
    if len(t) < 8 or "@" in t:
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= len(t) * 0.6


@dataclass
class Paper:
    id: str
    filename: str
    title: str
    pages: int
    chunks: list[Chunk]
    path: Path
    uploaded_at: str
    created_at: str = ""  # PDF 元数据里的生成日期（YYYY-MM-DD），判断谁更新靠它
    arxiv_id: str = ""  # 如 "1706.03762"（不带版本号）；引用图谱靠它认预印本
    arxiv_version: int = 0  # arXiv 的 vN（v7 → 7）；0 表示取不到，表示"不知道"而不是"v1"
    # 首页投稿戳上印的日期（YYYY-MM-DD）。它和 created_at **不是一回事**：
    # created_at 是"这份 PDF 什么时候导出的"，这个是"这一版什么时候投到 arXiv 的"。
    # 2025 年以后 arXiv 换了 GenPDF，导出时不再写 creationDate/modDate，
    # 一批新论文的 created_at 全是空串，界面上只能写"生成日期未知"。
    # 戳上的日期是 arXiv 自己印的，比元数据可靠，所以另立一栏，不去冒充 created_at。
    arxiv_date: str = ""
    scanned: bool = False  # 首页没有文本层：正文与标题都得靠 OCR 认
    # 首页版面上认出来的作者与机构（模块一要求）。认不出就是空列表 ——
    # 扫描件首页没有文本层，这里宁可不写，也不从别处猜一个机构填上去。
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    # 摘要：直接从切块里取（章节名命中 Abstract 的第一块），不从版面再猜一次。
    # 解析器本来就认得这个标题，只是以前只拿它当"作者区块的右边界"，没留下来。
    abstract: str = ""
    _index: TextIndex | None = field(default=None, repr=False)
    _bm25: BM25 | None = field(default=None, repr=False)
    _fingerprint: set[str] | None = field(default=None, repr=False)
    _refs: list | None = field(default=None, repr=False)
    _visuals: object | None = field(default=None, repr=False)
    _formulas: object | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def fingerprint(self) -> set[str]:
        """用于判断"是不是同一篇的不同版本"的 5-gram 指纹。

        用块文本而不是整篇原文：块已经去过页眉页脚，指纹比裸全文更干净
        （页眉里那行 arXiv 戳每版都不一样，留在指纹里会白白拉低相似度）。
        """
        if self._fingerprint is None:
            with self._lock:
                if self._fingerprint is None:
                    self._fingerprint = shingles(" ".join(c.text for c in self.chunks))
        return self._fingerprint

    @property
    def references(self) -> list:
        """参考文献条目（懒解析 + 缓存）。

        竞态说明：这里缓存的是**纯数据**，两个线程同时解析只会得到两份等价列表，
        后写入的那份胜出即可 —— 不像 chunk.index 那样要求全进程唯一编号，
        所以不必像 text_index 那样双检加锁。加锁仍然是为了别让同一篇被解析三遍。
        """
        if self._refs is None:
            with self._lock:
                if self._refs is None:
                    from app.references import extract_references

                    self._refs = extract_references(
                        self.chunks, self.path, ocr=_ocr_hook()
                    )
        return self._refs

    @property
    def visuals(self):
        """表格与插图（懒解析 + 进程内缓存，磁盘上还有一份）。

        为什么不放在上传时解析：`find_tables()` 实测 0.29 秒/页，75 页的 GPT-3
        要 22 秒 —— 上传接口干等这一趟不值得，用户也不一定马上要看图表。
        第一次点开"图表"标签时再算，算完 `visuals.analyze` 自己会落盘。
        """
        if self._visuals is None:
            with self._lock:
                if self._visuals is None:
                    from app.parsing.visuals import analyze

                    self._visuals = analyze(self.path)
        return self._visuals

    @property
    def formulas(self):
        """带编号的公式（懒解析 + 进程内缓存，磁盘上还有一份）。

        和图表一样是懒解析：`formulas.analyze` 要整篇走一遍 40 页，
        上传时干等它没有意义。算完自己落盘，重启也不用再算。
        """
        if self._formulas is None:
            with self._lock:
                if self._formulas is None:
                    from app.parsing.formulas import analyze

                    self._formulas = analyze(self.path)
        return self._formulas

    @property
    def text_index(self) -> TextIndex:
        """词级索引：溯源校验用它把引用文本落回页码 + 坐标。"""
        if self._index is None:
            with self._lock:
                if self._index is None:  # 双检：别让两个线程各建一份
                    self._index = TextIndex(str(self.path))
        return self._index

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            with self._lock:
                if self._bm25 is None:
                    self._bm25 = BM25(self.chunks)
        return self._bm25


def order_versions(a: Paper, b: Paper) -> tuple[Paper, Paper, str]:
    """谁旧谁新，外加一条"凭什么这么判"，前端要照实说。

    优先级：
    1. **同一篇（arXiv 号一致）且两边都取得到版本号** → vN 小的在前。这是唯一
       能顶住"扫描件没有元数据日期""两版同一天导出"的判据。
    2. 两份 PDF 生成日期都有 → 日期早的在前。
    3. 兜底用上传时间，并且明确标成 `uploaded_at` —— 它只说明"你什么时候传的"，
       和论文本身的新旧无关，界面上不该把它讲成版本先后。

    放在 registry 这一层而不是某个路由里：HTTP 接口和 Agent 工具都要判新旧，
    之前各写一遍，扫描件在两条路上的新旧就是相反的。
    """
    if a.arxiv_id and a.arxiv_id == b.arxiv_id and a.arxiv_version and b.arxiv_version:
        if a.arxiv_version != b.arxiv_version:
            older_first = a.arxiv_version < b.arxiv_version
            return (a, b, "arxiv_version") if older_first else (b, a, "arxiv_version")
    if a.created_at and b.created_at and a.created_at != b.created_at:
        return (a, b, "date") if a.created_at < b.created_at else (b, a, "date")
    # 这里**故意不用** arxiv_date：它只在前端展示时兜底。
    # 一边有 created_at、一边只有投稿戳日期，等于拿"PDF 导出时间"去比"arXiv 投稿时间"，
    # 比出来的先后没法解释；更要紧的是这一层的结论会被上传页拿去建议"删哪一份"，
    # 凭一份半残的证据劝人删文件，比老实说"判不出来"危险得多。
    if (a.created_at or a.uploaded_at) <= (b.created_at or b.uploaded_at):
        return a, b, "uploaded_at"
    return b, a, "uploaded_at"


class PaperRegistry:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._papers: dict[str, Paper] = {}
        self._lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self._graph_cache: tuple[tuple, int, object] | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        """重启后把资料库读回来。

        内存注册表一重启就空，演示时最丢人的事莫过于此：上传过的论文凭空消失。
        所以每篇文档落盘时旁边写一个同名的 .json 记住文件名与上传时间，
        启动时按 sidecar 重建（PDF 文件本身就是唯一的数据源，不会不同步）。
        解析是在启动时一次性的成本：15 页论文几十毫秒，541 页教材几秒。
        """
        for meta_path in sorted(self.store_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pdf_path = self.store_dir / f"{meta['id']}.pdf"
                if not pdf_path.exists():
                    continue
                chunks = _chunk(pdf_path)
                if not chunks:
                    continue
                filename = meta.get("filename") or pdf_path.name
                title = meta.get("title") or self._title(pdf_path, pdf_path.name)
                # 以前存下来的标题可能是文件名兜底（1706.03762v1_scan 就是那么来的）。
                # 标题等于文件主名就说明当初没认出来，重算一次 —— 现在扫描件会走 OCR，
                # 同一份文件能拿到真标题。算不出来还是原样，不会把好标题改坏。
                # 另外：识别规则升级过（TITLE_SCHEMA）的旧 sidecar 也重算一次，
                # 否则修好了规则、库里还是那串连体字，等于没修。
                stale = int(meta.get("title_v") or 0) < TITLE_SCHEMA
                if title == Path(filename).stem or stale:
                    better = self._title(pdf_path, filename)
                    if better and better != title:
                        title = better
                # 作者与机构是后加的字段：老 sidecar 里没有这两栏，就地补算一次。
                # 判据是"键在不在"而不是"列表空不空" —— 扫描件本来就该是空列表，
                # 每次启动都重算一遍纯属浪费（541 页教材那趟要读首页）。
                no_front = "authors" not in meta or "affiliations" not in meta
                front = _front_matter(pdf_path) if no_front else None
                paper = Paper(
                    id=meta["id"],
                    filename=filename,
                    title=title,
                    pages=meta.get("pages") or self._page_count(pdf_path),
                    chunks=chunks,
                    path=pdf_path,
                    uploaded_at=meta.get("uploaded_at") or "",
                    created_at=meta.get("created_at") or self._pdf_date(pdf_path),
                    # 兜底要用**原始文件名**：落盘的 PDF 叫 `<hash>.pdf`，里面没有 arXiv 号
                    arxiv_id=meta.get("arxiv_id")
                    or self._arxiv_id(pdf_path, filename),
                    arxiv_version=meta.get("arxiv_version")
                    or self._arxiv_version(pdf_path, filename),
                    arxiv_date=meta.get("arxiv_date") or self._arxiv_date(pdf_path),
                    scanned=bool(meta.get("scanned", False)) or self._is_scanned(pdf_path),
                    authors=meta.get("authors") if not no_front else front.authors,
                    affiliations=meta.get("affiliations")
                    if not no_front
                    else front.affiliations,
                    # 摘要同样是后加的字段：老 sidecar 没有就地补算。
                    # 从**已经切好的块**里取，不额外读版面 —— 启动时重建 15 篇也只多几毫秒。
                    abstract=meta.get("abstract") or _abstract_of(
                        chunks,
                        meta.get("authors") if not no_front else front.authors,
                        meta.get("affiliations") if not no_front else front.affiliations,
                    ),
                )
                self._papers[paper.id] = paper
                if no_front or "abstract" not in meta or (paper.title, paper.arxiv_version, paper.scanned, paper.arxiv_date) != (
                    meta.get("title"),
                    meta.get("arxiv_version") or 0,
                    bool(meta.get("scanned", False)),
                    meta.get("arxiv_date") or "",
                ) or int(meta.get("title_v") or 0) < TITLE_SCHEMA:
                    self._write_meta(paper)  # 补上新字段：下次启动就不用再算一遍
            except Exception:  # noqa: BLE001 —— 单个坏文件不该拖垮整个启动
                continue

        # 只认 sidecar 会有个洞：手动丢进目录的 PDF 永远加载不出来。
        # 所以再扫一遍裸 PDF，用文件名兜底 —— 少一点元数据，总好过"文件在库里却看不见"。
        for pdf_path in sorted(self.store_dir.glob("*.pdf")):
            if pdf_path.stem in self._papers:
                continue
            try:
                chunks = _chunk(pdf_path)
                if not chunks:
                    continue
                front = _front_matter(pdf_path)
                self._papers[pdf_path.stem] = Paper(
                    id=pdf_path.stem,
                    filename=pdf_path.name,
                    title=self._title(pdf_path, pdf_path.name),
                    pages=self._page_count(pdf_path),
                    chunks=chunks,
                    path=pdf_path,
                    uploaded_at="",
                    created_at=self._pdf_date(pdf_path),
                    arxiv_id=self._arxiv_id(pdf_path, pdf_path.name),
                    arxiv_version=self._arxiv_version(pdf_path, pdf_path.name),
                    arxiv_date=self._arxiv_date(pdf_path),
                    scanned=self._is_scanned(pdf_path),
                    authors=front.authors,
                    affiliations=front.affiliations,
                )
            except Exception:  # noqa: BLE001
                continue

    def remove(self, paper_id: str) -> Paper:
        """从库里删掉一篇：内存、sidecar、落盘的 PDF 三处一起删。

        三处必须同步，缺一处就会"删了又回来"：只删内存，重启时 sidecar 把它
        复活；只删 sidecar，裸 PDF 扫描那一步照样能把它捞回来。
        因此判定"删干净了"的标准是 store_dir 里既没有 `<id>.pdf` 也没有 `<id>.json`。

        删除是不可逆的，所以只删**存放在 store_dir 里的副本** —— 万一将来有人把
        原始文件路径塞进 Paper.path，也不该被这个接口连坐删掉。
        """
        with self._lock:
            paper = self._papers.pop(paper_id, None)
        if paper is None:
            raise PaperNotFound(paper_id)

        if paper.path.parent == self.store_dir:
            try:
                _unlink_retry(paper.path)
            except PaperBusy:
                # 删不掉就**撤销内存里的删除**：宁可整个操作失败、库里状态一致，
                # 也不要留下一份"内存没有、磁盘还在"的幽灵文件
                with self._lock:
                    self._papers[paper_id] = paper
                raise
        (self.store_dir / f"{paper_id}.json").unlink(missing_ok=True)
        # 图谱缓存的键是文档 id 集合，少一篇自然失配，不用手动清
        return paper

    def find_similar(self, paper: Paper) -> tuple[Paper, Relation] | None:
        """在库里找与它最像的另一篇。判定依据是正文内容，不是文件名。

        这正是官方用例 3.1 要的东西：把 v7 传进来，系统得自己认出
        "这跟库里那篇是同一篇的不同版本"，然后走版本谱系而不是拒收。
        """
        best: tuple[Paper, Relation] | None = None
        mine = paper.fingerprint
        for other in self._papers.values():
            if other.id == paper.id:
                continue
            relation = classify(jaccard(mine, other.fingerprint))
            if best is None or relation.similarity > best[1].similarity:
                best = (other, relation)
        if best is None or best[1].kind == "distinct":
            return None
        return best

    def graph(self, *, max_external: int = 24):
        """引用图谱（带缓存）。

        建图要把每篇的参考文献都抽一遍：五篇样本 0.5 秒，换成几十篇就是好几秒。
        图谱又只在"库变了"之后才变，所以按库内容缓存 —— 键就是文档 id 集合，
        新上传会自然让它失效，不需要另外记得去清。
        """
        from app.citegraph import build_graph

        key = tuple(sorted((p.id, p.arxiv_id) for p in self._papers.values()))
        cached = self._graph_cache
        if cached is not None and cached[0] == key and cached[1] == max_external:
            return cached[2]

        with self._graph_lock:
            cached = self._graph_cache
            if cached is not None and cached[0] == key and cached[1] == max_external:
                return cached[2]
            built = build_graph(list(self._papers.values()), max_external=max_external)
            self._graph_cache = (key, max_external, built)
            return built

    def lookup(self, data: bytes) -> Paper | None:
        """这份字节已经入库了就返回那篇，否则 None。

        存在的意义是让**重传同一个文件**和**传一篇新版本**分开：
        `add` 对内容哈希相同的输入会直接返回库里那篇，调用方若把它当成
        "刚上传的新对象"再走一遍去重策略，就可能把用户在库里的原始文件删掉。
        """
        digest = hashlib.sha256(data).hexdigest()[:12]
        return self._papers.get(digest)

    def get(self, paper_id: str) -> Paper:
        try:
            return self._papers[paper_id]
        except KeyError as exc:
            raise PaperNotFound(paper_id) from exc

    def all(self) -> list[Paper]:
        return sorted(self._papers.values(), key=lambda p: p.uploaded_at, reverse=True)

    def add(self, data: bytes, filename: str) -> Paper:
        """落盘 + 切块。同一份内容重复上传直接返回已有记录。"""
        filename = _clean_filename(filename)
        digest = hashlib.sha256(data).hexdigest()[:12]
        with self._lock:
            if digest in self._papers:
                return self._papers[digest]

        path = self.store_dir / f"{digest}.pdf"
        if not path.exists():
            path.write_bytes(data)

        chunks = _chunk(path)
        pages = self._page_count(path)
        title = self._title(path, filename)
        front = _front_matter(path)
        paper = Paper(
            id=digest,
            filename=filename,
            title=title,
            pages=pages,
            chunks=chunks,
            path=path,
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            created_at=self._pdf_date(path),
            arxiv_id=self._arxiv_id(path, filename),
            arxiv_version=self._arxiv_version(path, filename),
            arxiv_date=self._arxiv_date(path),
            scanned=self._is_scanned(path),
            authors=front.authors,
            affiliations=front.affiliations,
            abstract=_abstract_of(chunks, front.authors, front.affiliations, title),
        )
        with self._lock:
            # 两个请求同时上传同一份文件时，先写入的那个胜出，后一个拿到同一对象，
            # 保证内存里只有一份索引（重复的 Paper 对象会让 chunk.index 串味）
            self._papers.setdefault(digest, paper)
            stored = self._papers[digest]

        self._write_meta(stored)
        return stored

    def _write_meta(self, paper: Paper) -> None:
        """写 sidecar。先写临时文件再改名，避免进程半路被杀留下半截 JSON ——
        下次启动解析失败就被当成坏文件跳过了。"""
        meta_path = self.store_dir / f"{paper.id}.json"
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "id": paper.id,
                    "filename": paper.filename,
                    "title": paper.title,
                    "title_v": TITLE_SCHEMA,
                    "authors": paper.authors,
                    "affiliations": paper.affiliations,
                    "pages": paper.pages,
                    "uploaded_at": paper.uploaded_at,
                    "created_at": paper.created_at,
                    "arxiv_id": paper.arxiv_id,
                    "arxiv_version": paper.arxiv_version,
                    "arxiv_date": paper.arxiv_date,
                    "scanned": paper.scanned,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(meta_path)

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _pdf_date(path: Path) -> str:
        """从 PDF 元数据取生成日期。

        这是判断"哪一版更新"的唯一可靠依据 —— 文件名里的 v1/v7 是人为的，
        上传时间只能说明你什么时候传的。取不到就返回空，别瞎猜。
        """
        doc = pymupdf.open(str(path))
        try:
            raw = ((doc.metadata or {}).get("creationDate") or "").strip()
        finally:
            doc.close()
        # 形如 D:20230614112345Z 或 D:20230614112345+08'00'
        digits = "".join(ch for ch in raw[2:10] if ch.isdigit()) if raw.startswith("D:") else ""
        if len(digits) != 8:
            return ""
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"

    # 首页左侧那行旋转的投稿戳：arXiv:1706.03762v7 [cs.CL] 2 Aug 2023
    _ARXIV = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)

    @staticmethod
    def _arxiv_id(path: Path, filename: str) -> str:
        """取 arXiv 号（不带版本号），取不到就返回空串。

        为什么值得单独存一个字段：两张 PDF 的标题在抽取时可能差一个词
        （参考文献里常被截断、断词），而 arXiv 号是**精确**的标识。
        有它就先用它，没有才退回标题模糊比对。
        """
        doc = pymupdf.open(str(path))
        try:
            first = doc[0].get_text()
        finally:
            doc.close()
        m = PaperRegistry._ARXIV.search(first)
        if m:
            return m.group(1)
        # 报告类 PDF（GPT-3/4）没有投稿戳，但文件名常常就是 arXiv 号
        m = re.match(r"(\d{4}\.\d{4,5})", Path(filename).stem)
        return m.group(1) if m else ""

    # 投稿戳带版本号：arXiv:1706.03762v7 [cs.CL] 2 Aug 2023
    _ARXIV_VERSIONED = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}v(\d+)", re.IGNORECASE)
    _STEM_VERSIONED = re.compile(r"\d{4}\.\d{4,5}v(\d+)")

    @staticmethod
    def _arxiv_version(path: Path, filename: str) -> int:
        """arXiv 的版本号（v7 → 7）；取不到返回 0 —— 0 是"不知道"，不是 v1。

        为什么版本谱系非得有它：`created_at` 只精确到天，同一篇的两个版本
        常常由同一个人在同一天导出，比不出先后；扫描件连元数据日期都没有，
        以前就只能退回上传时间 —— 结果是"后传的扫描件 v1"被判成了更新的那版。
        vN 是作者发版时自己打的，对同一篇论文来说它才是最硬的新旧依据。
        """
        doc = pymupdf.open(str(path))
        try:
            first = doc[0].get_text()
        finally:
            doc.close()
        m = PaperRegistry._ARXIV_VERSIONED.search(first) or PaperRegistry._STEM_VERSIONED.search(
            Path(filename).stem
        )
        return int(m.group(1)) if m else 0

    # 首页左侧那行旋转的投稿戳，日期就在方括号后面：
    #   arXiv:2508.18224v3  [cs.DC]  1 Jul 2026
    # 方括号里的分类最长也就 "cs.CL"、"physics.soc-ph" 这种，给 20 个字符足够；
    # 版本号有的 PDF 没有（早期投稿戳），所以整段 `vN` 可选。
    _ARXIV_STAMP_DATE = re.compile(
        r"arxiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\s*\[[^\]]{0,20}\]\s*"
        r"(\d{1,2})\s+([A-Za-z]{3})[A-Za-z]*\.?\s+(\d{4})",
        re.IGNORECASE,
    )
    _MONTHS = {
        name: i
        for i, name in enumerate(
            ("jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"),
            start=1,
        )
    }

    @staticmethod
    def _arxiv_date(path: Path) -> str:
        """首页投稿戳上的日期（YYYY-MM-DD）；取不到返回空串。

        为什么要它：arXiv 从 2025 年起改用 GenPDF 生成 PDF，导出时不再写
        creationDate/modDate —— 实测 FSA（2508.18224v3）的元数据里两个日期都是
        空串，`producer` 是 `pikepdf 8.15.1`。于是文献库里一排论文全显示
        "生成日期未知"，看着像解析坏了，其实 PDF 里真的没有这个字段。

        而投稿戳上的日期是 arXiv 印在版面上的，随版本走、比元数据可靠。
        它**不等于** created_at（导出时间 ≠ 投稿时间），所以单独存一栏、
        前端也分开措辞，不拿它去冒充 PDF 生成日期。
        """
        doc = pymupdf.open(str(path))
        try:
            first = doc[0].get_text()
        finally:
            doc.close()
        m = PaperRegistry._ARXIV_STAMP_DATE.search(first)
        if not m:
            return ""
        day, month, year = int(m.group(1)), m.group(2).lower()[:3], int(m.group(3))
        month_no = PaperRegistry._MONTHS.get(month)
        if not month_no or not 1 <= day <= 31:
            return ""
        return f"{year:04d}-{month_no:02d}-{day:02d}"

    @staticmethod
    def _is_scanned(path: Path) -> bool:
        """首页抽不出文字就当扫描件。

        判据只看首页、而且门槛放得很低（20 个字）：论文首页一定有标题、作者、
        摘要，正常 PDF 不可能凑不出 20 个字；反过来，扫描件里偶尔混进一页有
        文字层的（版权页、封面）也不该让整篇被误当成有文本层。
        """
        doc = pymupdf.open(str(path))
        try:
            text = doc[0].get_text().strip()
        finally:
            doc.close()
        return len(text) < 20

    @staticmethod
    def _page_count(path: Path) -> int:
        doc = pymupdf.open(str(path))
        try:
            return doc.page_count
        finally:
            doc.close()

    @staticmethod
    def _title(path: Path, filename: str) -> str:
        """标题优先从首页版面上取，元数据只做兜底。

        踩过的坑：arXiv 这类 PDF 的元数据 title 常年是空串或者版权声明，
        1706.03762v7 的元数据标题就是 **"scholarly works."** —— 直接信它，
        文献库里七篇论文会有一半叫同一个名字（还是错的）。

        也不能靠切块器的 `section`：它只在 v7 那种排版下把标题当成了 heading，
        换成 v1 就认成 "Abstract"，标题直接掉回文件名 `1706.03762v1`。
        稳定的是字号 —— 首页最大的那行字就是标题，五份样本全部命中。
        """
        candidate = PaperRegistry._title_from_layout(path)
        if candidate:
            return PaperRegistry._prefer_meta_casing(path, candidate)

        # 扫描件：整页没有文本层，字号那套判据全都落空，只能让 OCR 来认
        candidate = PaperRegistry._title_from_ocr(path)
        if candidate:
            return candidate

        doc = pymupdf.open(str(path))
        try:
            meta = ((doc.metadata or {}).get("title") or "").strip()
        finally:
            doc.close()
        if meta and meta.lower() != "untitled" and len(meta) > 3:
            return meta[:200]
        return Path(filename).stem

    @staticmethod
    def _prefer_meta_casing(path: Path, title: str) -> str:
        """版面标题若是全大写、而元数据里是同一个标题的正常大小写，就用元数据的写法。

        只换大小写，**一个字都不换**：两者去掉标点、压成小写后必须完全一致才动手。
        ICLR/NeurIPS 的模板把标题排成全大写，文献库里一排 `FSA: AN ALTERNATIVE
        EFFICIENT IMPLEMENTATION OF…` 读起来像在喊。而这里敢信元数据，正是因为
        它和版面标题逐字相同 —— 换一篇（元数据是 "scholarly works." 那种）就对不上了。
        """
        letters = [ch for ch in title if ch.isalpha()]
        if not letters or any(ch.islower() for ch in letters):
            return title  # 本来就不是全大写，不动
        doc = pymupdf.open(str(path))
        try:
            meta = ((doc.metadata or {}).get("title") or "").strip()
        finally:
            doc.close()
        if not meta or len(meta) > 200:
            return title

        def flatten(text: str) -> str:
            return "".join(ch for ch in text if ch.isalnum()).lower()

        return meta if flatten(meta) == flatten(title) else title

    # arXiv 左侧那行旋转的投稿戳，字号 20 比标题还大，必须先剔掉
    # （规则搬到 app/parsing/layout.py 与首屏元信息共用，这里留个别名，
    #  测试和别的调用方还在用 PaperRegistry._STAMP）
    _STAMP = STAMP
    # 认出来的东西明显不是标题时的兜底判断：章节名、版权行、噪声
    _GENERIC = {
        "abstract", "introduction", "references", "conclusion", "conclusions",
        "acknowledgments", "acknowledgements", "appendix", "contents",
    }

    # 标题第二行的字号常常比第一行小一点点（实测 14.3 对 14.5）。
    # 要求"完全等于最大值"会只取到第一行，标题就断在半句上。
    _TITLE_SIZE_SLACK = 0.8

    @staticmethod
    def _line_of(line: dict) -> tuple[float, str] | None:
        """把 PyMuPDF 的一行还原成 (字号, 文本)。

        **行内拼 span 不加分隔符**：排版器省掉的空格本来就不在 span 文本里，
        补一个空格会造出 `Memory- efficient Attention` 这种原文没有的东西。
        字号取行内最大的那个 span —— 一行里混着主标题和小型大写字母是常态。

        整行是 arXiv 投稿戳时返回 None：它自带版本号和日期，两版必然不同，
        留着会让每份论文的标题都变成 `arXiv:1706.03762v7 [cs.CL] …`。

        实现在 `app/parsing/layout.py`：首屏元信息（作者与机构）要同一套读法，
        两处各写一份就会慢慢走岔。这里保留这个名字，是因为测试与外部调用在用它。
        """
        return line_of(line)

    @staticmethod
    def _title_from_layout(path: Path) -> str | None:
        """取首页字号最大的那几行，按纵向顺序拼起来。

        按**行**取，不按 span 取。排版器会把一行标题拆成一堆 span：逐字调字距、
        小型大写字母、局部上标都会让 PyMuPDF 断开。以前只收"字号恰好等于全页
        最大值"的 span，于是 2310.03294 的标题被拆成 `D`,`F`,`A`,`:`,`Distributed`…
        再按三个一段截断，最后什么也拼不出来（`_clean_title` 判太短返回 None），
        标题只好退到 OCR —— 而 OCR 那一路又吃掉了空格，
        文献库里就出现 `DISTFLASHATTN:Distributed Memory-efficientAttention`。

        行怎么拼见 `_line_of`。这里只管"哪些行算标题"。
        """
        lines = [(ln.size, ln.y0, ln.y1, ln.text) for ln in page_lines(path)]

        if not lines:
            return None

        biggest = max(size for size, _y0, _y1, _t in lines)
        same = sorted(
            (ln for ln in lines if biggest - ln[0] <= PaperRegistry._TITLE_SIZE_SLACK),
            key=lambda ln: ln[1],
        )

        # 只拼纵向相邻的行：隔太远的那是别的元素（页脚、图注），不是标题的换行
        picked = [same[0]]
        for line in same[1:]:
            if line[1] - picked[-1][2] > biggest * 2:
                break
            picked.append(line)
            if len(picked) == 3:
                break

        return PaperRegistry._clean_title(" ".join(t for _s, _y0, _y1, t in picked))

    @staticmethod
    def _clean_title(title: str) -> str | None:
        """两条路（文本层、OCR）共用的最后一道筛子。"""
        title = " ".join(title.split())
        if len(title) < 8 or title.lower().strip(".") in PaperRegistry._GENERIC:
            return None
        if title.endswith("."):  # "scholarly works." 这类页脚声明
            return None
        return title[:200]

    # OCR 常把标题里的空格吃掉：AttentionIsAllYouNeed
    _GLUED = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")

    @staticmethod
    def _despace(text: str) -> str:
        """给 OCR 标题补回被吃掉的空格。

        只在**整行一个空格都没有**时才按驼峰断词：有空格就说明识别器自己认出了
        词边界，这时候再按大写字母切，会把 TransformerXL 这类名字切成三段。
        """
        if " " in text or len(text) < 12:
            return text
        return PaperRegistry._GLUED.sub(" ", text)

    @staticmethod
    def _title_from_ocr(path: Path) -> str | None:
        """扫描件专用：用 OCR 的行框挑标题。

        不能照搬文本层那套"字号最大就是标题"：OCR 给出的 `size` 是行高像素当量，
        噪声很大 —— 实测 1706.03762v1 扫描件首页上，邮箱那行认出来的 size（19）
        比标题（16）还大，照字号挑会挑出 avaswani@google.com。
        稳定的判据是**版面宽度 + 居中**：标题是首页上半部分最宽且居中的那一行。
        """
        from app.parsing import scan

        if not scan.ocr_enabled():
            return None
        doc = pymupdf.open(str(path))
        try:
            page = doc[0]
            height = page.rect.height
            lines = scan.ocr_lines(page)  # 切块时已经识别过，这里基本是缓存命中
        finally:
            doc.close()

        cands = [ln for ln in lines if ln.bbox[3] <= height * 0.45 and _title_shaped(ln.text)]
        if not cands:
            return None

        widest = max(cands, key=lambda ln: ln.bbox[2] - ln.bbox[0])
        w = widest.bbox[2] - widest.bbox[0]
        cx = (widest.bbox[0] + widest.bbox[2]) / 2
        h = max(widest.bbox[3] - widest.bbox[1], 1.0)

        # 换行的标题：宽度不至于腰斩、而且和第一行共用同一条中轴线
        picked = [
            ln for ln in cands
            if (ln.bbox[2] - ln.bbox[0]) >= w * 0.55
            and abs((ln.bbox[0] + ln.bbox[2]) / 2 - cx) <= max(w * 0.15, 8)
        ]
        picked.sort(key=lambda ln: ln.bbox[1])
        out = [picked[0]]
        for ln in picked[1:]:
            if len(out) == 3 or ln.bbox[1] - out[-1].bbox[3] > h * 1.6:
                break
            out.append(ln)
        title = " ".join(PaperRegistry._despace(ln.text.strip()) for ln in out)
        return PaperRegistry._clean_title(title)
