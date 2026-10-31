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
    """有 Tesseract 就把页级 OCR 钩子交给切块器，没有就返回 None。

    在注册表这一层做判断，而不是在切块器里：切块器若直接去找 tesseract，
    每跑一次单元测试都要依赖本机装没装 OCR 引擎，测试结果就不可复现了。
    `ocr_enabled()` 带缓存，541 页的教材不会探 541 次子进程。
    """
    from app.parsing.scan import ocr_blocks_for_page, ocr_enabled

    return ocr_blocks_for_page if ocr_enabled() else None


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
    _index: TextIndex | None = field(default=None, repr=False)
    _bm25: BM25 | None = field(default=None, repr=False)
    _fingerprint: set[str] | None = field(default=None, repr=False)
    _refs: list | None = field(default=None, repr=False)
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

                    self._refs = extract_references(self.chunks, self.path)
        return self._refs

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
                chunks = chunk_pdf(str(pdf_path), ocr=_ocr_hook())
                if not chunks:
                    continue
                self._papers[meta["id"]] = Paper(
                    id=meta["id"],
                    filename=meta.get("filename") or pdf_path.name,
                    title=meta.get("title") or self._title(pdf_path, pdf_path.name),
                    pages=meta.get("pages") or self._page_count(pdf_path),
                    chunks=chunks,
                    path=pdf_path,
                    uploaded_at=meta.get("uploaded_at") or "",
                    created_at=meta.get("created_at") or self._pdf_date(pdf_path),
                    # 兜底要用**原始文件名**：落盘的 PDF 叫 `<hash>.pdf`，里面没有 arXiv 号
                    arxiv_id=meta.get("arxiv_id")
                    or self._arxiv_id(pdf_path, meta.get("filename") or pdf_path.name),
                )
            except Exception:  # noqa: BLE001 —— 单个坏文件不该拖垮整个启动
                continue

        # 只认 sidecar 会有个洞：手动丢进目录的 PDF 永远加载不出来。
        # 所以再扫一遍裸 PDF，用文件名兜底 —— 少一点元数据，总好过"文件在库里却看不见"。
        for pdf_path in sorted(self.store_dir.glob("*.pdf")):
            if pdf_path.stem in self._papers:
                continue
            try:
                chunks = chunk_pdf(str(pdf_path), ocr=_ocr_hook())
                if not chunks:
                    continue
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
        digest = hashlib.sha256(data).hexdigest()[:12]
        with self._lock:
            if digest in self._papers:
                return self._papers[digest]

        path = self.store_dir / f"{digest}.pdf"
        if not path.exists():
            path.write_bytes(data)

        chunks = chunk_pdf(str(path), ocr=_ocr_hook())
        pages = self._page_count(path)
        title = self._title(path, filename)
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
        )
        with self._lock:
            # 两个请求同时上传同一份文件时，先写入的那个胜出，后一个拿到同一对象，
            # 保证内存里只有一份索引（重复的 Paper 对象会让 chunk.index 串味）
            self._papers.setdefault(digest, paper)
            stored = self._papers[digest]

        # sidecar 用"先写临时文件再改名"，避免进程半路被杀留下半截 JSON，
        # 下次启动解析失败就被当成坏文件跳过了
        meta_path = self.store_dir / f"{stored.id}.json"
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "id": stored.id,
                    "filename": stored.filename,
                    "title": stored.title,
                    "pages": stored.pages,
                    "uploaded_at": stored.uploaded_at,
                    "created_at": stored.created_at,
                    "arxiv_id": stored.arxiv_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(meta_path)
        return stored

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
            return candidate

        doc = pymupdf.open(str(path))
        try:
            meta = ((doc.metadata or {}).get("title") or "").strip()
        finally:
            doc.close()
        if meta and meta.lower() != "untitled" and len(meta) > 3:
            return meta[:200]
        return Path(filename).stem

    # arXiv 左侧那行旋转的投稿戳，字号 20 比标题还大，必须先剔掉
    _STAMP = re.compile(r"^arxiv:\S+\s*\[", re.IGNORECASE)
    # 认出来的东西明显不是标题时的兜底判断：章节名、版权行、噪声
    _GENERIC = {
        "abstract", "introduction", "references", "conclusion", "conclusions",
        "acknowledgments", "acknowledgements", "appendix", "contents",
    }

    @staticmethod
    def _title_from_layout(path: Path) -> str | None:
        """取首页字号最大的那几行，按纵向顺序拼起来。"""
        doc = pymupdf.open(str(path))
        try:
            page = doc[0]
            spans: list[tuple[float, str, float]] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = " ".join(span["text"].split())
                        if not text or PaperRegistry._STAMP.match(text):
                            continue
                        spans.append((round(float(span["size"]), 1), text, float(span["bbox"][1])))
        finally:
            doc.close()

        if not spans:
            return None

        biggest = max(size for size, _t, _y in spans)
        lines = sorted((s for s in spans if s[0] == biggest), key=lambda s: s[2])

        # 只拼纵向相邻的行：隔太远的那是别的元素（页脚、图注），不是标题的换行
        picked = [lines[0]]
        for size, text, y in lines[1:]:
            if y - picked[-1][2] > biggest * 2:
                break
            picked.append((size, text, y))
            if len(picked) == 3:
                break

        title = " ".join(text for _s, text, _y in picked).strip()
        if len(title) < 8 or title.lower().strip(".") in PaperRegistry._GENERIC:
            return None
        if title.endswith("."):  # "scholarly works." 这类页脚声明
            return None
        return title[:200]
