"""arXiv 动态追踪与"和本地库相关"的推荐。

赛题模块四的后半段要三件事，这里一一对上：

1. **定期刷新追踪 arXiv 最新论文**：`arxiv.latest()` 按 `submittedDate` 倒序取分类流，
   本模块外面再套一层 TTL 缓存（见 `api/routers/recommend.py`）——
   页面每分钟被打开十次也只会真正出网一次。
2. **结合本地文献库计算相关性**：把库里每篇论文的标题 + 前两块的词，与候选论文的
   标题 + 摘要做**词面余弦**。不用向量模型有两个理由：库里可能没装模型（要降级），
   以及推荐理由必须能被用户看懂 —— "和《X》有 7 个共同关键词" 比 "相似度 0.83" 有用。
3. **优先推荐带开源代码的**：从摘要和 `<arxiv:comment>` 里找 `github.com/owner/repo`。
   作者常把代码地址只写在 comment 里，只看摘要会漏掉一大半。

**为什么推荐理由由模板拼、不交给大模型**：理由是"为什么推给你"的证据，一旦由模型
自由发挥，就会写出"这篇论文提出了全新的架构"这种原文里根本没有的断言。这里每一条
理由都能追到一个具体数字：重合了几篇、哪几篇、代码地址原文是哪个、几天前提交的。

**口味（点赞记录）是乘法项，不是第四项加法**：`score × (1 + 0.15 × 口味匹配)`。
加法会稀释相关性 —— 相关性是"和你的库像不像"，它是这个系统的主线；口味只该在
同一条水平线上分先后。点赞不足两篇时口味恒为 0（点一次就把排序带偏，那不叫个性化，
那叫过拟合）。

**打分权重是手调的，不是学出来的**：`0.75×相关性 + 0.20×有代码 + 0.05×新鲜度`。
相关性用 0.5 做归一化上限（实测词面余弦很少超过 0.5），这样它仍是决定性的那一项，
代码和新鲜度只用来在同一条水平线上分先后 —— 正好对应赛题"优先推荐易复现的"。
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.parsing.text_index import split_tokens
from app.tools.arxiv import ArxivEntry

GITHUB_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)

# 相关性归一化的上限：词面余弦超过 0.5 基本是同一篇论文的不同版本了
RELEVANCE_CAP = 0.5
W_RELEVANCE = 0.75
W_CODE = 0.20
W_FRESH = 0.05
# 口味是乘上去的（见模块说明）：最多把总分抬高 15%，足以在同分时换个先后，
# 不足以把一篇和库毫无关系的论文顶到第一篇。
W_TASTE = 0.15
# 少于这么多篇点赞，就不认为"这个人的口味"已经成形
MIN_TASTE_VOTES = 2

STOPWORDS = frozenset(
    """
    a an the of in on for to and or with by from at as is are was were be been
    this that these those we our their its it can may not no using used use based
    paper work method methods approach results show shows propose proposed
    """.split()
)


@dataclass(frozen=True, slots=True)
class LibraryPaper:
    id: str
    title: str
    tokens: frozenset[str]


@dataclass(frozen=True, slots=True)
class Match:
    paper_id: str
    title: str
    similarity: float


@dataclass(frozen=True, slots=True)
class Recommendation:
    entry: ArxivEntry
    score: float
    relevance: float
    matches: list[Match] = field(default_factory=list)
    code_url: str = ""
    reason: str = ""
    votes_up: int = 0
    votes_down: int = 0
    my_vote: str = ""


def keywords(text: str, limit: int = 400) -> frozenset[str]:
    """切词、丢停用词、去重。截断是为了让摘要的长尾不至于淹没标题。"""
    tokens = split_tokens(text)[:limit]
    return frozenset(t for t in tokens if len(t) > 2 and t not in STOPWORDS)


def build_profile(papers) -> list[LibraryPaper]:
    """把注册表里的论文变成"只留关键词"的画像。

    用标题 + 前两块而不是全文：摘要和引言已经交代了问题与做法，
    后面的实验细节会把关键词表撑得又大又杂，余弦反而被稀释。
    """
    profile: list[LibraryPaper] = []
    for paper in papers:
        head = " ".join(chunk.text for chunk in paper.chunks[:2])
        profile.append(
            LibraryPaper(
                id=paper.id,
                title=paper.title,
                tokens=keywords(f"{paper.title} {head}", limit=1200),
            )
        )
    return profile


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """二元向量的余弦：|A∩B| / sqrt(|A|·|B|)。

    不看词频，因为这里要回答的是"讲的是不是同一件事"，不是"哪个词出现得多"。
    """
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def extract_code_url(*texts: str) -> str:
    """从摘要 / comment 里找 GitHub 地址，找不到返回空串。

    只认 `github.com/owner/repo` 这一种形态：把任意链接当"代码仓库"是错的，
    用户点进去发现是篇博客，比不给链接更糟。
    """
    for text in texts:
        if not text:
            continue
        match = GITHUB_RE.search(text)
        if match:
            owner, repo = match.group(1), match.group(2).rstrip(".,;:)）】」")
            if repo.lower().endswith(".git"):
                repo = repo[:-4]
            if owner and repo:
                return f"https://github.com/{owner}/{repo}"
    return ""


def _freshness(published: str, today: str) -> tuple[float, str]:
    """新鲜度：当天算 1.0，之后按天衰减到底。

    日期解析失败就当"不知道"，给 0 分并且不写进理由 —— 宁可少一条理由，
    也不要编一个"刚刚提交"。
    """
    if not published or not today:
        return 0.0, ""
    try:
        from datetime import date

        then = date.fromisoformat(published[:10])
        now = date.fromisoformat(today[:10])
    except ValueError:
        return 0.0, ""
    days = (now - then).days
    if days <= 0:
        return 1.0, "今日提交"
    if days == 1:
        return 0.8, "昨日提交"
    if days < 7:
        return max(0.2, 0.8 - 0.1 * (days - 1)), f"提交于 {days} 天前"
    return 0.0, f"提交于 {days} 天前"


def rank(
    entries: list[ArxivEntry],
    profile: list[LibraryPaper],
    *,
    today: str = "",
    limit: int = 12,
    taste: "Taste | None" = None,
) -> list[Recommendation]:
    """给每篇候选论文算分、配理由，按分数从高到低返回。

    `limit` 是返回条数；排序在全量候选上做，先截断再排序会漏掉"相关但排在后面"的。
    """
    results: list[Recommendation] = []
    for entry in entries:
        tokens = keywords(f"{entry.title} {entry.abstract}")
        matches = sorted(
            (
                Match(paper.id, paper.title, round(similarity(tokens, paper.tokens), 3))
                for paper in profile
            ),
            key=lambda m: m.similarity,
            reverse=True,
        )
        matches = [m for m in matches if m.similarity > 0][:3]
        relevance = matches[0].similarity if matches else 0.0

        code_url = extract_code_url(entry.abstract, entry.comment)
        fresh, fresh_note = _freshness(entry.published, today)

        base = (
            W_RELEVANCE * min(1.0, relevance / RELEVANCE_CAP)
            + W_CODE * (1.0 if code_url else 0.0)
            + W_FRESH * fresh
        )
        taste_hit = taste.score(tokens) if taste is not None else 0.0
        score = base * (1.0 + W_TASTE * taste_hit)

        reasons: list[str] = []
        if matches:
            top = matches[0]
            reasons.append(f"与库内《{top.title}》词面重合度 {top.similarity:.2f}")
            if len(matches) > 1:
                reasons.append(f"另与 {len(matches) - 1} 篇库内文献词面匹配")
        elif profile:
            reasons.append("与库内现有论文无明显词面重合（可能为新方向）")
        else:
            reasons.append("文献库为空，暂按提交时间与开源情况排序")
        if code_url:
            reasons.append(f"摘要或备注中给出了开源代码 {code_url}")
        if fresh_note:
            reasons.append(fresh_note)
        if taste_hit >= 0.05:
            reasons.append(f"与你点赞过的 {taste.size} 篇口味匹配 {taste_hit:.2f}")

        results.append(
            Recommendation(
                entry=entry,
                score=round(score, 4),
                relevance=round(relevance, 3),
                matches=matches,
                code_url=code_url,
                reason="；".join(reasons),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


# ---------------------------------------------------------------- 点赞 / 踩


@dataclass(frozen=True, slots=True)
class Taste:
    """一个人的口味画像：点赞过的**标题**切出来的词。

    为什么用标题而不是摘要：投票那一刻手上只有标题（前端就传了这个）。
    标题里通常正好是"这件事叫什么"（retrieval augmentation / diffusion model），
    拿它做画像够用，也不至于为了个性化再回头去抓一遍全文。
    """

    liked: frozenset[str] = frozenset()
    disliked: frozenset[str] = frozenset()
    size: int = 0

    def score(self, tokens: frozenset[str]) -> float:
        """0~1 的口味匹配度：像点赞过的，减去像踩过的。"""
        if not tokens or not self.liked:
            return 0.0
        hit = similarity(tokens, self.liked)
        if self.disliked:
            # 不扣成负数：一次误点的代价不该是"这一篇永远排最后"，
            # 只把"像点赞过"和"像踩过"两边相减
            hit = max(0.0, hit - similarity(tokens, self.disliked))
        return hit


def build_taste(up_titles: list[str], down_titles: list[str]) -> Taste | None:
    """点赞够数才给口味。一篇就够的话，第一次误点就能把整个推荐页带偏。"""
    if len(up_titles) < MIN_TASTE_VOTES:
        return None
    return Taste(
        liked=keywords(" ".join(up_titles)),
        disliked=keywords(" ".join(down_titles)) if down_titles else frozenset(),
        size=len(up_titles),
    )


class FeedbackStore:
    """点赞/踩的落盘存储：`{"votes": {arxiv_id: {user: "up"|"down"}}, "titles": {...}}`。

    为什么按用户分开存而不是只存计数：赛题要求"多用户推荐算法拓展"，
    只留一个总数就没法做"这个人喜欢的别人不一定喜欢"。

    为什么还要存标题：点赞本身没有用 —— 有用的是"这些被点赞的论文在讲什么"。
    标题在投票那一刻就有，顺手记下来，之后就不用为了算口味再出网抓一次。

    老格式（`{arxiv_id: {user: vote}}`，没有外层 `votes`）读进来时自动按老格式解释，
    下次写盘就升级成新格式 —— 比赛前改存储格式，不能让上一版留下的文件读不出来。

    写盘用"临时文件 + 原子替换"：边演示边点赞的时候进程被杀，
    也不该留下一个半截 JSON 让下次启动直接崩。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
        """返回 (投票, 标题)。读不出来就是两个空表。"""
        if not self.path.exists():
            return {}, {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}
        if not isinstance(data, dict):
            return {}, {}
        if "votes" in data:                      # 新格式
            votes = data.get("votes") or {}
            titles = data.get("titles") or {}
            return (
                {str(k): {str(u): str(v) for u, v in (bucket or {}).items()}
                 for k, bucket in votes.items() if isinstance(bucket, dict)},
                {str(k): str(v) for k, v in titles.items()},
            )
        # 老格式：整个文件就是 votes
        return (
            {str(k): {str(u): str(v) for u, v in (bucket or {}).items()}
             for k, bucket in data.items() if isinstance(bucket, dict)},
            {},
        )

    def _write(self, votes: dict[str, dict[str, str]], titles: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": 2, "votes": votes, "titles": titles}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def counts(self, arxiv_id: str, user: str = "") -> tuple[int, int, str]:
        """返回 (赞, 踩, 我投的那一票)。"""
        user = user or "local"
        votes = self._read()[0].get(arxiv_id) or {}
        up = sum(1 for v in votes.values() if v == "up")
        down = sum(1 for v in votes.values() if v == "down")
        return up, down, votes.get(user, "")

    def cast(
        self, arxiv_id: str, user: str, vote: str, title: str = ""
    ) -> tuple[int, int, str]:
        """投票。`vote` 为空 = 取消之前那一票。

        同一个人再点同一个按钮就是取消：没有这个反悔入口，一次误点会把
        推荐排序永久带偏。
        """
        if vote not in ("", "up", "down"):
            raise ValueError(f"不认识的投票：{vote!r}")
        # 空 user 就是"本机这个人"：读和写必须用同一个名字，否则投完票
        # 读回来的时候找不到自己那一票，界面上按钮永远不亮
        user = user or "local"
        with self._lock:
            votes, titles = self._read()
            bucket = votes.setdefault(arxiv_id, {})
            if vote:
                bucket[user] = vote
            else:
                bucket.pop(user, None)
            if not bucket:
                votes.pop(arxiv_id, None)
            clean = " ".join((title or "").split())
            if clean:
                titles[arxiv_id] = clean[:300]
            self._write(votes, titles)
        return self.counts(arxiv_id, user)

    def taste(self, user: str = "") -> Taste | None:
        """这个人的口味；点赞不足 MIN_TASTE_VOTES 篇时返回 None（不参与排序）。"""
        user = user or "local"
        votes, titles = self._read()
        up = [titles.get(pid, "") for pid, bucket in votes.items() if bucket.get(user) == "up"]
        down = [titles.get(pid, "") for pid, bucket in votes.items() if bucket.get(user) == "down"]
        return build_taste([t for t in up if t], [t for t in down if t])


def cache_path(base: Path) -> Path:
    return Path(os.environ.get("FEEDBACK_PATH") or base / "feedback.json")
