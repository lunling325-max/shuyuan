"""按来源限流的令牌桶（token bucket）。

**为什么需要**：这台机器挂在公网 IP 上，后面是一个付费的大模型。
没有配额，任何人发现地址就能把密钥烧掉；而答辩现场又最怕"请求被卡住"，
所以阈值要**宽到正常演示碰不到，窄到滥用一眼就撞墙**。

**为什么是进程内的桶**：Dockerfile 里的启动命令没有 `--workers`，只有一个进程，
进程内的字典就是全局的。将来起多 worker，这里的桶会各算各的 —— 换成 Redis 才
是正解，这一点写在这里，免得下一个人以为它天然支持横向扩容。

**默认值的取舍**：每分钟 60 次、突发 10 次。一次问答约 3k token、三分钱上下，
所以封顶是"每小时十几块钱"——谈不让滥用者把密钥刷爆，同时正常演示（连点十几下、
隔几秒点一次）碰不到。阈值定得比这更严，第一个被挡住的就是演示者自己。

**Docker 的坑（实测过一遍）**：端口映射在某些配置下会让容器只看到网桥网关，
于是所有外部用户共用一个 key，限流退化成"全局桶"。这台机器上没有退化 ——
`docker compose logs` 里看到的是真实对端地址（`218.2.220.26`），所以按来源限流
是真的生效。key 仍然优先取 `X-Forwarded-For`：将来前面挡一层 nginx 时不用改代码。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# 各类请求的"价格"。用令牌数而不是"每类各一个桶"，是为了让一次上传
# （解析 + 可能触发 OCR，几秒到十几秒 CPU）和一次问答（一次模型调用）
# 记在同一个额度上 —— 否则刷上传的请求永远不会被问答题目的限额挡住。
COST_UPLOAD = 3.0
COST_BATCH = 6.0
COST_INGEST = 3.0
COST_DEFAULT = 1.0

_EXEMPT_PREFIXES = ("/api/health", "/docs", "/openapi.json", "/redoc")


def cost_of(method: str, path: str) -> float:
    """这次请求要花几个令牌。返回 0 表示不计数。

    只对 POST 计费：GET 都是读内存里的索引，纳秒级，限它没有意义，
    反而会让"刷新一下看结果"变成撞墙。
    """
    if method != "POST":
        return 0.0
    if path.startswith(_EXEMPT_PREFIXES):
        return 0.0
    if path == "/api/papers/batch":
        return COST_BATCH
    if path == "/api/papers" or path.startswith("/api/tools/arxiv/ingest"):
        return COST_UPLOAD
    if path.startswith("/api/"):
        return COST_DEFAULT
    return 0.0


@dataclass
class TokenBucket:
    """容量 `capacity`，每秒回填 `refill` 个令牌。"""

    capacity: float
    refill: float
    tokens: float = 0.0
    updated: float = 0.0

    def take(self, cost: float, now: float) -> tuple[bool, float, float]:
        """尝试扣掉 `cost` 个令牌，返回 `(是否放行, 还需等待秒数, 剩余令牌)`。"""
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill)
        self.updated = now
        if self.tokens + 1e-9 >= cost:
            self.tokens -= cost
            return True, 0.0, self.tokens
        need = cost - self.tokens
        # refill 为 0 时永远补不回来：返回一个足够大的等待值，别抛异常
        retry = need / self.refill if self.refill > 0 else 3600.0
        return False, retry, self.tokens


class PerClientLimiter:
    """每个 key 一个桶。线程安全 —— FastAPI 的同步路由跑在线程池里。

    `capacity` 是**突发额度**（能连着点几下），`per_minute` 是**持续速率**
    （长期平均每分钟几次）。两者分开是因为演示的真实形态是"安静一会儿，
    然后连着点三四下"，只给一个速率会让这种用法立刻撞墙。
    """

    def __init__(
        self,
        *,
        capacity: float,
        per_minute: float,
        clock=time.monotonic,
        max_keys: int = 2048,
    ) -> None:
        self.capacity = float(capacity)
        self.refill = float(per_minute) / 60.0
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, cost: float = 1.0) -> tuple[bool, float, float]:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self.capacity, refill=self.refill,
                    tokens=self.capacity, updated=now,
                )
                self._buckets[key] = bucket
                self._prune(now)
            return bucket.take(cost, now)

    def _prune(self, now: float) -> None:
        """桶太多时清掉已经回满的：回满的桶与新建的桶等价，删掉不丢信息。"""
        if len(self._buckets) <= self._max_keys:
            return
        for key in [k for k, b in self._buckets.items()
                    if b.tokens + (now - b.updated) * b.refill >= b.capacity]:
            del self._buckets[key]


def client_key(scope_headers: list[tuple[bytes, bytes]], client: object) -> str:
    """取来源标识：优先 `X-Forwarded-For` 的第一个地址，其次 socket 对端。

    为什么取**第一个**：`X-Forwarded-For` 是逐跳追加的，最左边那个是真实客户端，
    右边都是代理。取右边等于给所有用户发同一个 key。
    """
    for name, value in scope_headers:
        if name == b"x-forwarded-for":
            first = value.decode("latin-1").split(",")[0].strip()
            if first:
                return first
    host = getattr(client, "host", None) if client is not None else None
    return host or "unknown"


class RateLimitMiddleware:
    """纯 ASGI 中间件。不用 BaseHTTPMiddleware，是为了不额外包一层任务组 ——
    流式响应（SSE）经过它时不该被缓冲或打断。
    """

    def __init__(self, app, limiter: PerClientLimiter, *, enabled: bool = True) -> None:
        self.app = app
        self.limiter = limiter
        self.enabled = enabled

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        cost = cost_of(scope.get("method", "GET"), scope.get("path", ""))
        if cost <= 0:
            await self.app(scope, receive, send)
            return

        key = client_key(scope.get("headers") or [], scope.get("client"))
        allowed, retry, _remaining = self.limiter.hit(key, cost)
        if not allowed:
            wait = max(1, int(retry + 0.999))
            body = (
                '{"detail": "请求过于频繁，请约 %d 秒后再试。"}' % wait
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(wait).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def limiter_from_env(env) -> PerClientLimiter | None:
    """按环境变量建限流器。`RATE_LIMIT_PER_MINUTE=0` 表示关掉（本地开发、测试）。"""
    try:
        per_minute = float(env.get("RATE_LIMIT_PER_MINUTE", "60"))
    except ValueError:
        per_minute = 60.0
    if per_minute <= 0:
        return None
    try:
        burst = float(env.get("RATE_LIMIT_BURST", "10"))
    except ValueError:
        burst = 10.0
    return PerClientLimiter(capacity=max(burst, 1.0), per_minute=per_minute)
