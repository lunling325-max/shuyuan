"""限流的测试。

守住四条：
1. 桶按**时间**回填，不是按请求数回填 —— 否则"连着点两下"和"隔一分钟点一下"
   会被判成同一件事；
2. 突发额度（burst）与持续速率是两个参数，前者管"手快"，后者管"长跑"；
3. 被拦下时给出的是**人话 + 可执行的等待秒数**，不是一个光秃秃的 429；
4. 上传、问答、批量分别计不同的价 —— 否则刷上传永远撞不到墙。

这几条都在独立的 FastAPI app 上测，不碰真业务，避免把限流和业务耦合进测试。
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.ratelimit import (
    COST_BATCH,
    COST_UPLOAD,
    PerClientLimiter,
    RateLimitMiddleware,
    client_key,
    cost_of,
    limiter_from_env,
)


class FakeClock:
    """可以手拨的时钟 —— 让"等一分钟"变成一行赋值。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------- 计价


def test_get_requests_are_free():
    """读接口不限：刷新页面看结果不该撞墙。"""
    assert cost_of("GET", "/api/papers") == 0.0
    assert cost_of("GET", "/api/graph") == 0.0


def test_expensive_posts_cost_more():
    assert cost_of("POST", "/api/papers") == COST_UPLOAD
    assert cost_of("POST", "/api/papers/batch") == COST_BATCH
    assert cost_of("POST", "/api/tools/arxiv/ingest") == COST_UPLOAD
    assert cost_of("POST", "/api/papers/x/qa") == 1.0
    assert COST_UPLOAD < COST_BATCH


def test_health_is_exempt_even_if_posted():
    assert cost_of("POST", "/api/health") == 0.0


# ---------------------------------------------------------------- 桶


def test_burst_capacity_is_consumed_then_refused():
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=3, per_minute=6, clock=clock)
    assert [limiter.hit("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry, _ = limiter.hit("a")
    assert not allowed
    assert retry > 0


def test_refill_is_time_based_not_request_based():
    """撞墙之后狂点也没用：补令牌靠时间，不靠重试。"""
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=1, per_minute=60, clock=clock)
    assert limiter.hit("a")[0]
    for _ in range(20):
        assert not limiter.hit("a")[0]
    clock.advance(1.0)  # 每分钟 60 个 → 1 秒补 1 个
    assert limiter.hit("a")[0]


def test_retry_after_is_the_time_actually_needed():
    """告诉用户"等 30 秒"，就应该是真等 30 秒能过，不是拍脑袋的常数。"""
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=1, per_minute=6, clock=clock)  # 10 秒 1 个
    assert limiter.hit("a")[0]
    allowed, retry, _ = limiter.hit("a")
    assert not allowed
    assert retry == pytest.approx(10.0, abs=0.01)
    clock.advance(retry)
    assert limiter.hit("a")[0]


def test_buckets_are_per_key():
    """一个人撞墙不该连累另一个人。"""
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=1, per_minute=6, clock=clock)
    assert limiter.hit("a")[0]
    assert not limiter.hit("a")[0]
    assert limiter.hit("b")[0]


def test_expensive_requests_drain_faster():
    """一次批量 = 六次问答，额度是同一个。"""
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=6, per_minute=60, clock=clock)
    assert limiter.hit("a", COST_BATCH)[0]
    assert not limiter.hit("a", 1.0)[0]


def test_prune_drops_refilled_buckets_only():
    """桶太多时清掉已回满的；没回满的必须留着，否则等于给人重置额度。"""
    clock = FakeClock()
    limiter = PerClientLimiter(capacity=2, per_minute=60, clock=clock, max_keys=10)
    for i in range(10):
        limiter.hit(f"k{i}")
    clock.advance(30)     # 其余九个桶都已回满（按需结算估算）
    limiter.hit("k0")     # k0 被消耗一次，只剩 1 个令牌，不算回满
    limiter.hit("new")    # 触发清理
    # k0 的额度没有被重置：还能过一次，第二次就撞墙
    assert limiter.hit("k0")[0]
    assert not limiter.hit("k0")[0]


# ---------------------------------------------------------------- 中间件


def _app(limiter: PerClientLimiter, enabled: bool = True) -> TestClient:
    app = FastAPI()

    @app.post("/api/papers")
    def upload() -> dict:
        return {"ok": True}

    @app.get("/api/papers")
    def listing() -> dict:
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, limiter=limiter, enabled=enabled)
    return TestClient(app)


def test_middleware_returns_429_with_a_readable_message():
    clock = FakeClock()
    # 上传一次 3 个令牌，所以容量按 3 起算 —— 这里测的是"拦下时说什么"，不是配额本身
    client = _app(PerClientLimiter(capacity=COST_UPLOAD, per_minute=6, clock=clock))
    assert client.post("/api/papers").status_code == 200
    r = client.post("/api/papers")
    assert r.status_code == 429
    assert "请求过于频繁" in r.json()["detail"]
    assert int(r.headers["retry-after"]) >= 1


def test_middleware_does_not_touch_reads():
    clock = FakeClock()
    client = _app(PerClientLimiter(capacity=COST_UPLOAD, per_minute=6, clock=clock))
    assert client.post("/api/papers").status_code == 200
    assert client.post("/api/papers").status_code == 429
    assert client.get("/api/papers").status_code == 200


def test_middleware_can_be_disabled():
    clock = FakeClock()
    client = _app(PerClientLimiter(capacity=1, per_minute=6, clock=clock), enabled=False)
    assert all(client.post("/api/papers").status_code == 200 for _ in range(5))


def test_client_key_prefers_the_first_forwarded_address():
    """代理链最左边那个才是真实客户端；取右边等于给所有人发同一个 key。"""
    headers = [(b"x-forwarded-for", b"203.0.113.9, 10.0.0.1")]

    class Peer:
        host = "172.18.0.1"

    assert client_key(headers, Peer()) == "203.0.113.9"
    assert client_key([], Peer()) == "172.18.0.1"


# ---------------------------------------------------------------- 配置


def test_zero_per_minute_disables_the_limiter():
    assert limiter_from_env({"RATE_LIMIT_PER_MINUTE": "0"}) is None


def test_bad_env_falls_back_instead_of_crashing():
    """环境变量写错了不该让服务起不来 —— 按默认值走。"""
    limiter = limiter_from_env({"RATE_LIMIT_PER_MINUTE": "abc", "RATE_LIMIT_BURST": "x"})
    assert limiter is not None
    assert limiter.capacity == 10.0
    assert limiter.refill == pytest.approx(1.0)


def test_defaults_are_wide_enough_for_a_demo():
    """默认值要"宽到正常演示碰不到"：连点五下不能撞墙。"""
    limiter = limiter_from_env({})
    assert limiter is not None
    assert all(limiter.hit("demo", 1.0)[0] for _ in range(5))


def test_clock_is_monotonic_by_default():
    """默认用单调时钟：系统时间被改回去时，限流不该跟着倒回去。"""
    limiter = PerClientLimiter(capacity=1, per_minute=60)
    assert limiter.hit("a")[0]
    before = time.monotonic()
    assert limiter.hit("a")[1] <= 1.0
    assert time.monotonic() >= before
