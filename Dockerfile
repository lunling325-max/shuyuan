# syntax=docker/dockerfile:1
#
# 两阶段构建：
#   builder 只负责装依赖；runtime 只从 builder 搬 .venv，不把 pip 缓存、
#   编译工具链或测试依赖带进最终镜像。
#
# 基础镜像固定 python:3.11-slim，与本地开发版本一致。不用 alpine：
# alpine 用 musl libc，onnxruntime / pymupdf 的预编译 wheel 覆盖不全，
# 很可能退化成源码编译，反而需要装工具链、镜像更大。

FROM python:3.11-slim AS builder

# 国内网络现实：默认走清华镜像，构建时可用
#   docker build --build-arg PYPI_INDEX_URL=https://your.mirror/simple .
# 覆盖。默认值与 apps/api/pyproject.toml 里固定的索引一致，避免构建阶段
# 悄悄去连 pypi.org。
#
# 为什么用 `pip install uv` 而不是 COPY ghcr.io/astral-sh/uv 的二进制：
# ghcr.io 在国内经常拉不动，而 pip 可以走同一个可配置的 PyPI 镜像；
# uv 本身就是官方发布在 PyPI 上的包，版本可固定，来源也容易审计。
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG UV_VERSION=0.12.2

# 禁止 uv 自己下载 Python：基础镜像里已经有 3.11，下载既慢又绕过镜像。
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir -i "${PYPI_INDEX_URL}" "uv==${UV_VERSION}"

WORKDIR /app

# 先只拷依赖清单，让"依赖没变"时这一层能命中缓存（源码改动不会重装依赖）。
# 注意：没有 `COPY . .`，所以 models/（128MB）、pdf/、.git、.env、
# __pycache__ 都不会进入构建上下文或镜像。
COPY apps/api/pyproject.toml apps/api/uv.lock ./

# --no-install-project：此时 app/ 源码还没进来，只装第三方依赖。
# --frozen：严格按 uv.lock 装，保证可复现。
# --default-index：显式指定索引，和 pyproject.toml 里的清华源形成双保险。
#
# uv.lock 里记录了默认镜像的完整下载 URL。如果构建者把 PYPI_INDEX_URL 换成
# 别的源，frozen 同步仍会去读旧 URL，等于覆盖没生效；所以只在 ARG 与锁文件
# 一致时用 --frozen，否则重新解析。默认路径（清华源）仍然完全可复现。
RUN if grep -qF "${PYPI_INDEX_URL}" uv.lock; then \
        uv sync --frozen --no-dev --no-install-project --default-index "${PYPI_INDEX_URL}"; \
    else \
        uv sync --no-dev --no-install-project --default-index "${PYPI_INDEX_URL}"; \
    fi


FROM python:3.11-slim AS runtime

# onnxruntime 的 Linux CPU wheel 依赖 libgomp（OpenMP 运行时）。
# 缺了它 import 会失败，向量检索只能降级；所以装运行时库，但不装任何编译工具链。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 非 root 运行：应用被攻破时拿不到容器 root 权限。
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# PYTHONDONTWRITEBYTECODE：容器文件系统层不写 __pycache__，避免无谓膨胀；
# PYTHONUNBUFFERED：日志立刻刷出，docker logs 才能实时看到。
# PAPER_STORE_DIR 固定在数据卷挂载点内；compose 会把 ./data 挂到 /app/data。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH" \
    PAPER_STORE_DIR=/app/data/papers \
    PORT=8017

WORKDIR /app

# 只搬 venv。builder 里的 pip 缓存、uv 二进制、编译期临时文件都不进最终镜像。
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# 必须保持和仓库一致的层级：deps.py / main.py 用 `parents[4]` 推导仓库根。
# 源码放在 /app/apps/api/app 时，parents[4] 正好是 /app；
# 若放进 /app/app，parents[4] 会越界，WEB_DIR 与 data/papers 都会指错。
COPY --chown=appuser:appuser apps/api/app ./apps/api/app
COPY --chown=appuser:appuser apps/web ./apps/web

RUN mkdir -p /app/data/papers && chown -R appuser:appuser /app/data

# 以 apps/api 为工作目录，`uvicorn app.api.main:app` 才能导入 app 包；
# 同时 ROOT 仍然是 /app，静态前端与资料目录的位置都不变。
WORKDIR /app/apps/api

USER appuser

# 数据集目录声明为卷：不在镜像层里固化用户上传的 PDF。
VOLUME ["/app/data"]

# EXPOSE 只是文档性声明；真正监听端口由 CMD 里的 ${PORT:-8017} 决定。
EXPOSE 8017

# 用 Python 而不是 curl：slim 镜像没有 curl，少装一个包就少一处攻击面。
# /api/health 同时检查服务存活、文献数和是否配好模型。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8017') + '/api/health', timeout=3)"

# 应用代码不解析 PORT（main.py 只导出 app），所以必须由启动命令传给 uvicorn。
# exec 形式套 sh -c 有两个目的：让 ${PORT:-8017} 真的展开；
# 用 exec 让 uvicorn 成为 PID 1，docker stop 的信号能直接送达。
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8017}"]
