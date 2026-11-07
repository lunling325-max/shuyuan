#!/usr/bin/env bash
#
# 一条命令把项目跑起来（macOS / Linux / Git Bash）。
#
#   ./scripts/dev.sh              # 默认 8017 端口
#   PORT=8020 ./scripts/dev.sh    # 换端口
#
# 只做四件事，每件失败都停在这件上并说清怎么办：
#   1. 确认 uv 装了（没有就报错，不偷偷去下）
#   2. 没有 .env 就从 .env.example 生成一份，并说清"没填 key 会怎样"
#   3. uv sync 装依赖（走 pyproject.toml 里固定的清华镜像）
#   4. 起 uvicorn，前端由同一个进程托管
#
# 刻意不做的：不问任何交互问题、不自动下载模型（128MB 的 vectors 模型是可选的，
# 缺了就走纯词法 BM25，服务照样能跑）、不改动 .env 里已有的任何值。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="$ROOT/apps/api"
PORT="${PORT:-8017}"

say() { printf '%s\n' "$*"; }
die() { printf '出错了：%s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "没找到 uv。装一次就行：
  Windows:  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"
  macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 \
  || say "提醒：没找到 python，uv 会自己挑一个（首次可能要下载）"

if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  say "已从 .env.example 生成 .env。"
  say "  填了 DEEPSEEK_API_KEY 才有生成式问答；不填也能启动，"
  say "  依赖模型的接口会返回 200 + degraded（检索、溯源、OCR、图表都照常）。"
fi

if [ ! -d "$ROOT/models/bge-small-en-v1.5" ]; then
  say "没有 models/bge-small-en-v1.5 —— 向量检索通道关闭，默认走纯词法 BM25（够用）。"
fi

say "安装依赖（首次约 1-2 分钟）…"
(cd "$API" && uv sync --quiet)

say ""
say "服务地址：http://127.0.0.1:${PORT}/   接口文档：http://127.0.0.1:${PORT}/docs"
say "按 Ctrl+C 停止。"
say ""

cd "$API"
exec uv run uvicorn app.api.main:app --reload --port "$PORT"
