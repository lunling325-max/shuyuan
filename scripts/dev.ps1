# 一条命令把项目跑起来（Windows PowerShell）。
#
#   .\scripts\dev.ps1              # 默认 8017 端口
#   .\scripts\dev.ps1 -Port 8020   # 换端口
#
# 只做四件事，每件失败都停在这件上并说清怎么办：
#   1. 确认 uv 装了（没有就报错，不偷偷去下）
#   2. 没有 .env 就从 .env.example 生成一份，并说清"没填 key 会怎样"
#   3. uv sync 装依赖（走 pyproject.toml 里固定的清华镜像）
#   4. 起 uvicorn，前端由同一个进程托管
#
# 刻意不做的：不问任何交互问题、不自动下载模型（128MB 的向量模型是可选的，
# 缺了就走纯词法 BM25，服务照样能跑）、不改动 .env 里已有的任何值。
#
# 本文件存为 UTF-8 with BOM：Windows PowerShell 5.1 读无 BOM 的 UTF-8 时
# 会按 ANSI 解释，中文注释和提示全变乱码。

param(
    [int]$Port = 8017
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Api = Join-Path $Root 'apps/api'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '没找到 uv。装一次就行：' -ForegroundColor Red
    Write-Host '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host '装完重开一个终端，再跑一次这个脚本。'
    exit 1
}

$EnvFile = Join-Path $Root '.env'
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Root '.env.example') $EnvFile
    Write-Host '已从 .env.example 生成 .env。'
    Write-Host '  填了 DEEPSEEK_API_KEY 才有生成式问答；不填也能启动，'
    Write-Host '  依赖模型的接口会返回 200 + degraded（检索、溯源、OCR、图表都照常）。'
}

if (-not (Test-Path (Join-Path $Root 'models/bge-small-en-v1.5'))) {
    Write-Host '没有 models/bge-small-en-v1.5 —— 向量检索通道关闭，默认走纯词法 BM25（够用）。'
}

Write-Host '安装依赖（首次约 1-2 分钟）…'
Push-Location $Api
try {
    & uv sync --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'uv sync 失败，往上翻第一条报错。' -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host "服务地址：http://127.0.0.1:$Port/   接口文档：http://127.0.0.1:$Port/docs"
Write-Host '按 Ctrl+C 停止。'
Write-Host ''

Push-Location $Api
try {
    & uv run uvicorn app.api.main:app --reload --port $Port
} finally {
    Pop-Location
}
