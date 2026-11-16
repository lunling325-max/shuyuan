#!/usr/bin/env bash
#
# 一条命令把本项目部署到云服务器（阿里云 ECS · Ubuntu 22.04）。
#
#   ./scripts/deploy_remote.sh                          # 默认 root@101.37.190.214
#   SERVER=root@1.2.3.4 ./scripts/deploy_remote.sh      # 换机器
#   PORT=8080 SERVER=root@1.2.3.4 ./scripts/deploy_remote.sh
#
# 做四件事，每件失败都停在原地并说清原因：
#   1. 本地打包：排除 .env / data / models / pdf / .git / .venv（密钥与大文件都不上去）
#   2. 流式传给服务器并解包到 /root/research-agent（不落中间文件）
#   3. 远端准备：确认 Docker 与 compose（缺则用阿里云镜像装）、加 2 GB swap、建 .env
#   4. 构建并起服务，等 /api/health 变成 ok
#
# **不包含 DEEPSEEK_API_KEY**：密钥只在服务器上由你手填，脚本从头到尾不碰它。
# 服务不填 key 也能起，只是"生成中文结论"这一步降级。

set -euo pipefail

SERVER="${SERVER:-root@101.37.190.214}"
TARGET="${TARGET:-/root/research-agent}"
PORT="${PORT:-8017}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '出错了：%s\n' "$*" >&2; exit 1; }

for f in Dockerfile docker-compose.yml .dockerignore .env.example apps/api/pyproject.toml apps/api/uv.lock; do
  [ -e "$ROOT/$f" ] || die "仓库里缺 $f，请确认在项目根目录执行"
done

say "1/4 打包（排除 .env / data / models / pdf / .git / .venv）"
cd "$ROOT"
tar czf - \
  --exclude='.git' --exclude='.env' \
  --exclude='data' --exclude='models' --exclude='pdf' --exclude='.cache' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='.pytest_cache' \
  Dockerfile docker-compose.yml .dockerignore .env.example \
  apps/api/pyproject.toml apps/api/uv.lock apps/api/app apps/web \
  scripts docs README.md \
  | ssh "$SERVER" "mkdir -p '$TARGET' && tar xz -C '$TARGET'"
# 故意不 rm -rf：$TARGET/data 里是已上传的文献（bind mount），重跑脚本不能把它们删掉。
# 同名文件直接覆盖，旧版本多出来的文件留着也无害。

say "2/4 传完。检查远端 Docker / swap / .env"
ssh "$SERVER" "TARGET='$TARGET' PORT='$PORT' bash -s" <<'REMOTE'
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "没装 docker，用阿里云镜像装（约 1 分钟）…"
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg git
  curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg \
    | gpg --dearmor -o /usr/share/keyrings/docker.gpg
  echo "deb [signed-by=/usr/share/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu jammy stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "缺 compose 插件，装一下…"
  apt-get install -y -qq docker-compose-plugin
fi
echo "docker: $(docker --version)"
echo "compose: $(docker compose version --short 2>/dev/null || echo unknown)"

# swap：首次构建峰值内存会冲高，宁可慢一点也别被 OOM 杀
# 不用 mkswap -q：Ubuntu 22.04 的 util-linux 2.37 没这个选项，会直接报 invalid option
if ! swapon --show | grep -q .; then
  if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
    chmod 600 /swapfile
  fi
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "swap 已启用：$(swapon --show=SIZE --noheadings | tr -d ' ')"
fi

cd "$TARGET"

# 容器内以 appuser(10001) 运行，而 ./data 是 bind mount、宿主机上是 root 建的，
# 不 chown 就会 PermissionError: '/app/data/papers' 然后无限重启（E-020）。
mkdir -p data/papers && chown -R 10001:10001 data

[ -f .env ] || cp .env.example .env
if grep -q '^DEEPSEEK_API_KEY=$' .env; then
  echo "提醒：.env 里的 DEEPSEEK_API_KEY 还是空的 —— 服务能起，但生成式问答会降级。"
fi

# compose 低于 2.24 时不认 env_file 的 required 长语法，就地换成短写法
if ! docker compose config -q >/dev/null 2>&1; then
  echo "compose 校验没过，改用兼容写法（.env 已存在，短语法足够）"
  python3 - <<'PY'
import re, pathlib
p = pathlib.Path("docker-compose.yml")
text = p.read_text(encoding="utf-8")
text = re.sub(r"\n\s*env_file:\s*\n\s*- path: \.env\s*\n\s*required: false", "\n    env_file:\n      - .env", text)
p.write_text(text, encoding="utf-8")
PY
fi
REMOTE

say "3/4 构建镜像并启动（首次 3-6 分钟）"
ssh "$SERVER" "cd '$TARGET' && docker compose up -d --build"

say "4/4 等 /api/health"
for i in $(seq 1 60); do
  body="$(ssh "$SERVER" "curl -fsS http://127.0.0.1:${PORT}/api/health 2>/dev/null || true" || true)"
  if printf '%s' "$body" | grep -q '"ok"'; then
    echo "健康检查通过：$body"
    HOST="${SERVER#*@}"
    printf '\n打开 http://%s:%s/\n' "$HOST" "$PORT"
    printf '接口文档 http://%s:%s/docs\n' "$HOST" "$PORT"

    # 健康检查只看"服务活着"，不看"赛题要的素材在不在"。
    # E-063 的教训：服务健康、本地四项齐全，线上却连 541 页教材都没有。
    say "覆盖核查：赛题 3.1 / 3.2 的四份素材"
    if PYTHONIOENCODING=utf-8 python3 "$ROOT/scripts/check_testcases.py" --server "http://$HOST:$PORT" 2>/dev/null         || PYTHONIOENCODING=utf-8 python "$ROOT/scripts/check_testcases.py" --server "http://$HOST:$PORT"; then
      printf '
素材齐全，可以按赛题逐项验收。
'
    else
      printf '
[1;33m部署成功，但测试用例素材不全 —— 评审按赛题点开会对不上。[0m
'
      printf '补法：./scripts/seed_demo.py <目录> --server http://%s:%s
' "$HOST" "$PORT"
    fi
    # 只有还缺密钥时才提示；已经配好的机器重跑脚本不该再催一遍
    if ! printf '%s' "$body" | grep -q '"llm_configured": *true'; then
      printf '\n还差一步：在服务器上填密钥\n  ssh %s "nano %s/.env"   # 填 DEEPSEEK_API_KEY\n  ssh %s "cd %s && docker compose up -d --force-recreate api"\n' \
        "$SERVER" "$TARGET" "$SERVER" "$TARGET"
      # 注意是 --force-recreate 而不是 restart：env_file 只在创建容器时读一次，
      # restart 沿用旧环境变量，改完 key 仍然 llm_configured=false（E-020 实测）。
    fi
    exit 0
  fi
  sleep 5
done

echo "健康检查 5 分钟没通过。看日志："
echo "  ssh $SERVER \"cd $TARGET && docker compose logs --tail=80 api\""
exit 1
