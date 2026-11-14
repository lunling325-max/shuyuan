# 部署到云服务器（阿里云 ECS · Ubuntu 22.04）

本文件按 2026-11-07 实际买下的那台机器写成，命令可以直接复制。
换机器也能照用，只需改 IP。

> **当前状态（2026-11-14 重新部署）**：<http://101.37.190.214:8017/>
> `{"ok":true,"papers":14,"llm_configured":true}`，安全组 8017 已放行。
> 一步步自动走完可用 `./scripts/deploy_remote.sh`。
> 这一版含 E-038 ~ E-058 的全部改动（跨文献对比、公式块、订阅追踪、口味排序、
> BibTeX 导出、窄屏修复等）；部署前打了回滚标签 `research-agent-api:prev-20261113`。

| 项 | 值 |
|---|---|
| 公网 IP | `101.37.190.214` |
| 地域 | 华东1（杭州） |
| 规格 | `ecs.u1-c1m2.large`（2 vCPU / 4 GiB / ESSD Entry 40 GiB） |
| 系统 | Ubuntu 22.04 64 位 |
| 计费 | 按量付费，公网按流量计费（中国大陆 20 GiB 免费额度） |
| 访问端口 | `8017` |

为什么用 8017 而不是 80：中国大陆地域，IP 直连 80 端口容易被拦，还牵扯域名备案。
非标准端口 + IP 直连是这类比赛 demo 最省事的一条路。

---

## 0. 只做一次：放行端口

控制台 → 实例 → 网络与安全组 → 安全组规则 → 入方向，加一条：

| 端口范围 | 授权对象 | 说明 |
|---|---|---|
| `8017/8017` | `0.0.0.0/0` | 演示用；访问结束后建议删掉 |
| `22/22` | 你的出口 IP | 不要留 `0.0.0.0/0` |

## 1. 登录并确认 Docker

```bash
ssh root@101.37.190.214
docker --version && docker compose version
```

两条都有输出就跳到第 3 步。若缺少 `docker compose`（预装镜像常只有引擎）：

```bash
# 用阿里云自己的 docker-ce 源，国内比 get.docker.com 稳
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg \
  | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [signed-by=/usr/share/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu jammy stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**版本要求**：`docker-compose.yml` 用了 `env_file` 的 `required: false` 长语法，
需要 **Compose v2.24+**。低于这个版本会报 `Additional property path is not allowed`，
那就把该段改成 `- .env`，并保证 `.env` 一定存在（第 4 步会建）。

## 2. 把代码放上去（两条路，选一条）

> 注意：`.env`、`data/`、`models/`、`pdf/` 都不该上去。
> `.env` 里有密钥，`data/` 是运行产物，`models/` 是 128 MB 的本地模型。

**路线 A：走 Git（推荐，后面更新只要 `git pull`）**

```bash
# 在你的开发机上把仓库推到你自己的 Gitee / GitHub（私有库即可）
# 然后在服务器上：
apt-get install -y git
git clone <你的仓库地址> research-agent
cd research-agent
```

**路线 B：本地打包上传**

```bash
# 开发机上（Git Bash / WSL / macOS）
tar --exclude=.git --exclude=data --exclude=models --exclude=pdf \
    --exclude=.env --exclude=__pycache__ --exclude='*.pyc' \
    -czf research-agent.tar.gz Dockerfile docker-compose.yml .dockerignore \
    apps scripts docs README.md .env.example

scp research-agent.tar.gz root@101.37.190.214:/root/
# 服务器上
tar -xzf /root/research-agent.tar.gz && cd /root
```

## 3. 加 2 GB swap（4 GiB 内存也建议加）

首次 `docker build` 要装依赖、解析大 PDF 时内存会冲高。swap 让它在最坏情况下变慢
而不是被 OOM 杀掉——演示现场最怕后者。

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
free -h      # 确认 Swap 一行有 2.0Gi
```

## 4. 建 `.env`（密钥只放这里，别写进 Dockerfile）

```bash
cp .env.example .env
nano .env        # 填 DEEPSEEK_API_KEY=sk-...
```

不填 key 也能起服务，但"生成中文结论"这一步会降级；检索、溯源校验、
OCR、图表解析照常工作。演示要完整效果就填上。

`SEMANTIC_SCHOLAR_API_KEY` 是可选的第二个 key，**建议填**：多源检索里的
Semantic Scholar 不带 key 也能查，但公共配额全站共享，从机房 IP 出去几乎必然
拿到 429（本机实测 5 站里就它一格红）。免费注册，填上之后是 1 次/秒的专属配额。

## 5. 起服务

```bash
# 先给数据目录正确的属主：容器内跑的是 appuser(10001)，而 ./data 由 root 创建，
# 不改就会 PermissionError: '/app/data/papers' 然后无限重启
mkdir -p data/papers && chown -R 10001:10001 data

docker compose up -d --build      # 首次 3-6 分钟，主要在装依赖
docker compose ps                 # 等 STATUS 变成 healthy
curl -s http://127.0.0.1:8017/api/health
```

期望输出类似：

```json
{"ok": true, "papers": 0, "llm_configured": true}
```

浏览器打开 <http://101.37.190.214:8017/> 就能看到界面。

## 6. 灌几篇文献（可选，但演示前必做）

两条都行：

- 界面上「＋ 上传 PDF」；
- 或者从开发机批量灌（`--server` 指向公网地址）：

```bash
uv run python scripts/seed_demo.py --server http://101.37.190.214:8017
```

上传的 PDF 落在服务器的 `./data/`，容器重建不丢。

> 重跑这个脚本时如果库里已经有这几篇，每次上传都是"已存在"的快路径（几十毫秒），
> 连着传会撞上**限流**（一次上传计 3 个令牌、突发额度 10）。脚本会按退避重试
> （2 秒、4 秒、6 秒，最多 4 次），所以你看到「被限流，N 秒后重试」是正常的；
> 真失败会打「失败」并在最后统计篇数。
> 想彻底避开：服务端设 `RATE_LIMIT_PER_MINUTE=0`（本地开发用），或分批灌。

## 7. 日常运维

```bash
docker compose logs -f api        # 实时日志
docker compose up -d --force-recreate api   # 改了 .env 之后必须用这个，不能用 restart
docker compose up -d --build      # 拉了新代码后重建
docker compose down               # 停掉（数据保留在 ./data）
docker stats                      # 看内存/CPU 占用
```

### 7.1 重新部署前先打一个回滚标签

`docker compose up -d --build` 会**覆盖** `research-agent-api:local` 这个标签。
构建成功但新容器起不来时，旧镜像就只剩构建缓存里那一份了。所以重建之前先留一份：

```bash
docker tag research-agent-api:local research-agent-api:prev-$(date +%Y%m%d)
```

回滚（30 秒，数据不动）：

```bash
docker tag research-agent-api:prev-20261113 research-agent-api:local
docker compose up -d --no-build --force-recreate api
curl -fsS http://127.0.0.1:8017/api/health
```

> 2026-11-13 已为线上打过 `research-agent-api:prev-20261113`（上线于 11-11 的那一版）。

### 7.2 arXiv 追踪的定时器

服务里有一个后台任务，每 60 秒看一眼有没有**到期的**订阅（`app/tracking.py`）。
它只读写 `./data/tracking.json`，没有订阅时是空转。要关掉：

```
PAPER_TRACKING_SCHEDULER=0
```

写在 `.env` 里，然后 `--force-recreate`（`restart` 不重读 env_file）。
单测里默认关着 —— 测试进程不该因为有人建了一条订阅就去连 arXiv。

## 8. 已知限制与对策

| 现象 | 原因 | 对策 |
|---|---|---|
| 容器一直 `Restarting`，日志是 `PermissionError: '/app/data/papers'` | `./data` 是 bind mount，宿主机由 root 创建，容器内是 uid 10001 | `chown -R 10001:10001 data` 后 `docker compose up -d --force-recreate api`（第 5 步已包含属主修复） |
| 上传扫描件返回 422「本机没有可用的 OCR 引擎」，但镜像里明明装了 rapidocr | `python:3.11-slim` 缺 OpenCV 依赖的 `libxcb.so.1` / `libGL.so.1`，`import cv2` 直接 ImportError，被业务层当成「没装 OCR」 | Dockerfile 的 runtime 层已补 `libgl1 libglib2.0-0 libxcb1`；自查：`docker compose exec -T api python -c "import cv2; print(cv2.__version__)"` |
| 扫描件第一次识别很慢 | RapidOCR 是 CPU 推理，实测单页 13–27 秒 | 演示前先把要用的扫描件跑一遍，结果会落盘缓存，现场就是秒开 |
| 向量检索不可用 | 镜像里不带 `models/bge-small-en-v1.5`（128 MB，未入库） | 默认走纯词法 BM25，够用；确实要用就把模型放到 `./models/` 并解开 compose 里那行注释 |
| 首屏 3 秒以上才打开 | 首次访问要加载前端与建索引 | 演示前先访问一次预热 |
| 「订阅追踪」点「立即抓取」要等十几秒 | 一次性向 arXiv 拉 40 条（约 100 KB），服务器到 arXiv 实测 13 KB/s 上下 | 属正常：按钮上带秒表，超过 90 秒会放弃并说明原因，订阅不受影响 |
| 公网流量被刷 | 20 GiB 免费额度，PDF 每篇 2–3 MB | 结束后删掉安全组里 8017 那条规则 |

## 9. 省钱 / 省额度

这台机器是**按量付费**，费用只在运行时产生。演示结束：

```bash
# 控制台点"停止"即可；要长期停用就先 docker compose down 再停机
docker compose down
```

试用额度会按小时扣（控制台显示已用 0.380 元/小时，免费额度 0.833 元/小时），
所以**不用的时候一定要关机**，别让它空跑。
