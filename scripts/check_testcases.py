"""核查一台服务上的文献库有没有覆盖赛题点名的测试用例。

    uv run python ../../scripts/check_testcases.py                      # 查线上
    uv run python ../../scripts/check_testcases.py --server http://127.0.0.1:8017

为什么要这个脚本：赛题 3.1/3.2 点名了四份文件（v1、v7、扫描件、541 页教材）。
「本地能演示」不等于「评审打开网站能用」—— 曾经在本地库里把这四份都验过，
线上库却连那本教材都没有，而所有测试与代码审查都发现不了这件事。
所以把它做成一条命令，本地与线上各跑一次，对照输出。

退出码 0 表示四项全覆盖，1 表示有缺口（可用于部署后的自动检查）。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_SERVER = "http://101.37.190.214:8017"


def fetch(server: str) -> list[dict]:
    with urllib.request.urlopen(f"{server.rstrip('/')}/api/papers", timeout=30) as r:
        return json.load(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核查测试用例覆盖情况")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    args = parser.parse_args(argv)

    try:
        papers = fetch(args.server)
    except Exception as exc:  # noqa: BLE001 —— 连不上就是结论本身，不需要区分异常类型
        print(f"连不上 {args.server}：{exc}")
        return 2

    attn = [p for p in papers if p.get("arxiv_id") == "1706.03762"
            or p["title"].startswith("Attention Is All You Need")]

    def has_versions() -> tuple[bool, bool]:
        v1 = any(p.get("arxiv_version") == 1 and not p.get("scanned") for p in attn)
        v7 = any(p.get("arxiv_version") == 7 for p in attn)
        return v1, v7

    v1, v7 = has_versions()
    scanned = [p for p in attn if p.get("scanned")]
    long_doc = [p for p in papers if p["pages"] >= 500]

    checks = [
        ("3.1 去重用例 · v1（矢量）", v1,
         "上传 v7 后再传 v1，应出现「库内已有同一篇的另一个版本」"),
        ("3.1 去重用例 · v7", v7, "需要与 v1 同时存在才能触发预警与版本对比"),
        ("3.2 扫描件用例", bool(scanned), "无文字层，走 OCR；目录会标「扫描件」"),
        ("3.2 长篇教材用例（≥500 页）", bool(long_doc), "百页以上大体量文档的切块与检索"),
    ]

    print(f"服务：{args.server}    文献 {len(papers)} 篇，合计 {sum(p['pages'] for p in papers)} 页")
    print("-" * 68)
    missing = 0
    for name, ok, why in checks:
        # 用 ASCII 记号：Windows 控制台是 GBK，✓/✗ 会抛 UnicodeEncodeError，
        # 于是"检查通过"被脚本判成"检查失败"，误报素材不全（真踩过）
        print(f"{'[OK] ' if ok else '[缺失]'} {name:<26} {'就绪' if ok else '缺失'}")
        if not ok:
            missing += 1
            print(f"    需要：{why}")
    if long_doc:
        p = max(long_doc, key=lambda x: x["pages"])
        print(f"\n最长的文档：{p['title']} · {p['pages']} 页 / {p['chunks']} 块")
    if scanned:
        p = scanned[0]
        print(f"扫描件：{p['title']} · v{p.get('arxiv_version')} · {p['pages']} 页 / {p['chunks']} 块")

    print("-" * 68)
    print("四项全覆盖" if not missing else f"缺 {missing} 项 —— 评审按赛题点开时会对不上")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
