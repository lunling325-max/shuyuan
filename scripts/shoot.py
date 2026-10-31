"""用 CDP 拍界面截图（要点击、要暗色、要窄屏时，命令行 `--screenshot` 不够用）。

为什么不用 `chrome --headless --screenshot=…`：
它开不了暗色主题、点不了图上的节点，而且窗口宽度有 512px 的下限 ——
拍 390px 手机视图会被浏览器悄悄按 512px 渲染，看上去像页面被裁掉了一截。

用法:
    uv run --with websockets python scripts/shoot.py \
        --url http://127.0.0.1:8017/#graph --out docs/screenshots/live/07-graph-live.png
    … --dark            暗色主题（模拟 prefers-color-scheme）
    … --width 390       窄屏（同时改视口和 deviceScaleFactor）
    … --js "…"          加载完再执行一段 JS（用来点节点）
    … --wait 1200       执行 JS 之后再等多少毫秒（等动画/请求落地）
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def _wait_target(port: int, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as r:
                for target in json.load(r):
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        return target["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 —— Chrome 还没起完，继续等
            pass
        time.sleep(0.2)
    raise SystemExit("Chrome 没起来：拿不到调试端口")


class Cdp:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise SystemExit(f"{method} 失败: {msg['error']}")
                return msg.get("result", {})


def shoot(url: str, out: Path, width: int, height: int, dark: bool, js: str, wait: int,
          js2: str = "", wait2: int = 0) -> None:
    port = 9333
    profile = tempfile.mkdtemp(prefix="shoot-")
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        cdp = Cdp(_wait_target(port))
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride",
                 width=width, height=height, deviceScaleFactor=1, mobile=width < 600)
        if dark:
            cdp.call("Emulation.setEmulatedMedia",
                     features=[{"name": "prefers-color-scheme", "value": "dark"}])
        cdp.call("Page.navigate", url=url)
        time.sleep(1.6)  # 等首屏渲染 + 接口返回
        if js:
            res = cdp.call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
            value = res.get("result", {}).get("value")
            if res.get("exceptionDetails"):
                print(f"eval ✗ {res['exceptionDetails'].get('exception', {}).get('description', '')[:200]}")
            elif value not in (None, ""):
                print(f"eval → {value}")
            time.sleep(wait / 1000)
        if js2:
            res = cdp.call("Runtime.evaluate", expression=js2, awaitPromise=True, returnByValue=True)
            value = res.get("result", {}).get("value")
            print(f"eval2 → {value if not res.get('exceptionDetails') else '异常'}")
            time.sleep(wait2 / 1000)
        data = cdp.call("Page.captureScreenshot", format="png")["data"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        print(f"{out}  {out.stat().st_size // 1024} KB  {width}x{height}{'  暗色' if dark else ''}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--js", default="")
    ap.add_argument("--wait", type=int, default=700)
    ap.add_argument("--js2", default="", help="--wait 之后再执行一段，用来读第一段执行后的状态")
    ap.add_argument("--wait2", type=int, default=0)
    a = ap.parse_args()
    shoot(a.url, Path(a.out), a.width, a.height, a.dark, a.js, a.wait, a.js2, a.wait2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
