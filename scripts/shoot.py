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
    … --keys Tab,Enter  --wait 之后真的按键盘（走 CDP，事件是 trusted 的）。
                        合成 KeyboardEvent 触发不了原生按钮的默认行为，
                        测"键盘能不能用"必须用这个
    … --wait2 5000      再等多少毫秒之后执行 --js2（等 --js 触发的事做完）
    … --js2 "…"         读执行完的状态，和 --js 的返回值对照看
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

# `--js` 里最长的等待是等一篇综述/一次 Agent 执行跑完，设 5 分钟。
# 注意 Chrome 那边的默认上限很短（约 30 秒），不显式给就会中途报错。
EVAL_TIMEOUT_MS = 300_000


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


# 键名 -> (windowsVirtualKeyCode, key, code)
KEY_CODES = {
    "Tab": (9, "Tab", "Tab"),
    "Enter": (13, "Enter", "Enter"),
    "Escape": (27, "Escape", "Escape"),
    "Space": (32, " ", "Space"),
    "ArrowLeft": (37, "ArrowLeft", "ArrowLeft"),
    "ArrowUp": (38, "ArrowUp", "ArrowUp"),
    "ArrowRight": (39, "ArrowRight", "ArrowRight"),
    "ArrowDown": (40, "ArrowDown", "ArrowDown"),
}


def press_keys(cdp: "Cdp", spec: str) -> None:
    """真的按键盘（走 CDP 输入通道，事件是 trusted 的）。

    为什么非要用真按键：`element.dispatchEvent(new KeyboardEvent('keydown'))`
    能被 JS 监听器收到，但**不会**触发浏览器给原生按钮的默认行为 ——
    <button> 上的 Enter 是浏览器把它翻译成 click 的，合成事件没这待遇。
    用合成事件测"键盘能不能用"，原生的永远测不出来（会误报成坏了）。
    """
    for name in (n.strip() for n in spec.split(",")):
        if not name:
            continue
        if name not in KEY_CODES:
            raise SystemExit(f"--keys 不认识的键：{name}（可选：{', '.join(KEY_CODES)}）")
        vk, key, code = KEY_CODES[name]
        for kind in ("keyDown", "keyUp"):
            params = dict(type=kind, windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk,
                          key=key, code=code)
            if kind == "keyDown" and name in ("Enter", "Space"):
                params["text"] = "\r" if name == "Enter" else " "
            cdp.call("Input.dispatchKeyEvent", **params)
            time.sleep(0.05)
        time.sleep(0.08)


def shoot(url: str, out: Path, width: int, height: int, dark: bool, js: str, wait: int,
          js2: str = "", wait2: int = 0, keys: str = "") -> None:
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
        # 两样都做：模拟媒体查询对"跟随系统的样式"有效，而这个应用的主题
        # 是 `<html data-theme>` 上的一个属性（存在 localStorage 里），
        # 只看媒体查询会拍出一张"自称暗色、其实是浅色"的图。
        if dark:
            cdp.call("Emulation.setEmulatedMedia",
                     features=[{"name": "prefers-color-scheme", "value": "dark"}])
        cdp.call("Page.navigate", url=url)
        time.sleep(1.6)  # 等首屏渲染 + 接口返回
        if dark:
            # 有主题按钮就点它（这样按钮上的文案、aria 也一起变，图里不会出现
            # "明明是暗色却还写着深色"）；没有就直接改属性。
            cdp.call("Runtime.evaluate", expression=(
                "(function(){if(document.documentElement.dataset.theme!=='dark'){"
                "var b=document.getElementById('theme');"
                "if(b){b.click();}else{document.documentElement.dataset.theme='dark';}}})()"
            ))
            time.sleep(0.2)
        if js:
            # timeout 必须显式给：`awaitPromise` 的默认上限远小于"等模型写完一篇综述"，
            # 超时后 CDP 报 "Inspected target navigated or closed"，看着像页面崩了
            res = cdp.call("Runtime.evaluate", expression=js, awaitPromise=True,
                           returnByValue=True, timeout=EVAL_TIMEOUT_MS)
            value = res.get("result", {}).get("value")
            if res.get("exceptionDetails"):
                print(f"eval ✗ {res['exceptionDetails'].get('exception', {}).get('description', '')[:200]}")
            elif value not in (None, ""):
                print(f"eval → {value}")
            time.sleep(wait / 1000)
        if keys:
            # 放在 --wait 之后：先让 --js 把焦点摆好，按键才有地方落
            press_keys(cdp, keys)
        if js2:
            # wait2 是"等 js 触发的那件事做完"再读状态，所以必须先睡再执行。
            # 以前睡在 js2 之后——睡的是截图之前那一下，等于什么也没等，
            # 上传/生成这类慢操作永远读到"进行中"。
            time.sleep(wait2 / 1000)
            res = cdp.call("Runtime.evaluate", expression=js2, awaitPromise=True,
                           returnByValue=True, timeout=EVAL_TIMEOUT_MS)
            value = res.get("result", {}).get("value")
            print(f"eval2 → {value if not res.get('exceptionDetails') else '异常'}")
        data = cdp.call("Page.captureScreenshot", format="png")["data"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        print(f"{out}  {out.stat().st_size // 1024} KB  {width}x{height}{'  暗色' if dark else ''}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    # 页面里的返回值什么都可能有：`²`、`ℓ`、`→`。Windows 控制台默认 GBK，
    # 打印时直接 UnicodeEncodeError，截图已经拍完却拿不到 --js 的结果。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
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
    ap.add_argument("--keys", default="", help="--wait 之后按的键，逗号分隔，如 Tab,Enter")
    a = ap.parse_args()
    shoot(a.url, Path(a.out), a.width, a.height, a.dark, a.js, a.wait, a.js2, a.wait2, a.keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
