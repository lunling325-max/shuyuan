"""验一件事：问答结论里点「跳转到原文」之后，正文里高亮的**是不是那几个字**。

这是我们整个产品的核心承诺（"每句话都能点回原文"），所以给它一个可重复的检查，
而不是每次靠人眼看截图。检查三样：

1. 落点页码对不对（`#page-mark`）；
2. 正文里有没有 `<mark>`，高亮的文本是不是引文本身（或它的最长可命中前缀）；
3. 落在哪一节（`#page-title`）—— 页对、节错一样会误导人。

为什么单独写一个脚本而不塞进 pytest：它要真开一个浏览器、真调一次大模型。
单测里既没有浏览器也不该花钱；这个脚本按需手动跑。

用法（起好服务后）：
    uv run --with websockets python scripts/check_jump.py --url http://127.0.0.1:8017/

它不是断言工具，是"读数的工具"：把落点、节名、高亮文本打出来，人看一眼就知道对不对。
"""
import base64, json, shutil, subprocess, tempfile, time, urllib.request
from websockets.sync.client import connect

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def wait_target(port, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as r:
                for t in json.load(r):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.2)
    raise SystemExit("no chrome")

class Cdp:
    def __init__(self, url):
        self.ws = connect(url, max_size=64*1024*1024); self.n = 0
    def call(self, m, **p):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": m, "params": p}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg: raise SystemExit(f"{m}: {msg['error']}")
                return msg.get("result", {})
    def js(self, e):
        r = self.call("Runtime.evaluate", expression=e, awaitPromise=True, returnByValue=True, timeout=180000)
        if r.get("exceptionDetails"):
            return "ERR " + str(r["exceptionDetails"].get("exception", {}).get("description", ""))[:150]
        return r.get("result", {}).get("value")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8017/")
ap.add_argument("--out", default="")
args = ap.parse_args()

port = 9336
prof = tempfile.mkdtemp(prefix="hl-")
proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}",
    f"--user-data-dir={prof}", "--no-first-run", "--hide-scrollbars", "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    c = Cdp(wait_target(port))
    c.call("Page.enable"); c.call("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1.4, mobile=False)
    c.call("Page.navigate", url=args.url)
    time.sleep(3)
    print("选一篇非扫描件：", c.js("""(() => {
      const rows=[...document.querySelectorAll('.rowopen')].filter(n=>n.textContent.includes('Attention Is All You Need'));
      const row=rows.map(r=>r.closest('tr')).find(tr=>!tr.textContent.includes('扫描件'));
      row.querySelector('.rowopen').click(); return row.textContent.slice(0,40);
    })()"""))
    time.sleep(2.5)
    print("提问：", c.js("""(async () => {
      const i=document.getElementById('qa-input'); i.value='How many attention heads does the Transformer use?';
      document.getElementById('qa-send').click();
      await new Promise(r=>setTimeout(r,20000));
      return document.getElementById('qa-chip').textContent;
    })()"""))
    print("跳转前高亮数：", c.js("document.querySelectorAll('#page-text mark, #page-text .hl, #page-text [data-hl]').length"))
    print("点跳转：", c.js("""(async () => {
      const a=[...document.querySelectorAll('#qa-body button, #qa-body a')].find(n=>/跳转|原文/.test(n.textContent));
      if(!a) return '没有跳转按钮';
      a.click(); await new Promise(r=>setTimeout(r,1500));
      return document.getElementById('page-mark').textContent;
    })()"""))
    time.sleep(1)
    info = c.js("""(() => {
      const marks=[...document.querySelectorAll('#page-text mark, #page-text .hl, #page-text [data-hl]')];
      return JSON.stringify({
        markCount: marks.length,
        texts: marks.slice(0,3).map(m=>m.textContent.slice(0,80)),
        flashNodes: document.querySelectorAll('.flash').length,
        pageMark: document.getElementById('page-mark').textContent
      });
    })()""")
    print("跳转后：", info)
    print("节名：", c.js("document.getElementById('page-title').textContent"))
    data = c.call("Page.captureScreenshot", format="png")["data"] if args.out else ""
    if args.out:
        open(args.out, "wb").write(base64.b64decode(data))
        print("截图 →", args.out)
finally:
    proc.terminate(); proc.wait(timeout=10); shutil.rmtree(prof, ignore_errors=True)
