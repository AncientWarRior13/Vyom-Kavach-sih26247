import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_DIR = r"C:\Users\swast\vyomkavach-drone\screenshots"
TEMP_PROFILE = r"C:\Users\swast\AppData\Local\Temp\edge_verify_profile"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_PROFILE, exist_ok=True)

RESOLUTIONS = [
    (1280, 720),
    (1366, 768),
    (1920, 900),
    (1920, 1080)
]

SECTIONS = [
    {"id": "hero", "name": "00 Hero", "side": "left"},
    {"id": "detect", "name": "01 Detect", "side": "right"},
    {"id": "respond", "name": "02 Respond", "side": "left"},
    {"id": "swarm", "name": "03 Swarm", "side": "right"},
    {"id": "sensors", "name": "04 Sensors", "side": "left"},
    {"id": "review", "name": "05 Review", "side": "right"},
    {"id": "modes", "name": "06 Modes", "side": "left"},
    {"id": "land", "name": "07 Land", "side": "center"}
]

def rects_intersect(r1, r2):
    if r1['right'] <= r2['left'] or r1['left'] >= r2['right']:
        return False
    if r1['bottom'] <= r2['top'] or r1['top'] >= r2['bottom']:
        return False
    return True

def rect_in_viewport(r, vw, vh, tolerance=1.0):
    return (
        r['left'] >= -tolerance and
        r['top'] >= -tolerance and
        r['right'] <= vw + tolerance and
        r['bottom'] <= vh + tolerance
    )

class CDPClient:
    def __init__(self, ws):
        self.ws = ws
        self.msg_id = 0

    async def call(self, method, params=None):
        self.msg_id += 1
        req = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(req))
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("id") == self.msg_id:
                return resp.get("result", {})

    async def eval_js(self, expr):
        res = await self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return res.get("result", {}).get("value")

async def run_verification():
    print("=" * 70)
    print("LANDING PAGE OVERLAY VERIFICATION SUITE")
    print("=" * 70)

    # Launch Edge with remote debugging
    cmd = [
        EDGE,
        "--headless=new",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        f"--user-data-dir={TEMP_PROFILE}",
        "http://127.0.0.1:8090/?section=0"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.5)

    all_passed = True
    failures = []
    results_summary = []

    try:
        resp = urllib.request.urlopen("http://127.0.0.1:9222/json")
        targets = json.loads(resp.read().decode())
        ws_url = targets[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            cdp = CDPClient(ws)
            await cdp.call("Page.enable")

            # Wait for initial page ready
            for _ in range(20):
                hdr = await cdp.eval_js("!!document.getElementById('app-header')")
                if hdr:
                    break
                await asyncio.sleep(0.3)

            for width, height in RESOLUTIONS:
                print(f"\n--- Testing Resolution: {width}x{height} ---")
                await cdp.call("Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": False
                })
                await asyncio.sleep(0.3)

                for sec_idx, sec in enumerate(SECTIONS):
                    sec_id = sec["id"]
                    expected_side = sec["side"]

                    # Jump directly to section
                    await cdp.eval_js(f"window.__setSection({sec_idx})")
                    await asyncio.sleep(0.3)

                    # Extract getBoundingClientRect for all 4 overlay elements
                    extract_js = f"""
                    (() => {{
                        const header = document.getElementById('app-header')?.getBoundingClientRect();
                        const sectors = document.getElementById('sectors-box')?.getBoundingClientRect();
                        const panel = (document.getElementById('section-panel-{sec_id}') || document.querySelector('.active-section-panel'))?.getBoundingClientRect();
                        const telemetry = document.getElementById('telemetry-card')?.getBoundingClientRect();
                        return {{
                            vw: window.innerWidth,
                            vh: window.innerHeight,
                            header: header ? {{ left: header.left, top: header.top, right: header.right, bottom: header.bottom, width: header.width, height: header.height }} : null,
                            sectors: sectors ? {{ left: sectors.left, top: sectors.top, right: sectors.right, bottom: sectors.bottom, width: sectors.width, height: sectors.height }} : null,
                            panel: panel ? {{ left: panel.left, top: panel.top, right: panel.right, bottom: panel.bottom, width: panel.width, height: panel.height }} : null,
                            telemetry: telemetry ? {{ left: telemetry.left, top: telemetry.top, right: telemetry.right, bottom: telemetry.bottom, width: telemetry.width, height: telemetry.height }} : null
                        }};
                    }})()
                    """
                    data = await cdp.eval_js(extract_js)
                    if not data or not data.get("header") or not data.get("panel"):
                        # retry once
                        await asyncio.sleep(0.4)
                        data = await cdp.eval_js(extract_js)

                    vw = data["vw"]
                    vh = data["vh"]
                    header = data["header"]
                    sectors = data["sectors"]
                    panel = data["panel"]
                    telemetry = data["telemetry"]

                    rects = {
                        "header": header,
                        "sectors": sectors,
                        "panel": panel,
                        "telemetry": telemetry
                    }

                    # 1. Viewport boundary assertions
                    viewport_errors = []
                    for name, r in rects.items():
                        if r and not rect_in_viewport(r, vw, vh):
                            viewport_errors.append(f"{name} leaves viewport: {r}")

                    # 2. Pairwise intersection assertions
                    intersections = []
                    names = list(rects.keys())
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            n1, n2 = names[i], names[j]
                            r1, r2 = rects[n1], rects[n2]
                            if r1 and r2 and rects_intersect(r1, r2):
                                intersections.append(f"{n1} ({r1['left']:.0f},{r1['top']:.0f} to {r1['right']:.0f},{r1['bottom']:.0f}) intersects {n2} ({r2['left']:.0f},{r2['top']:.0f} to {r2['right']:.0f},{r2['bottom']:.0f})")

                    # 3. Side orientation check
                    side_errors = []
                    if expected_side == "left":
                        if panel["left"] > vw / 2:
                            side_errors.append(f"Expected left panel, but panel left is {panel['left']:.0f} > {vw/2}")
                    elif expected_side == "right":
                        if panel["right"] < vw / 2:
                            side_errors.append(f"Expected right panel, but panel right is {panel['right']:.0f} < {vw/2}")
                    elif expected_side == "center":
                        panel_mid = (panel["left"] + panel["right"]) / 2
                        vw_mid = vw / 2
                        if abs(panel_mid - vw_mid) > 30:
                            side_errors.append(f"Expected center panel, but mid is {panel_mid:.0f} vs {vw_mid:.0f}")

                    # Take Screenshot
                    ss_shot = await cdp.call("Page.captureScreenshot", {"format": "png"})
                    img_data = base64.b64decode(ss_shot["data"])
                    ss_filename = f"section_{sec_idx}_{sec_id}_{width}x{height}.png"
                    ss_path = os.path.join(OUT_DIR, ss_filename)
                    with open(ss_path, "wb") as f:
                        f.write(img_data)

                    passed = (len(viewport_errors) == 0 and len(intersections) == 0 and len(side_errors) == 0)
                    status_str = "PASS" if passed else "FAIL"
                    print(f"  Sec {sec_idx} [{sec['name']:<12}] side={expected_side:<6} -> {status_str} (panel: {panel['width']:.0f}x{panel['height']:.0f} at x={panel['left']:.0f}, y={panel['top']:.0f})")

                    if not passed:
                        all_passed = False
                        msg = f"Failure at {width}x{height} Sec {sec_idx} ({sec['name']}):"
                        if viewport_errors:
                            msg += "\n    Viewport: " + "; ".join(viewport_errors)
                        if intersections:
                            msg += "\n    Overlap: " + "; ".join(intersections)
                        if side_errors:
                            msg += "\n    Side: " + "; ".join(side_errors)
                        failures.append(msg)
                        print(msg)

                    results_summary.append({
                        "res": f"{width}x{height}",
                        "sec": sec_idx,
                        "name": sec["name"],
                        "side": expected_side,
                        "passed": passed,
                        "panel_pos": f"({panel['left']:.0f}, {panel['top']:.0f}, {panel['right']:.0f}, {panel['bottom']:.0f})",
                        "screenshot": ss_filename
                    })

    finally:
        proc.terminate()
        proc.wait()

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {'ALL TESTS PASSED!' if all_passed else 'FAILURES ENCOUNTERED'}")
    print(f"Total test cases executed: {len(results_summary)}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f)
    print("=" * 70)
    return all_passed, results_summary

if __name__ == "__main__":
    success, summary = asyncio.run(run_verification())
    import sys
    sys.exit(0 if success else 1)
