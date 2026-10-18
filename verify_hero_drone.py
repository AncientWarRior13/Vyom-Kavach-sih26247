import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.request
import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_DIR = r"C:\Users\swast\vyomkavach-drone\screenshots"
TEMP_PROFILE = r"C:\Users\swast\AppData\Local\Temp\edge_hero_profile"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_PROFILE, exist_ok=True)

RESOLUTIONS = [
    (1280, 720),
    (1366, 768),
    (1920, 1080),
    (800, 600)  # Responsive under 900px test
]

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

async def run_hero_verification():
    print("=" * 70, flush=True)
    print("HERO SECTION DRONE CLEAR ZONE VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    cmd = [
        EDGE,
        "--headless=new",
        "--disable-gpu",
        "--remote-debugging-port=9226",
        f"--user-data-dir={TEMP_PROFILE}",
        "http://127.0.0.1:8090/?section=0"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.5)

    all_passed = True

    try:
        resp = urllib.request.urlopen("http://127.0.0.1:9226/json")
        targets = json.loads(resp.read().decode())
        ws_url = targets[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            cdp = CDPClient(ws)
            await cdp.call("Page.enable")

            # Wait for app-header
            for _ in range(20):
                hdr = await cdp.eval_js("!!document.getElementById('app-header')")
                if hdr:
                    break
                await asyncio.sleep(0.3)

            for width, height in RESOLUTIONS:
                print(f"\n--- Testing Resolution: {width}x{height} ---", flush=True)
                await cdp.call("Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": (width < 900)
                })
                await asyncio.sleep(0.3)

                # Set section 0 (hero)
                await cdp.eval_js("window.__setSection && window.__setSection(0)")
                await asyncio.sleep(0.4)

                extract_js = """
                (() => {
                    try {
                        const header = document.getElementById('app-header')?.getBoundingClientRect();
                        const sectors = document.getElementById('sectors-box')?.getBoundingClientRect();
                        const card = (document.getElementById('section-panel-hero') || document.querySelector('.active-section-panel'))?.getBoundingClientRect();
                        const telemetry = document.getElementById('telemetry-card')?.getBoundingClientRect();
                        const headlineEl = document.querySelector('#section-panel-hero h2');
                        const headlineFontSize = headlineEl ? window.getComputedStyle(headlineEl).fontSize : '';
                        const overlayContainer = document.querySelector('.section-overlay-container');
                        const overlayPE = overlayContainer ? window.getComputedStyle(overlayContainer).pointerEvents : '';
                        const cardEl = document.getElementById('section-panel-hero');
                        const cardPE = cardEl ? window.getComputedStyle(cardEl).pointerEvents : '';
                        const sectorsEl = document.getElementById('sectors-box');
                        const sectorsPE = sectorsEl ? window.getComputedStyle(sectorsEl).pointerEvents : '';

                        return {
                            vw: window.innerWidth,
                            vh: window.innerHeight,
                            overlayPE,
                            cardPE,
                            sectorsPE,
                            headlineFontSize,
                            header: header ? { left: header.left, top: header.top, right: header.right, bottom: header.bottom, width: header.width, height: header.height } : null,
                            sectors: sectors ? { left: sectors.left, top: sectors.top, right: sectors.right, bottom: sectors.bottom, width: sectors.width, height: sectors.height } : null,
                            card: card ? { left: card.left, top: card.top, right: card.right, bottom: card.bottom, width: card.width, height: card.height } : null,
                            telemetry: telemetry ? { left: telemetry.left, top: telemetry.top, right: telemetry.right, bottom: telemetry.bottom, width: telemetry.width, height: telemetry.height } : null
                        };
                    } catch (e) {
                        return { error: e.toString() };
                    }
                })()
                """
                data = await cdp.eval_js(extract_js)
                vw = data["vw"]
                vh = data["vh"]
                sectors = data["sectors"]
                card = data["card"]
                telemetry = data["telemetry"]
                header = data["header"]

                print(f"  Header: {header['width']:.0f}x{header['height']:.0f} at y={header['top']:.0f}..{header['bottom']:.0f}", flush=True)
                print(f"  Sectors: {sectors['width']:.0f}x{sectors['height']:.0f} at x={sectors['left']:.0f}..{sectors['right']:.0f}, y={sectors['top']:.0f}..{sectors['bottom']:.0f}", flush=True)
                print(f"  Hero Card: {card['width']:.0f}x{card['height']:.0f} at x={card['left']:.0f}..{card['right']:.0f}, y={card['top']:.0f}..{card['bottom']:.0f}", flush=True)
                print(f"  Headline font-size: {data['headlineFontSize']}", flush=True)
                print(f"  Pointer-events: overlay={data['overlayPE']}, card={data['cardPE']}, sectors={data['sectorsPE']}", flush=True)

                # Assertions
                errors = []

                # 1. Pointer events check
                if data["overlayPE"] != "none":
                    errors.append(f"Overlay wrapper pointer-events should be 'none', got '{data['overlayPE']}'")
                if data["cardPE"] != "auto":
                    errors.append(f"Card pointer-events should be 'auto', got '{data['cardPE']}'")
                if data["sectorsPE"] != "auto":
                    errors.append(f"Sectors pointer-events should be 'auto', got '{data['sectorsPE']}'")

                if width >= 900:
                    # 2. Sectors top-left check
                    if sectors["top"] < 70 or sectors["top"] > 90:
                        errors.append(f"Sectors top should be ~80px (24px below 56px header), got {sectors['top']:.0f}")
                    if sectors["left"] < 20 or sectors["left"] > 30:
                        errors.append(f"Sectors left should be ~24px, got {sectors['left']:.0f}")
                    if sectors["width"] < 210 or sectors["width"] > 230:
                        errors.append(f"Sectors width should be ~220px, got {sectors['width']:.0f}")

                    # 3. Main text card bottom-left check
                    dist_from_bottom = vh - card["bottom"]
                    if dist_from_bottom < 20 or dist_from_bottom > 30:
                        errors.append(f"Card should be ~24px from bottom, got {dist_from_bottom:.0f}")
                    if card["left"] < 20 or card["left"] > 30:
                        errors.append(f"Card left should be ~24px, got {card['left']:.0f}")
                    if card["width"] > 425:
                        errors.append(f"Card max-width should be <= 420px, got {card['width']:.0f}")

                    # 4. Drone clear zone check (center 40% horizontally)
                    # Center 40% is from 30% to 70% of viewport width
                    center_start = vw * 0.30 + 1.0
                    center_end = vw * 0.70 - 1.0
                    if card["right"] > center_start:
                        errors.append(f"Card extends into center clear zone: card right is {card['right']:.0f} > {center_start:.0f} (30% mark)")
                    if sectors["right"] > center_start:
                        errors.append(f"Sectors extends into center clear zone: sectors right is {sectors['right']:.0f} > {center_start:.0f}")

                    # 5. Middle vertical band check (between Sectors bottom and Card top)
                    middle_band_gap = card["top"] - sectors["bottom"]
                    if middle_band_gap < 50:
                        errors.append(f"Vertical middle band should be open, gap between sectors and card is only {middle_band_gap:.0f}px")
                    else:
                        print(f"  Vertical middle band clear gap: {middle_band_gap:.0f}px (Sectors bottom={sectors['bottom']:.0f}, Card top={card['top']:.0f})", flush=True)

                else:
                    # Under 900px responsive check
                    # Sectors at top, Card at bottom
                    if sectors["top"] > 80:
                        errors.append(f"On <900px, sectors should be stacked at top, got top={sectors['top']:.0f}")
                    dist_from_bottom = vh - card["bottom"]
                    if dist_from_bottom > 25:
                        errors.append(f"On <900px, card should be at bottom, got {dist_from_bottom:.0f}px from bottom")
                    # Make sure middle is open
                    middle_gap = card["top"] - sectors["bottom"]
                    if middle_gap < 50:
                        errors.append(f"On <900px, middle should remain clear, gap is {middle_gap:.0f}px")
                    else:
                        print(f"  Under 900px middle clear gap: {middle_gap:.0f}px", flush=True)

                # Capture screenshot
                ss_shot = await cdp.call("Page.captureScreenshot", {"format": "png"})
                img_data = base64.b64decode(ss_shot["data"])
                ss_filename = f"hero_drone_clear_{width}x{height}.png"
                ss_path = os.path.join(OUT_DIR, ss_filename)
                with open(ss_path, "wb") as f:
                    f.write(img_data)
                print(f"  Screenshot saved: {ss_filename}", flush=True)

                if errors:
                    all_passed = False
                    print(f"  RESULT: FAIL", flush=True)
                    for err in errors:
                        print(f"    - {err}", flush=True)
                else:
                    print(f"  RESULT: PASS (All clear zone and positioning requirements met!)", flush=True)

    finally:
        proc.terminate()
        proc.wait()

    print("\n" + "=" * 70, flush=True)
    print(f"HERO VERIFICATION SUMMARY: {'ALL TESTS PASSED!' if all_passed else 'FAILURES FOUND'}", flush=True)
    print("=" * 70, flush=True)
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(run_hero_verification())
    import sys
    sys.exit(0 if success else 1)
