"""
Cross‑platform client to start a dispense job & display live progress bar.
"""

import argparse
import json
import requests
import sys
import time
from websocket import create_connection
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="armbian.local", help="Pump daemon host or IP")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--volume", type=float, required=True, help="Target volume (mL)")
    p.add_argument("--rpm", type=int, default=30)
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}"

    # 1. POST dispense
    resp = requests.post(
        f"{base}/pump/dispense",
        json={"volume": args.volume, "rpm": args.rpm, "direction": 0},
        timeout=5,
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"Started run_id={run_id}")

    # 2. open WebSocket
    ws = create_connection(f"{ws_url}/ws/pump/{run_id}", timeout=5)
    bar = tqdm(total=100, ncols=70)

    try:
        while True:
            raw = ws.recv()
            if not raw:
                break
            msg = json.loads(raw)
            if msg["state"] in ("running", "started"):
                bar.n = int(msg["percent"] * 100)
                bar.refresh()
            elif msg["state"] == "finished":
                bar.n = 100
                bar.refresh()
                print(f"\nDone! Delivered {msg['ml']:.2f} mL in {msg['elapsed']:.1f}s.")
                break
            elif msg["state"] == "error":
                print("Error:", msg["msg"])
                break
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        ws.close()
        bar.close()

if __name__ == "__main__":
    main()
