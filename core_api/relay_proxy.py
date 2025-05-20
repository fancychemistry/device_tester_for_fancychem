"""relay_proxy.py – v3
gcode_macro 名称格式: RELAY_ON_<idx> / RELAY_OFF_<idx> / RELAY_TOGGLE_<idx>
"""
import requests
import logging

log = logging.getLogger(__name__)


class MoonrakerError(RuntimeError):
    pass


class RelayProxy:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip('/')

    # internal
    def _send(self, script: str):
        url = f"{self.base}/printer/gcode/script"
        log.debug("POST %s | %s", url, script)
        r = requests.post(url, json={"script": script}, timeout=30)
        if r.status_code != 200:
            raise MoonrakerError(f"{r.status_code}: {r.text}")
        data = r.json()
        # 改用普通字符替代特殊Unicode箭头，避免GBK编码错误
        log.info(">> %s -> %s", script, data)
        return data

    # public
    def on(self, idx: int):
        return self._send(f"RELAY_ON_{idx}")

    def off(self, idx: int):
        return self._send(f"RELAY_OFF_{idx}")

    def toggle(self, idx: int, state: str | None = None):
        if state:
            return self._send(f"RELAY_TOGGLE_{idx} STATE={state.upper()}")
        return self._send(f"RELAY_TOGGLE_{idx}")
