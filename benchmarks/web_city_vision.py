"""Official vision scores from the existing JevPilot web city.

Two headless laps, seed 42: vision on, and vision off. The CUDA report is
written only when both laps reach ``complete`` on a non-mock CUDA server.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPORT = ROOT / "results" / "phase5-jevpilot-vision-cuda.json"
WORLD = "web-city-onboard-camera"

_READ_LAP = """
(() => {
  const sim = window.SEMIF_SIM;
  if (!sim) return {ready: false};
  if (!sim.crash && !sim.complete) sim.autopilot = true;
  return {
    ready: true,
    complete: !!sim.complete,
    red_light: sim.redLightViolations || 0,
    vehicle_collisions: sim.vehicleCollisions || 0,
    pedestrian: sim.pedestrianCasualties || 0,
    crash: sim.crash || null,
    speeding: sim.speedingViolations || 0,
    autopilot: !!sim.autopilot,
    time_s: sim.time || 0,
    seed: sim.world ? sim.world.seed : null
  };
})()
"""


def lap_url(base: str, seed: int, vision_on: bool) -> str:
    root = base.rstrip("/")
    query = f"seed={int(seed)}&lap=1"
    if not vision_on:
        query += "&vision=0"
    return f"{root}/jevpilot/?{query}"


def score_lap(obs: dict, *, seed: int, vision_on: bool) -> dict:
    complete = bool(obs.get("complete"))
    red = int(obs.get("red_light") or 0)
    vehicles = int(obs.get("vehicle_collisions") or 0)
    pedestrians = int(obs.get("pedestrian") or 0)
    crash = obs.get("crash")
    speeding = int(obs.get("speeding") or 0)
    observed = obs.get("seed")
    lap_seed = int(observed) if observed is not None else int(seed)
    clean = (
        complete
        and red == 0
        and vehicles == 0
        and pedestrians == 0
        and not crash
        and lap_seed == int(seed)
    )
    return {
        "seed": lap_seed,
        "vision": "on" if vision_on else "off",
        "complete": complete,
        "red_light": red,
        "vehicle_collisions": vehicles,
        "pedestrian": pedestrians,
        "crash": crash,
        "speeding": speeding,
        "clean": clean,
        "autopilot": bool(obs.get("autopilot")),
        "time_s": obs.get("time_s"),
    }


def write_official_report(path: Path, report: dict) -> bool:
    path = Path(path)
    laps = list(report.get("laps") or [])
    official = (
        report.get("device") == "cuda"
        and report.get("mock") is False
        and report.get("model") not in (None, "", "MockDecisionEngine")
        and report.get("seed") == 42
        and report.get("world") == WORLD
        and [lap.get("vision") for lap in laps] == ["on", "off"]
        and all(lap.get("complete") and lap.get("seed") == 42 for lap in laps)
    )
    if not official:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return True


def _browser_executable() -> str:
    env = os.environ.get("SEMIF_CHROME")
    candidates = [
        env,
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    raise RuntimeError("headless browser not found; set SEMIF_CHROME to Chrome or Edge")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Cdp:
    def __init__(self, sock: socket.socket, leftover: bytes = b""):
        self.sock = sock
        self.buf = leftover
        self.msg_id = 0

    def evaluate(self, expression: str) -> Any:
        self.msg_id += 1
        ident = self.msg_id
        payload = {
            "id": ident,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True, "awaitPromise": True},
        }
        self._send(json.dumps(payload))
        while True:
            message = self._recv()
            if message.get("id") == ident:
                result = message.get("result", {}).get("result", {})
                if "value" in result:
                    return result["value"]
                desc = message.get("result", {}).get("exceptionDetails") or result
                raise RuntimeError(f"page evaluate failed: {desc}")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _send(self, text: str) -> None:
        data = text.encode("utf-8")
        header = bytes([0x81, 0x80 | min(len(data), 125)])
        if len(data) > 125:
            header = bytes([0x81, 0x80 | 126]) + struct.pack("!H", len(data))
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + mask + masked)

    def _recv(self) -> dict:
        while True:
            frame = self._read_frame()
            if frame is None:
                continue
            opcode, payload = frame
            if opcode == 0x9:
                self._send_pong(payload)
                continue
            if opcode == 0x8:
                raise RuntimeError("browser closed the debugger socket")
            if opcode == 0x1:
                return json.loads(payload.decode("utf-8"))

    def _send_pong(self, payload: bytes) -> None:
        header = bytes([0x8A, 0x80 | len(payload)])
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("debugger socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read_frame(self) -> Optional[tuple[int, bytes]]:
        first = self._read_exact(2)
        opcode = first[0] & 0x0F
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        payload = self._read_exact(length)
        if not (first[0] & 0x80):
            extra = self._read_frame()
            if extra is None:
                return opcode, payload
            return opcode, payload + extra[1]
        return opcode, payload


def _cdp_connect(port: int) -> _Cdp:
    deadline = time.monotonic() + 20
    last = "no page"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            last = str(exc)
            time.sleep(0.2)
            continue
        page = next((t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")), None)
        if page:
            ws = page["webSocketDebuggerUrl"]
            host_path = ws.split("://", 1)[1]
            host_port, path = host_path.split("/", 1)
            host, ws_port = host_port.rsplit(":", 1)
            sock = socket.create_connection((host, int(ws_port)), timeout=10)
            key = base64.b64encode(os.urandom(16)).decode()
            request = (
                f"GET /{path} HTTP/1.1\r\n"
                f"Host: {host}:{ws_port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            sock.sendall(request.encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    raise RuntimeError("debugger handshake failed")
                buf += chunk
            head, leftover = buf.split(b"\r\n\r\n", 1)
            if b" 101 " not in head.split(b"\r\n", 1)[0]:
                raise RuntimeError(f"debugger handshake rejected: {head[:200]!r}")
            return _Cdp(sock, leftover)
        time.sleep(0.2)
    raise RuntimeError(f"debugger page did not appear: {last}")


def _launch_browser(url: str) -> tuple[subprocess.Popen, _Cdp]:
    profile = Path(os.environ.get("TEMP", ".")) / f"semif-web-city-{os.getpid()}-{time.time_ns()}"
    profile.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            _browser_executable(),
            "--headless=new",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--window-size=1280,720",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    port_file = profile / "DevToolsActivePort"
    deadline = time.monotonic() + 20
    port = None
    while time.monotonic() < deadline:
        if port_file.is_file():
            port = int(port_file.read_text(encoding="utf-8").splitlines()[0])
            break
        if proc.poll() is not None:
            raise RuntimeError(f"browser exited {proc.returncode} before the debugger port appeared")
        time.sleep(0.1)
    if port is None:
        proc.kill()
        raise RuntimeError("browser did not open a debugger port")
    try:
        return proc, _cdp_connect(port)
    except Exception:
        proc.kill()
        raise


def drive_lap(
    evaluate: Callable[[str], Any],
    *,
    seed: int,
    vision_on: bool,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    deadline = clock() + timeout_s
    last: dict = {}
    while True:
        raw = evaluate(_READ_LAP) or {}
        if raw.get("ready"):
            last = raw
            if raw.get("complete"):
                break
        if clock() >= deadline:
            break
        sleep(0.5)
    return score_lap(last, seed=seed, vision_on=vision_on)


def run_browser_lap(base: str, *, seed: int, vision_on: bool, timeout_s: float) -> dict:
    proc, cdp = _launch_browser(lap_url(base, seed, vision_on))
    try:
        return drive_lap(
            lambda expression: cdp.evaluate(expression),
            seed=seed,
            vision_on=vision_on,
            timeout_s=timeout_s,
        )
    finally:
        cdp.close()
        proc.kill()
        proc.wait(timeout=10)


def _wait_health(base: str, timeout_s: float = 600) -> dict:
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("status") == "online":
                return body
        except OSError as exc:
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"decision server did not become healthy: {last}")


def run_official_web_city(
    *,
    seed: int = 42,
    output: Path = OFFICIAL_REPORT,
    lap_timeout_s: float = 180,
    base_url: Optional[str] = None,
) -> dict:
    """Run the two web-city laps. Write ``output`` only when the official gate passes."""
    proc = None
    if base_url is None:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "demo" / "server.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--device",
                "cuda",
            ],
            cwd=str(ROOT),
        )
    try:
        health = _wait_health(base_url)
        laps = [
            run_browser_lap(base_url, seed=seed, vision_on=True, timeout_s=lap_timeout_s),
            run_browser_lap(base_url, seed=seed, vision_on=False, timeout_s=lap_timeout_s),
        ]
        report = {
            "benchmark": "JevPilot web-city vision",
            "seed": seed,
            "mock": bool(health.get("mock_mode")),
            "model": health.get("model"),
            "device": health.get("device"),
            "world": WORLD,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "laps": laps,
        }
        written = write_official_report(output, report)
        report["official_report"] = str(output) if written else None
        return report
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=20)
