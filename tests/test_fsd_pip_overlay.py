"""FSD picture-in-picture frame view (#115).

The node harness loads demo/jevpilot/semif-layer.js and calls the animation
callback that script registers. That is the painter the page runs each frame.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OVERLAY_JS = REPO / "demo" / "jevpilot" / "semif-layer.js"
OVERLAY_CSS = REPO / "demo" / "jevpilot" / "semif-layer.css"

PIP_W = 320
PIP_H = 180
TITLE = "CAMERA: Front Main (60° FOV)"
RIBBON = "rgba(34, 211, 238, 0.45)"
PED_STROKE = "#22c55e"
CUT_STROKE = "#f5c518"
LIGHT_RED = "#ef4444"


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> list[float]:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = [0.0] * 16
    m[0] = f / aspect
    m[5] = f
    m[10] = (far + near) / (near - far)
    m[11] = -1.0
    m[14] = (2.0 * far * near) / (near - far)
    return m


def _view_at(x: float, y: float, z: float) -> list[float]:
    """Camera at (x, y, z), looking down -Z. Column-major matrixWorldInverse."""
    m = [0.0] * 16
    m[0] = m[5] = m[10] = m[15] = 1.0
    m[12] = -x
    m[13] = -y
    m[14] = -z
    return m


_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

function makeCtx(canvas) {
  const ops = [];
  let fillStyle = "";
  let strokeStyle = "";
  const ctx = {
    ops,
    canvas,
    save() { ops.push({ op: "save" }); },
    restore() { ops.push({ op: "restore" }); },
    beginPath() { ops.push({ op: "beginPath" }); },
    closePath() { ops.push({ op: "closePath" }); },
    moveTo(x, y) { ops.push({ op: "moveTo", x, y }); },
    lineTo(x, y) { ops.push({ op: "lineTo", x, y }); },
    arc(x, y, r) { ops.push({ op: "arc", x, y, r, strokeStyle, fillStyle }); },
    fill() { ops.push({ op: "fill", fillStyle }); },
    stroke() { ops.push({ op: "stroke", strokeStyle }); },
    clearRect() { ops.push({ op: "clearRect" }); },
    setTransform() {},
    strokeRect(x, y, w, h) { ops.push({ op: "strokeRect", x, y, w, h, strokeStyle }); },
    fillRect(x, y, w, h) { ops.push({ op: "fillRect", x, y, w, h, fillStyle }); },
    fillText(text, x, y) { ops.push({ op: "fillText", text: String(text), x, y, fillStyle }); },
    drawImage(src, dx, dy, dw, dh) { ops.push({ op: "drawImage", src, dx, dy, dw, dh }); },
    setLineWidth() {},
    measureText(text) { return { width: String(text).length * 6 }; },
  };
  Object.defineProperty(ctx, "fillStyle", {
    get() { return fillStyle; },
    set(v) { fillStyle = String(v); },
  });
  Object.defineProperty(ctx, "strokeStyle", {
    get() { return strokeStyle; },
    set(v) { strokeStyle = String(v); },
  });
  Object.defineProperty(ctx, "font", { get() { return ""; }, set() {} });
  Object.defineProperty(ctx, "textAlign", { get() { return "left"; }, set() {} });
  Object.defineProperty(ctx, "textBaseline", { get() { return "alphabetic"; }, set() {} });
  Object.defineProperty(ctx, "lineWidth", { get() { return 1; }, set() {} });
  canvas.getContext = () => ctx;
  canvas.__ctx = ctx;
  return ctx;
}

function el(tag) {
  const node = {
    tagName: String(tag || "div").toUpperCase(),
    id: "",
    className: "",
    children: [],
    parentElement: null,
    style: {},
    listeners: {},
    textContent: "",
    value: "",
    width: 300,
    height: 150,
    clientWidth: 640,
    clientHeight: 480,
    hidden: false,
  };
  const classes = new Set();
  node.classList = {
    add(...names) { names.forEach((n) => classes.add(n)); node.className = [...classes].join(" "); },
    remove(...names) { names.forEach((n) => classes.delete(n)); node.className = [...classes].join(" "); },
    contains(n) { return classes.has(n); },
    toggle(n) { classes.has(n) ? classes.delete(n) : classes.add(n); node.className = [...classes].join(" "); return classes.has(n); },
  };
  node.appendChild = (child) => {
    node.children.push(child);
    child.parentElement = node;
    return child;
  };
  node.addEventListener = (type, fn) => {
    (node.listeners[type] || (node.listeners[type] = [])).push(fn);
  };
  node.dispatchEvent = (ev) => {
    const e = ev || {};
    e.target = e.target || node;
    e.preventDefault = e.preventDefault || function () {};
    (node.listeners[e.type] || []).forEach((fn) => fn(e));
    if (node.parentElement && node.parentElement.dispatchEvent && e.bubbles) {
      node.parentElement.dispatchEvent(e);
    }
  };
  node.contains = (other) => {
    if (other === node) return true;
    return node.children.some((child) => child.contains && child.contains(other));
  };
  node.click = () => node.dispatchEvent({ type: "click", target: node, bubbles: true });
  node.toDataURL = () => "data:image/jpeg;base64,AAAA";
  Object.defineProperty(node, "innerHTML", {
    get() { return node._html || ""; },
    set(html) {
      node._html = String(html);
      node.children = [];
      parseInto(node, node._html);
    },
  });
  makeCtx(node);
  return node;
}

function parseInto(parent, html) {
  const tagRe = /<\/?([a-zA-Z0-9]+)([^>]*)>/g;
  const stack = [parent];
  let match;
  while ((match = tagRe.exec(html))) {
    const raw = match[0];
    const closing = raw[1] === "/";
    const tag = match[1].toLowerCase();
    const attrs = match[2] || "";
    const self = /\/$/.test(attrs) || raw.endsWith("/>");
    if (closing) {
      const node = stack[stack.length - 1];
      if (stack.length > 1 && node.tagName === tag.toUpperCase()) {
        if (!node.children.length) {
          node.textContent = html.slice(node._htmlStart, match.index).replace(/<[^>]+>/g, "").trim();
        }
        stack.pop();
      }
      continue;
    }
    const node = el(tag);
    const id = /id="([^"]*)"/.exec(attrs);
    const cls = /class="([^"]*)"/.exec(attrs);
    const w = /width="(\d+)"/.exec(attrs);
    const h = /height="(\d+)"/.exec(attrs);
    if (id) node.id = id[1];
    if (cls) cls[1].split(/\s+/).filter(Boolean).forEach((name) => node.classList.add(name));
    if (w) node.width = Number(w[1]);
    if (h) node.height = Number(h[1]);
    node._htmlStart = tagRe.lastIndex;
    stack[stack.length - 1].appendChild(node);
    if (!self) stack.push(node);
  }
}

function walk(node, visit) {
  visit(node);
  (node.children || []).forEach((child) => walk(child, visit));
}

const document = el("document");
document.documentElement = el("html");
document.body = el("body");
document.appendChild(document.documentElement);
document.documentElement.appendChild(document.body);
document.createElement = (tag) => el(tag);
document.getElementById = (id) => {
  let found = null;
  walk(document, (node) => {
    if (!found && node.id === id) found = node;
  });
  return found;
};
document.querySelector = (sel) => {
  let found = null;
  walk(document, (node) => {
    if (found || !node.tagName) return;
    if (sel[0] === "#" && node.id === sel.slice(1)) found = node;
    else if (sel[0] === "." && node.classList && node.classList.contains(sel.slice(1))) found = node;
  });
  return found;
};

const location = { search: "?vision=0" };
let nowMs = 0;
const window = global;
window.window = window;
window.document = document;
window.location = location;
window.performance = { now: () => nowMs };
window.requestAnimationFrame = (fn) => {
  window.__raf = fn;
  return 1;
};
window.cancelAnimationFrame = () => {};
window.fetch = function () {
  return Promise.resolve({ ok: false, json: async () => ({}) });
};
window.URL = { createObjectURL: () => "blob:pip", revokeObjectURL() {} };
global.document = document;
global.location = location;
global.performance = window.performance;
global.requestAnimationFrame = window.requestAnimationFrame;
global.cancelAnimationFrame = window.cancelAnimationFrame;

const code = fs.readFileSync(process.argv[1], "utf8");
vm.runInThisContext(code, { filename: process.argv[1] });

const pip = document.getElementById("fsd-pip");
const canvas = document.getElementById("fsd-camera-canvas");
const title = document.querySelector(".fsd-pip-title");
const header = document.querySelector(".fsd-pip-header");
const fps = document.getElementById("fsd-pip-fps");
const api = window.SEMIF_PIP || {};

function camera(w, h) {
  return {
    matrixWorldInverse: { elements: JSON.parse(process.argv[2]) },
    projectionMatrix: { elements: _perspectiveElements(w, h) },
  };
}

function plainOps(ops) {
  return ops.map((op) => {
    const copy = {};
    for (const key of Object.keys(op)) copy[key] = key === "src" ? true : op[key];
    return copy;
  });
}

function boot(sim, world) {
  window.SEMIF_SIM = sim;
  window.SEMIF_WORLD = world;
  canvas.__ctx.ops.length = 0;
  window.__raf();
  return plainOps(canvas.__ctx.ops);
}

const spec = JSON.parse(process.argv[3]);
const view = JSON.parse(process.argv[2]);
const out = { title: title && title.textContent, fps0: fps && fps.textContent, canvas: canvas && { w: canvas.width, h: canvas.height }, hasPip: !!pip };

if (spec.cmd === "dom") {
  process.stdout.write(JSON.stringify(out));
} else if (spec.cmd === "project") {
  const pt = api.project(camera(spec.w, spec.h), spec.x, spec.y, spec.z, spec.w, spec.h);
  const back = api.unprojectGround(camera(spec.w, spec.h), pt.x, pt.y, spec.w, spec.h);
  process.stdout.write(JSON.stringify({ pt, back }));
} else if (spec.cmd === "paint") {
  const srcW = spec.srcW;
  const srcH = spec.srcH;
  const worldCanvas = el("canvas");
  worldCanvas.width = srcW;
  worldCanvas.height = srcH;
  worldCanvas.clientWidth = srcW;
  worldCanvas.clientHeight = srcH;
  const world = { canvas: worldCanvas, camera: camera(srcW, srcH) };
  const player = { x: 0, z: 0, heading: 0, speed: spec.speed };
  const sim = {
    player,
    pedestrians: spec.ped ? [spec.ped] : [],
    traffic: [],
    _fsdAgents: spec.agents || [],
    step() {},
    lastDecisionState: spec.decision || null,
    decisionState() {
      throw new Error("pip must not call decisionState");
    },
  };
  sim.player.maneuver = spec.maneuver;
  nowMs = 0;
  const first = boot(sim, world);
  sim.player.maneuver = spec.maneuver2 || spec.maneuver;
  nowMs = 500;
  canvas.__ctx.ops.length = 0;
  window.__raf();
  const second = plainOps(canvas.__ctx.ops);
  nowMs = 1000;
  canvas.__ctx.ops.length = 0;
  window.__raf();
  process.stdout.write(JSON.stringify({
    first, second,
    fps: fps.textContent,
    fit: api.fitContain(srcW, srcH, canvas.width, canvas.height),
    sameRaf: window.__raf.length >= 0,
  }));
} else if (spec.cmd === "ribbon") {
  const pts = api.ribbonPoints(spec.player, spec.maneuver);
  process.stdout.write(JSON.stringify(pts));
} else if (spec.cmd === "keys") {
  const beforeHidden = pip.classList.contains("fsd-pip-hidden");
  const beforeFold = pip.classList.contains("fsd-pip-collapsed");
  document.dispatchEvent({ type: "keydown", key: "v", target: document.body, bubbles: true });
  const afterVHidden = pip.classList.contains("fsd-pip-hidden");
  document.dispatchEvent({ type: "keydown", key: "v", target: document.body, repeat: true, bubbles: true });
  const afterRepeat = pip.classList.contains("fsd-pip-hidden");
  header.dispatchEvent({ type: "click", target: header, bubbles: true });
  const afterClickFold = pip.classList.contains("fsd-pip-collapsed");
  const seed = document.getElementById("fsd-seed-input");
  const hiddenBeforeType = pip.classList.contains("fsd-pip-hidden");
  seed.dispatchEvent({ type: "keydown", key: "v", target: seed, bubbles: true });
  const hiddenAfterType = pip.classList.contains("fsd-pip-hidden");
  process.stdout.write(JSON.stringify({
    beforeHidden, beforeFold, afterVHidden, afterRepeat, afterClickFold, hiddenBeforeType, hiddenAfterType,
  }));
} else {
  throw new Error("unknown cmd");
}

function _perspectiveElements(w, h) {
  const fov = 60 * Math.PI / 180;
  const aspect = w / h;
  const near = 0.1;
  const far = 400;
  const f = 1 / Math.tan(fov / 2);
  const m = Array(16).fill(0);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}
"""


def _run(cmd: dict, view: list[float] | None = None) -> dict:
    view = view if view is not None else _view_at(0.0, 1.4, 8.0)
    proc = subprocess.run(
        ["node", "-e", _HARNESS, str(OVERLAY_JS), json.dumps(view), json.dumps(cmd)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-2000:] or proc.stdout[-2000:] or "node failed")
    return json.loads(proc.stdout)


def test_pip_shell_is_in_the_loaded_overlay():
    dom = _run({"cmd": "dom"})
    assert dom["hasPip"] is True
    assert dom["title"] == TITLE
    assert dom["canvas"] == {"w": PIP_W, "h": PIP_H}
    assert dom["fps0"] == "-- FPS"
    css = OVERLAY_CSS.read_text(encoding="utf-8")
    assert "#fsd-pip" in css
    assert "pointer-events: auto" in css
    assert "fsd-pip-collapsed" in css
    assert "fsd-pip-hidden" in css


def test_ground_contact_unprojects_through_the_same_camera():
    back = _run({"cmd": "project", "w": PIP_W, "h": PIP_H, "x": 1.5, "y": 0.05, "z": -12.0})
    assert back["pt"]["x"] > PIP_W / 2
    assert abs(back["back"]["x"] - 1.5) < 0.05
    assert abs(back["back"]["z"] + 12.0) < 0.05


def test_frame_paints_the_grab_canvas_and_ped_label():
    ped = {"type": "pedestrian", "x": 0.0, "z": -12.0, "height": 1.7}
    painted = _run({
        "cmd": "paint",
        "srcW": PIP_W,
        "srcH": PIP_H,
        "speed": 10,
        "ped": ped,
        "maneuver": {"lane_offset_m": 0.0, "velocity_mps": 10.0, "lookahead_m": 8.0},
        "agents": [],
        "decision": {},
    })
    ops = painted["first"]
    images = [op for op in ops if op["op"] == "drawImage"]
    assert images and images[0]["dw"] == PIP_W and images[0]["dh"] == PIP_H
    texts = [op["text"] for op in ops if op["op"] == "fillText"]
    assert any(t.startswith("PED 12.0m") for t in texts)
    assert any("(0.0, 12.0)" in t for t in texts)
    boxes = [op for op in ops if op["op"] == "strokeRect"]
    assert any(op["strokeStyle"] == PED_STROKE for op in boxes)
    assert any(op["op"] == "stroke" and op["strokeStyle"] == PED_STROKE for op in ops)
    assert painted["fps"] == "2 FPS"


def test_cut_in_and_signal_use_distinct_marks():
    painted = _run({
        "cmd": "paint",
        "srcW": PIP_W,
        "srcH": PIP_H,
        "speed": 8,
        "ped": None,
        "maneuver": {"lane_offset_m": 0.0, "velocity_mps": 8.0, "lookahead_m": 6.0},
        "agents": [{
            "type": "car",
            "_fsd": "cutin",
            "x": 2.0,
            "z": -16.0,
            "height": 1.5,
            "_cut": 1.8,
        }],
        "decision": {
            "scene": {
                "intersection": {
                    "signal": "red",
                    "visible": True,
                    "stop_line_position": {"x": 0.0, "z": -20.0},
                }
            }
        },
    })
    ops = painted["first"]
    texts = [op["text"] for op in ops if op["op"] == "fillText"]
    assert any("CUT-IN" in t and "tau 1.8s" in t for t in texts)
    assert any(op["op"] == "strokeRect" and op["strokeStyle"] == CUT_STROKE for op in ops)
    assert any(op["op"] == "arc" and LIGHT_RED in op["strokeStyle"] for op in ops)


def test_ribbon_tracks_the_selected_maneuver_on_the_next_frame():
    painted = _run({
        "cmd": "paint",
        "srcW": 640,
        "srcH": 480,
        "speed": 10,
        "ped": None,
        "maneuver": {"lane_offset_m": 0.0, "velocity_mps": 10.0, "lookahead_m": 8.0},
        "maneuver2": {"lane_offset_m": 4.0, "velocity_mps": 10.0, "lookahead_m": 8.0},
        "agents": [],
        "decision": {},
    })
    fit = painted["fit"]
    assert fit["w"] < PIP_W or fit["h"] == PIP_H
    assert fit["x"] > 0

    def far_x(ops):
        fills = [i for i, op in enumerate(ops) if op["op"] == "fill" and op["fillStyle"] == RIBBON]
        assert fills, ops
        pts = [op for op in ops[: fills[0]] if op["op"] == "lineTo"]
        assert len(pts) >= 4
        return min(pts, key=lambda op: op["y"])["x"]

    assert abs(far_x(painted["second"]) - far_x(painted["first"])) > 5


def test_ribbon_follows_the_route_instead_of_a_straight_heading():
    """A 90 degree route must not keep the 3s ribbon on the initial heading."""
    route = []
    for i in range(11):
        s = i * 2.0
        if s <= 10:
            route.append({"x": 0.0, "z": -s, "s": s})
        else:
            route.append({"x": s - 10.0, "z": -10.0, "s": s})
    pts = _run({
        "cmd": "ribbon",
        "player": {
            "x": 0.0,
            "z": 0.0,
            "s": 0.0,
            "heading": 0.0,
            "speed": 10,
            "route": {"points": route},
        },
        "maneuver": {"lane_offset_m": 0.0, "velocity_mps": 10.0, "lookahead_m": 8.0},
    })
    far = pts[-1]
    assert far["x"] > 8
    assert abs(far["z"] + 10) < 1.5
    assert abs(far["heading"] - (math.pi / 2)) < 0.2


def test_v_and_header_toggle_without_stealing_seed_input():
    keys = _run({"cmd": "keys"})
    assert keys["beforeHidden"] is False
    assert keys["afterVHidden"] is True
    assert keys["afterRepeat"] is keys["afterVHidden"]
    assert keys["afterClickFold"] is not keys["beforeFold"]
    assert keys["hiddenAfterType"] is keys["hiddenBeforeType"]
