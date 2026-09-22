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
TITLE = "CAMERA: onboard"


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
    createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
    putImageData() { ops.push({ op: "putImageData" }); },
    drawImage(src, dx, dy, dw, dh) { ops.push({ op: "drawImage", dx, dy, dw, dh }); },
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

const specEarly = JSON.parse(process.argv[3]);
let search = specEarly.vision === "1" ? "" : "?vision=0";
if (specEarly.lap === "1") search += (search ? "&" : "?") + "lap=1";
const location = { search };
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
const visionPosts = [];
const visionIntervals = [];
window.fetch = function () {
  return Promise.resolve({ ok: false, json: async () => ({}) });
};
if (specEarly.cmd === "upload" || specEarly.cmd === "vision-ack" || specEarly.cmd === "vision-order") {
  global.setInterval = (fn, ms) => {
    visionIntervals.push(ms);
    return visionIntervals.length;
  };
  global.setTimeout = () => 1;
}
if (specEarly.cmd === "vision-ack") {
  window.SEMIF_TELEMETRY_CORE = {
    createLatencyTelemetry() {
      const api = {
        records: [],
        recordVision(row) { api.records.push(row); },
        hudText() { return "hud"; },
      };
      return api;
    },
  };
  const replies = specEarly.replies || [{}];
  let replyAt = 0;
  window.fetch = function (url) {
    const href = typeof url === "string" ? url : "";
    const body = replies[Math.min(replyAt, replies.length - 1)];
    if (href.indexOf("/v1/vision") !== -1) replyAt += 1;
    return Promise.resolve({ ok: true, json: async () => body });
  };
}
if (specEarly.cmd === "vision-order") {
  const pending = [];
  window.__releaseVision = (index, body) => {
    const resolve = pending[index];
    if (resolve) resolve({ ok: true, json: async () => body });
  };
  window.fetch = function (url) {
    const href = typeof url === "string" ? url : "";
    if (href.indexOf("/v1/vision") === -1) {
      return Promise.resolve({ ok: false, json: async () => ({}) });
    }
    return new Promise((resolve) => { pending.push(resolve); });
  };
}
if (specEarly.cmd === "upload") {
  window.fetch = function (url) {
    const href = typeof url === "string" ? url : "";
    if (href.indexOf("/v1/vision") !== -1) visionPosts.push(nowMs);
    return Promise.resolve({
      ok: true,
      json: async () => ({ vision: { signal: "green" }, vision_encode_ms: 3 }),
    });
  };
}
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

const spec = specEarly;
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
  }));
} else if (spec.cmd === "ribbon") {
  const pts = api.ribbonPoints(spec.player, spec.maneuver);
  process.stdout.write(JSON.stringify(pts));
} else if (spec.cmd === "grab") {
  const worldCanvas = el("canvas");
  worldCanvas.toDataURL = () => "data:image/jpeg;base64,PLAYER";
  canvas.toDataURL = () => "data:image/jpeg;base64,ONBOARD";
  const shots = [];
  function makeCam() {
    const cam = {
      fov: 52,
      aspect: 1,
      position: { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
      lookAt(x, y, z) { this.look = { x, y, z }; },
      updateProjectionMatrix() {},
      updateMatrixWorld() {},
    };
    cam.clone = () => makeCam();
    return cam;
  }
  function RT(w, h) {
    this.w = w;
    this.h = h;
    this.isWebGLRenderTarget = true;
    this.texture = { colorSpace: "srgb-linear" };
  }
  const world = {
    mode: "map",
    canvas: worldCanvas,
    scene: {},
    player: { traverse() {} },
    sim: { player: { x: 10, z: -4, heading: Math.PI / 2 } },
    camera: makeCam(),
    sun: { shadow: { map: { constructor: RT } } },
    renderer: {
      getRenderTarget() { return null; },
      outputColorSpace: "srgb",
      setRenderTarget(target) {
        shots.push({
          op: "target",
          w: target && target.w,
          h: target && target.h,
          xr: target ? target.isXRRenderTarget === true : false,
          colorSpace: target && target.texture && target.texture.colorSpace,
          internalFormat: target && target.texture ? target.texture.internalFormat || null : null,
        });
      },
      render(_scene, cam) {
        shots.push({ op: "render", x: cam.position.x, y: cam.position.y, z: cam.position.z, fov: cam.fov, look: cam.look });
      },
      readRenderTargetPixels(_t, _x, _y, w, h, buf) { buf.fill(8); },
    },
  };
  window.SEMIF_WORLD = world;
  canvas.__ctx.ops.length = 0;
  nowMs = 0;
  const url = window.SEMIF_GRAB_FRAME();
  const first = shots.filter((s) => s.op === "render").pop();
  world.mode = "chase";
  window.SEMIF_GRAB_FRAME();
  const second = shots.filter((s) => s.op === "render").pop();
  const beforeView = shots.filter((s) => s.op === "render").length;
  nowMs = 0;
  window.__raf();
  nowMs = 500;
  window.__raf();
  nowMs = 1000;
  window.__raf();
  const viewShots = shots.filter((s) => s.op === "render").length - beforeView;
  const target = shots.filter((s) => s.op === "target" && s.w)[0];
  process.stdout.write(JSON.stringify({
    url,
    mode: world.mode,
    first,
    second,
    target,
    viewShots,
    ops: plainOps(canvas.__ctx.ops),
    fps: fps.textContent,
  }));
} else if (spec.cmd === "upload") {
  canvas.toDataURL = () => "data:image/jpeg;base64,ONBOARD";
  function makeCam() {
    const cam = {
      fov: 52,
      aspect: 1,
      position: { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
      lookAt(x, y, z) { this.look = { x, y, z }; },
      updateProjectionMatrix() {},
      updateMatrixWorld() {},
    };
    cam.clone = () => makeCam();
    return cam;
  }
  function RT(w, h) {
    this.w = w;
    this.h = h;
    this.isWebGLRenderTarget = true;
    this.texture = {};
  }
  window.SEMIF_WORLD = {
    mode: "chase",
    scene: {},
    player: { traverse() {} },
    sim: { player: { x: 10, z: -4, heading: Math.PI / 2 } },
    camera: makeCam(),
    sun: { shadow: { map: { constructor: RT } } },
    renderer: {
      getRenderTarget() { return null; },
      outputColorSpace: "srgb",
      setRenderTarget() {},
      render() {},
      readRenderTargetPixels(_t, _x, _y, w, h, buf) { buf.fill(8); },
    },
  };
  const frames = spec.frames || 120;
  const dt = 1000 / 60;
  const ticks = [];
  let renders = 0;
  const renderer = window.SEMIF_WORLD.renderer;
  const paint = renderer.render;
  renderer.render = function () {
    renders += 1;
    return paint.apply(this, arguments);
  };
  nowMs = 0;
  for (let i = 0; i < frames; i++) {
    nowMs += dt;
    ticks.push(nowMs);
    window.__raf();
  }
  process.stdout.write(JSON.stringify({
    posts: visionPosts.length,
    postTimes: visionPosts,
    tickTimes: ticks,
    intervals: visionIntervals,
    fps: fps.textContent,
    renders,
  }));
} else if (spec.cmd === "vision-order") {
  canvas.toDataURL = () => "data:image/jpeg;base64,ONBOARD";
  function makeCam() {
    const cam = {
      fov: 52,
      aspect: 1,
      position: { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
      lookAt(x, y, z) { this.look = { x, y, z }; },
      updateProjectionMatrix() {},
      updateMatrixWorld() {},
    };
    cam.clone = () => makeCam();
    return cam;
  }
  function RT(w, h) {
    this.w = w;
    this.h = h;
    this.isWebGLRenderTarget = true;
    this.texture = {};
  }
  window.SEMIF_WORLD = {
    scene: {},
    player: { traverse() {} },
    sim: { player: { x: 10, z: -4, heading: 0 } },
    camera: makeCam(),
    sun: { shadow: { map: { constructor: RT } } },
    renderer: {
      getRenderTarget() { return null; },
      outputColorSpace: "srgb",
      setRenderTarget() {},
      render() {},
      readRenderTargetPixels(_t, _x, _y, w, h, buf) { buf.fill(8); },
    },
  };
  nowMs = 16;
  window.__raf();
  nowMs = 32;
  window.__raf();
  window.__releaseVision(1, {
    vision: { signal: "red", event: "newer frame" },
    vision_gen: 2,
    vision_encode_ms: 5,
  });
  setImmediate(() => {
    window.__releaseVision(0, {
      vision: { signal: "green", event: "older frame" },
      vision_gen: 1,
      vision_encode_ms: 9,
    });
    setImmediate(() => {
      process.stdout.write(JSON.stringify({
        signal: window.SEMIF_VISION && window.SEMIF_VISION.signal,
        gen: window.SEMIF_VISION_GEN,
      }));
    });
  });
} else if (spec.cmd === "vision-ack") {
  canvas.toDataURL = () => "data:image/jpeg;base64,ONBOARD";
  function makeCam() {
    const cam = {
      fov: 52,
      aspect: 1,
      position: { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
      lookAt(x, y, z) { this.look = { x, y, z }; },
      updateProjectionMatrix() {},
      updateMatrixWorld() {},
    };
    cam.clone = () => makeCam();
    return cam;
  }
  function RT(w, h) {
    this.w = w;
    this.h = h;
    this.isWebGLRenderTarget = true;
    this.texture = {};
  }
  window.SEMIF_WORLD = {
    scene: {},
    player: { traverse() {} },
    sim: { player: { x: 10, z: -4, heading: 0 } },
    camera: makeCam(),
    sun: { shadow: { map: { constructor: RT } } },
    renderer: {
      getRenderTarget() { return null; },
      outputColorSpace: "srgb",
      setRenderTarget() {},
      render() {},
      readRenderTargetPixels(_t, _x, _y, w, h, buf) { buf.fill(8); },
    },
  };
  const replies = spec.replies || [{}];
  nowMs = 0;
  for (let i = 0; i < replies.length; i++) {
    nowMs += 16;
    window.__raf();
  }
  setImmediate(() => {
    process.stdout.write(JSON.stringify({
      vision: window.SEMIF_VISION,
      text: document.getElementById("fsd-vision").textContent,
      records: (window.SEMIF_TELEMETRY && window.SEMIF_TELEMETRY.records) || [],
    }));
  });
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
} else if (spec.cmd === "lap") {
  const player = {
    x: 10,
    z: 0.5,
    speed: 0.2,
    target: 4,
    route: { points: [{ x: 0, z: 0 }, { x: 10, z: 0 }] },
  };
  const sim = {
    complete: false,
    freeExplore: false,
    autopilot: true,
    chained: false,
    player,
    world: { seed: 42 },
    step() {
      const last = player.route.points[player.route.points.length - 1];
      const dist = Math.hypot(player.x - last.x, player.z - last.z);
      if (!sim.complete && dist < 3 && Math.abs(player.speed) < 1) {
        player.route = { points: [{ x: 0, z: 80 }] };
        sim.complete = false;
        sim.chained = true;
      }
    },
  };
  window.SEMIF_SIM = sim;
  window.SEMIF_WORLD = {};
  window.__raf();
  sim.step(0.016);
  const end = player.route.points[player.route.points.length - 1];
  process.stdout.write(JSON.stringify({
    complete: sim.complete,
    chained: sim.chained,
    target: player.target,
    endZ: end.z,
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


def test_lap_query_stops_at_the_route_end():
    """?lap=1 keeps the finished route and sets complete after the sim chains."""
    lap = _run({"cmd": "lap", "vision": "1", "lap": "1"})
    assert lap["complete"] is True
    assert lap["chained"] is True
    assert lap["endZ"] == 0
    assert lap["target"] == 0


def test_pip_shell_is_in_the_loaded_overlay():
    dom = _run({"cmd": "dom"})
    assert dom["hasPip"] is True
    assert dom["title"] == TITLE
    assert dom["canvas"] == {"w": PIP_W, "h": PIP_H}
    assert dom["fps0"] == "-- FPS"
    css = OVERLAY_CSS.read_text(encoding="utf-8")
    html = (REPO / "demo" / "jevpilot" / "index.html").read_text(encoding="utf-8")
    assert "preserveDrawingBuffer" in html.split('type="module"')[0]
    assert "#fsd-pip" in css
    assert "pointer-events: auto" in css
    assert "fsd-pip-collapsed" in css
    assert "fsd-pip-hidden" in css


def test_each_display_frame_posts_onboard_jpeg():
    """Live tick posts one /v1/vision JPEG per painted onboard frame."""
    pumped = _run({"cmd": "upload", "vision": "1", "frames": 120})
    assert pumped["posts"] == 120
    assert pumped["renders"] == 120
    assert pumped["postTimes"] == pumped["tickTimes"]
    assert 700 not in pumped["intervals"]
    assert pumped["fps"] == "60 FPS"


def test_late_older_vision_does_not_replace_newer_evidence():
    """An older inference that arrives last leaves the newer evidence in place."""
    ordered = _run({"cmd": "vision-order", "vision": "1"})
    assert ordered["signal"] == "red"
    assert ordered["gen"] == 2


def test_empty_vision_ack_keeps_prior_evidence():
    """A reply without vision does not replace evidence or record encode time."""
    ack = _run({
        "cmd": "vision-ack",
        "vision": "1",
        "replies": [
            {
                "vision": {"signal": "green", "event": "road clear ahead"},
                "vision_encode_ms": 4,
            },
            {},
        ],
    })
    assert ack["vision"]["signal"] == "green"
    assert ack["text"] == "VISION waiting"
    assert ack["records"] == [{"encode_ms": 4, "rtt_ms": ack["records"][0]["rtt_ms"], "grab_ms": ack["records"][0]["grab_ms"]}]
    assert len(ack["records"]) == 1


def test_pip_shows_the_fixed_onboard_camera_not_the_player_view():
    """Change camera is the player view. The posted frame stays on the car."""
    grabbed = _run({"cmd": "grab"})
    assert grabbed["url"] == "data:image/jpeg;base64,ONBOARD"
    assert grabbed["mode"] == "chase"
    shot = grabbed["first"]
    assert shot["fov"] == 60
    assert shot["x"] == pytest.approx(10.15)
    assert shot["y"] == pytest.approx(1.45)
    assert shot["z"] == pytest.approx(-4)
    assert shot["look"]["x"] == pytest.approx(10.15 + 25)
    assert shot["look"]["z"] == pytest.approx(-4)
    again = grabbed["second"]
    assert again["x"] == pytest.approx(shot["x"])
    assert again["z"] == pytest.approx(shot["z"])
    assert any(op["op"] == "putImageData" for op in grabbed["ops"])
    target = grabbed["target"]
    assert target["xr"] is True
    assert target["colorSpace"] == "srgb"
    assert target["internalFormat"] == "RGBA8"
    assert grabbed["viewShots"] == 3
    assert grabbed["fps"] == "2 FPS"
    assert any(op["op"] == "putImageData" for op in grabbed["ops"])
    assert not any(op["op"] == "drawImage" for op in grabbed["ops"])
    js = OVERLAY_JS.read_text(encoding="utf-8")
    vision = js.split("async function visionTick")[1].split("if (visionOn)")[0]
    assert "grabFrame()" in vision


def test_v_and_header_toggle_without_stealing_seed_input():
    keys = _run({"cmd": "keys"})
    assert keys["beforeHidden"] is False
    assert keys["afterVHidden"] is True
    assert keys["afterRepeat"] is keys["afterVHidden"]
    assert keys["afterClickFold"] is not keys["beforeFold"]
    assert keys["hiddenAfterType"] is keys["hiddenBeforeType"]
