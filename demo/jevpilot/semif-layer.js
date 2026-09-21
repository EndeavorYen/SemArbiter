(function () {
  const params = new URLSearchParams(location.search);
  const rawParam = params.get("raw") === "1" || params.get("rawMode") === "1";
  window.SEMIF_RAW_MODE = window.SEMIF_RAW_MODE || rawParam;
  const seedParam = params.get("seed");
  if (seedParam && Number.isFinite(Number(seedParam))) {
    window.SEMIF_SEED = Number(seedParam);
  }

  document.documentElement.classList.add("fsd-theme");
  document.body.classList.add("fsd-theme");

  const chrome = document.createElement("div");
  chrome.id = "fsd-chrome";
  chrome.innerHTML = `
    <div id="fsd-halo" aria-hidden="true"></div>
    <div id="fsd-boxes"></div>
    <div id="fsd-pip">
      <div class="fsd-pip-header">
        <span class="fsd-pip-title">CAMERA: onboard</span>
        <span id="fsd-pip-fps" class="fsd-pip-fps">-- FPS</span>
      </div>
      <div class="fsd-pip-body">
        <canvas id="fsd-camera-canvas" width="320" height="180"></canvas>
      </div>
    </div>
    <div id="fsd-status" aria-live="polite">
      <label id="fsd-seed-box">seed
        <input id="fsd-seed-input" type="number" min="0" max="999999" step="1" />
      </label>
      <button type="button" id="fsd-seed-apply">Apply</button>
      <button type="button" id="fsd-seed-rand">Random</button>
      <span id="fsd-intent">SemArbiter</span>
      <span id="fsd-vision">VISION off</span>
      <span id="fsd-latency">e2e —  P50 —  P95 —</span>
      <button type="button" id="fsd-latency-export">Export latency</button>
    </div>
  `;
  document.body.appendChild(chrome);

  const halo = document.getElementById("fsd-halo");
  const boxes = document.getElementById("fsd-boxes");
  const seedInput = document.getElementById("fsd-seed-input");
  const visionEl = document.getElementById("fsd-vision");
  const intentEl = document.getElementById("fsd-intent");
  const latencyEl = document.getElementById("fsd-latency");
  const visionOn = params.get("vision") !== "0";
  window.SEMIF_VISION = null;

  const telemetryCore = window.SEMIF_TELEMETRY_CORE;
  const telemetry = telemetryCore && telemetryCore.createLatencyTelemetry
    ? telemetryCore.createLatencyTelemetry()
    : null;
  window.SEMIF_TELEMETRY = telemetry;

  function refreshLatencyHud() {
    if (latencyEl && telemetry && typeof telemetry.hudText === "function") {
      latencyEl.textContent = telemetry.hudText();
    }
  }

  document.getElementById("fsd-latency-export").addEventListener("click", () => {
    if (!telemetry || typeof telemetry.exportJSON !== "function") return;
    const blob = new Blob([JSON.stringify(telemetry.exportJSON(), null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "semif-web-latency.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  function reloadWithSeed(seed) {
    const q = new URLSearchParams(location.search);
    q.set("seed", String(Math.max(0, Math.floor(Number(seed) || 0) % 1000000)));
    const world = document.getElementById("world-select");
    if (world && world.value) q.set("world", world.value);
    location.search = q.toString();
  }

  document.getElementById("fsd-seed-apply").addEventListener("click", () => {
    reloadWithSeed(seedInput.value);
  });
  document.getElementById("fsd-seed-rand").addEventListener("click", () => {
    reloadWithSeed(Math.floor(Math.random() * 999999));
  });
  seedInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") reloadWithSeed(seedInput.value);
  });

  const pipRoot = document.getElementById("fsd-pip");
  const pipHeader = document.querySelector(".fsd-pip-header");
  const pipFps = document.getElementById("fsd-pip-fps");
  const pipCanvas = document.getElementById("fsd-camera-canvas");
  const PIP_W = 320;
  const PIP_H = 180;
  if (pipCanvas) {
    pipCanvas.width = PIP_W;
    pipCanvas.height = PIP_H;
  }
  if (pipHeader && pipRoot) {
    pipHeader.addEventListener("click", () => {
      pipRoot.classList.toggle("fsd-pip-collapsed");
    });
  }
  document.addEventListener("keydown", (ev) => {
    if (!pipRoot) return;
    if (ev.repeat) return;
    if (ev.key !== "v" && ev.key !== "V") return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    pipRoot.classList.toggle("fsd-pip-hidden");
  });

  const STALL_SPEED = 0.2;
  const STALL_PROGRESS_M = 0.5;
  const STALL_HOLD_S = 3.0;
  const STALL_COOLDOWN_S = 8.0;

  function requestEgoReplan(sim, dt) {
    const p = sim && sim.player;
    if (!p) return;
    if (!sim.autopilot) {
      sim._stallHeld = 0;
      sim._stallS0 = p.s;
      return;
    }
    sim._stallCool = Math.max(0, (sim._stallCool || 0) - (dt || 0));
    const speed = Math.abs(Number(p.speed) || 0);
    if (speed >= STALL_SPEED) {
      sim._stallHeld = 0;
      sim._stallS0 = p.s;
      return;
    }
    if (sim._stallS0 == null) sim._stallS0 = p.s;
    sim._stallHeld = (sim._stallHeld || 0) + (dt || 0);
    const progress = Number.isFinite(p.s) && Number.isFinite(sim._stallS0)
      ? Math.abs(p.s - sim._stallS0)
      : 0;
    if (sim._stallHeld < STALL_HOLD_S || progress >= STALL_PROGRESS_M || sim._stallCool > 0) return;
    sim._stallHeld = 0;
    sim._stallS0 = p.s;
    sim._stallCool = STALL_COOLDOWN_S;
    sim.offRouteSince = sim.time;
    sim.nextRouteCheck = sim.time;
    if (typeof sim.rerouteIfNeeded === "function") sim.rerouteIfNeeded();
  }

  const LANE_KEEP_OFFSET_M = 1.4;
  const LOOKAHEAD_MIN_M = 4.0;
  const LOOKAHEAD_MAX_M = 8.0;
  const LOOKAHEAD_S = 0.40;

  window.SEMIF_APPLY_STEER = function (_player, u) {
    return u;
  };

  function laneKeepManeuver(selectedOffset, speed) {
    if (
      selectedOffset != null &&
      Number.isFinite(selectedOffset) &&
      Math.abs(selectedOffset) > LANE_KEEP_OFFSET_M
    ) {
      return { lane_offset_m: selectedOffset, lookahead_m: null };
    }
    const look = Math.max(
      LOOKAHEAD_MIN_M,
      Math.min(LOOKAHEAD_MAX_M, LOOKAHEAD_S * Math.abs(Number(speed) || 0))
    );
    return { lane_offset_m: 0, lookahead_m: look };
  }

  function applyLaneKeepReference(sim) {
    const p = sim && sim.player;
    if (!p || !sim.autopilot || sim.paused || sim.crash) return;
    const src = p.maneuver;
    if (!src) return;
    if (src._pdCaptured !== true) {
      src._pdCaptured = true;
      src._pdSelOff = Number.isFinite(src.lane_offset_m) ? src.lane_offset_m : null;
    }
    const keep = laneKeepManeuver(src._pdSelOff, p.speed);
    if (keep.lookahead_m == null) {
      src.lane_offset_m = keep.lane_offset_m;
      return;
    }
    src.lane_offset_m = 0;
    const speed = Number(p.speed) || 0;
    if (sim._pdLook == null || Math.abs(speed - (sim._pdLookSpeed || 0)) > 3) {
      sim._pdLook = keep.lookahead_m;
      sim._pdLookSpeed = speed;
    }
    src.lookahead_m = sim._pdLook;
  }

  function applyRawMode(on) {
    window.SEMIF_RAW_MODE = !!on;
    const sim = window.SEMIF_SIM;
    if (!sim) return;
    sim.rawMode = !!on;
    sim.safety = !on;
    if (sim.player) {
      sim.player.rawMode = !!on;
      if (on && sim.player.maneuver) {
        sim.player.maneuver = { ...sim.player.maneuver, stop_at_line: null };
      }
    }
  }

  const origFetch = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    const href = typeof url === "string" ? url : url && url.url;
    if (href && href.indexOf("/classifier") !== -1 && opts && typeof opts.body === "string") {
      try {
        const body = JSON.parse(opts.body);
        body.raw_mode = !!window.SEMIF_RAW_MODE;
        if (window.SEMIF_SIM && window.SEMIF_SIM.world) {
          body.state = body.state || {};
          body.state.seed = window.SEMIF_SIM.world.seed;
          body.state.raw_mode = !!window.SEMIF_RAW_MODE;
        }
        const sim = window.SEMIF_SIM;
        const laneOffset =
          sim &&
          ((sim.lastDecisionState && sim.lastDecisionState.lane && sim.lastDecisionState.lane.offset_m) ??
            (sim.lastPlan && sim.lastPlan.lane && sim.lastPlan.lane.offset_m));
        if (typeof laneOffset === "number" && Number.isFinite(laneOffset)) {
          body.state = body.state || {};
          body.state.lateral_offset_m = laneOffset;
        }
        if (window.SEMIF_VISION) {
          body.state = body.state || {};
          body.state.vision = window.SEMIF_VISION;
        }
        opts = Object.assign({}, opts, { body: JSON.stringify(body) });
      } catch (_err) {
        /* leave request unchanged */
      }
      const tClass = performance.now();
      return origFetch(url, opts).then(async (res) => {
        let data;
        try {
          data = await res.json();
        } catch (_err) {
          return res;
        }
        try {
          data = fillJevAnswers(data, JSON.parse(opts.body));
        } catch (_err) {
          /* keep server JSON */
        }
        if (telemetry && data && res.ok) {
          const rtt = performance.now() - tClass;
          const cls = Number(data.classifier_ms != null ? data.classifier_ms : (data.meta && data.meta.classifier_ms));
          if (Number.isFinite(cls)) {
            telemetry.recordClassifier({
              classifier_ms: cls,
              rtt_ms: rtt,
            });
            refreshLatencyHud();
          }
        }
        if (data && data.meta) renderDecision(data);
        return new Response(JSON.stringify(data), {
          status: res.status,
          statusText: res.statusText,
          headers: { "Content-Type": "application/json" },
        });
      });
    }
    return origFetch(url, opts);
  };

  function mockChoice(ids, pick) {
    const probs = {};
    const rest = Math.max(ids.length - 1, 1);
    for (const id of ids) probs[id] = id === pick ? 0.85 : 0.15 / rest;
    const total = Object.values(probs).reduce((a, b) => a + b, 0) || 1;
    for (const id of ids) probs[id] = Math.round((probs[id] / total) * 1e4) / 1e4;
    return { choice: pick, probabilities: probs };
  }

  function fillJevAnswers(data, req) {
    if (!data || typeof data !== "object") return data;
    const answers = data.answers || (data.answers = {});
    const qs = (req && req.questions) || {};
    const motionCrit = qs.motion && qs.motion.criteria;
    if (motionCrit && typeof motionCrit === "object" && !answers.motion) {
      const ids = Object.keys(motionCrit);
      if (ids.length) {
        const pick = ids.indexOf("drive") >= 0 ? "drive" : ids[0];
        answers.motion = mockChoice(ids, pick);
      }
    }
    const vectorCrit = qs.vector && qs.vector.criteria;
    if (vectorCrit && typeof vectorCrit === "object" && answers.vector) {
      const ids = Object.keys(vectorCrit);
      if (ids.length) {
        const raw = answers.vector.probabilities || {};
        const pick = ids.indexOf(answers.vector.choice) >= 0 ? answers.vector.choice : ids[0];
        const filled = {};
        for (const id of ids) {
          const p = Number(raw[id]);
          filled[id] = Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : id === pick ? 0.8 : 0;
        }
        const sum = Object.values(filled).reduce((a, b) => a + b, 0) || 1;
        for (const id of ids) filled[id] = filled[id] / sum;
        answers.vector = { choice: pick, probabilities: filled };
      }
    }
    return data;
  }

  function renderDecision(data) {
    const meta = data.meta || {};
    const intent = meta.tier1_maneuver || "SemArbiter";
    const choice = data.answers && data.answers.vector && data.answers.vector.choice;
    intentEl.textContent = choice ? `${intent} · ${choice}` : String(intent).replace(/_/g, " ");
  }

  function project(camera, x, y, z, width, height) {
    if (!camera || !camera.matrixWorldInverse || !camera.projectionMatrix) return null;
    const e = camera.matrixWorldInverse.elements;
    const p = camera.projectionMatrix.elements;
    const cx = e[0] * x + e[4] * y + e[8] * z + e[12];
    const cy = e[1] * x + e[5] * y + e[9] * z + e[13];
    const cz = e[2] * x + e[6] * y + e[10] * z + e[14];
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
    const nx = p[0] * cx + p[4] * cy + p[8] * cz + p[12] * cw;
    const ny = p[1] * cx + p[5] * cy + p[9] * cz + p[13] * cw;
    const nz = p[2] * cx + p[6] * cy + p[10] * cz + p[14] * cw;
    const nw = p[3] * cx + p[7] * cy + p[11] * cz + p[15] * cw;
    if (!nw) return null;
    const ndcZ = nz / nw;
    if (ndcZ < -1 || ndcZ > 1) return null;
    return {
      x: (nx / nw * 0.5 + 0.5) * width,
      y: (-ny / nw * 0.5 + 0.5) * height,
    };
  }

  function aheadOf(player, dist, lateral) {
    const h = player.heading || 0;
    const right = (lateral || 0);
    return {
      x: player.x + Math.sin(h) * dist + Math.cos(h) * right,
      z: player.z - Math.cos(h) * dist + Math.sin(h) * right,
    };
  }

  function dist2(a, b) {
    const dx = a.x - b.x;
    const dz = a.z - b.z;
    return Math.hypot(dx, dz);
  }

  let lastInject = -1e9;
  function injectFrustumEvents(sim, dt) {
    if (!sim || !sim.player) return;
    const player = sim.player;
    sim._fsdAgents = sim._fsdAgents || [];
    if (sim.time - lastInject > 7.5) {
      lastInject = sim.time;
      const roll = (sim.planRandom ? sim.planRandom() : Math.random());
      if (roll < 0.45) {
        const pose = aheadOf(player, 26 + roll * 12, (roll > 0.2 ? 6 : -6));
        sim._fsdAgents.push({
          id: `frustum-ped-${sim.time.toFixed(2)}`,
          type: "pedestrian",
          x: pose.x,
          z: pose.z,
          heading: player.heading + Math.PI / 2,
          walking: true,
          direction: 1,
          speed: 1.4,
          height: 1.7,
          _fsd: "pedestrian",
          walkPath: { heading: player.heading + Math.PI / 2 },
        });
      } else if (roll < 0.75) {
        const pose = aheadOf(player, 22, 3.1);
        sim._fsdAgents.push({
          id: `frustum-park-${sim.time.toFixed(2)}`,
          type: "car",
          x: pose.x,
          z: pose.z,
          heading: player.heading,
          speed: 0,
          height: 1.5,
          hazardLights: true,
          _fsd: "parked",
        });
      } else {
        const pose = aheadOf(player, 24, 3.4);
        sim._fsdAgents.push({
          id: `frustum-cut-${sim.time.toFixed(2)}`,
          type: "car",
          x: pose.x,
          z: pose.z,
          heading: player.heading,
          speed: Math.max(6, (player.speed || 12) * 0.7),
          height: 1.5,
          _fsd: "cutin",
          _cut: 1.6,
        });
      }
    }

    for (const agent of sim._fsdAgents) {
      if (agent._fsd === "pedestrian" && agent.walking && agent.walkPath) {
        const h = agent.walkPath.heading;
        agent.x += Math.sin(h) * 1.6 * dt * agent.direction;
        agent.z -= Math.cos(h) * 1.6 * dt * agent.direction;
      }
      if (agent._fsd === "cutin" && agent._cut > 0) {
        const h = player.heading;
        agent.x -= Math.cos(h) * 1.4 * dt;
        agent.z -= Math.sin(h) * 0.2 * dt;
        agent._cut -= dt;
      }
    }
    sim._fsdAgents = sim._fsdAgents.filter((a) => dist2(a, player) < 80);
  }

  function drawBoxes(sim, world) {
    boxes.innerHTML = "";
    if (!sim || !world || !world.camera || !world.canvas) return;
    const w = world.canvas.clientWidth;
    const h = world.canvas.clientHeight;
    const camera = world.camera;
    const player = sim.player;
    const targets = []
      .concat(sim.pedestrians || [])
      .concat((sim.traffic || []).filter((v) => v.hazardLights));
    let nearest = Infinity;
    for (const obj of targets) {
      const d = dist2(obj, player);
      if (d < nearest) nearest = d;
      if (d > 55) continue;
      const top = project(camera, obj.x, (obj.height || 1.6), obj.z, w, h);
      const bot = project(camera, obj.x, 0.05, obj.z, w, h);
      if (!top || !bot) continue;
      const height = Math.max(18, Math.abs(bot.y - top.y));
      const width = Math.max(16, height * 0.45);
      const el = document.createElement("div");
      el.className = "fsd-box" + (obj.hazardLights ? " is-hazard" : "");
      el.style.left = `${top.x - width / 2}px`;
      el.style.top = `${Math.min(top.y, bot.y)}px`;
      el.style.width = `${width}px`;
      el.style.height = `${height}px`;
      const label = obj.type === "pedestrian" ? `PED ${d.toFixed(0)}m` : `VEH ${d.toFixed(0)}m`;
      el.innerHTML = `<span>${label}</span><i class="fsd-vec"></i>`;
      boxes.appendChild(el);
    }
    if (nearest < 12) {
      halo.style.borderColor = "rgba(239, 68, 68, 0.8)";
      halo.style.boxShadow = "0 0 36px 8px rgba(239, 68, 68, 0.35)";
    } else if (nearest < 24) {
      halo.style.borderColor = "rgba(62, 106, 225, 0.85)";
      halo.style.boxShadow = "0 0 28px 6px rgba(62, 106, 225, 0.28)";
    } else {
      halo.style.borderColor = "transparent";
      halo.style.boxShadow = "none";
    }
  }

  const pipFrameMs = [];
  function pipFpsText(now) {
    pipFrameMs.push(now);
    const cutoff = now - 1000;
    while (pipFrameMs.length && pipFrameMs[0] < cutoff) pipFrameMs.shift();
    if (pipFrameMs.length < 2) return "-- FPS";
    const span = pipFrameMs[pipFrameMs.length - 1] - pipFrameMs[0];
    if (span <= 0) return "-- FPS";
    return Math.round(((pipFrameMs.length - 1) * 1000) / span) + " FPS";
  }

  function onboardMount(player) {
    const ahead = 0.15;
    const height = 1.45;
    const h = Number(player.heading) || 0;
    const x = player.x + Math.sin(h) * ahead;
    const z = player.z - Math.cos(h) * ahead;
    return {
      x: x,
      y: height,
      z: z,
      lookX: x + Math.sin(h) * 25,
      lookY: height,
      lookZ: z - Math.cos(h) * 25,
    };
  }

  function paintOnboardPixels(pixels) {
    const ctx = pipCanvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(PIP_W, PIP_H);
    const row = PIP_W * 4;
    for (let y = 0; y < PIP_H; y++) {
      const src = (PIP_H - 1 - y) * row;
      image.data.set(pixels.subarray(src, src + row), y * row);
    }
    ctx.putImageData(image, 0, 0);
  }

  function renderOnboard(world) {
    const player = world && world.sim && world.sim.player;
    const renderer = world && world.renderer;
    const scene = world && world.scene;
    const sample = world && world.sun && world.sun.shadow && world.sun.shadow.map;
    if (!player || !renderer || !scene || !sample || !world.camera || !pipCanvas) return false;
    const mount = onboardMount(player);
    if (!world._onboardCam) world._onboardCam = world.camera.clone();
    const cam = world._onboardCam;
    cam.fov = 60;
    cam.aspect = PIP_W / PIP_H;
    cam.position.set(mount.x, mount.y, mount.z);
    cam.lookAt(mount.lookX, mount.lookY, mount.lookZ);
    if (cam.updateProjectionMatrix) cam.updateProjectionMatrix();
    if (cam.updateMatrixWorld) cam.updateMatrixWorld();
    if (!world._onboardTarget) {
      world._onboardTarget = new sample.constructor(PIP_W, PIP_H);
    }
    const target = world._onboardTarget;
    target.isXRRenderTarget = true;
    if (target.texture) {
      target.texture.colorSpace = renderer.outputColorSpace || "srgb";
      target.texture.internalFormat = "RGBA8";
    }
    const hidden = [];
    if (world.player && world.player.traverse) {
      world.player.traverse((obj) => {
        if (obj.isMesh && obj.material && obj.material.name === "Glass" && obj.visible) {
          hidden.push(obj);
          obj.visible = false;
        }
      });
    }
    const prev = renderer.getRenderTarget ? renderer.getRenderTarget() : null;
    try {
      renderer.setRenderTarget(target);
      renderer.render(scene, cam);
      const pixels = new Uint8Array(PIP_W * PIP_H * 4);
      if (renderer.readRenderTargetPixels) {
        renderer.readRenderTargetPixels(target, 0, 0, PIP_W, PIP_H, pixels);
      }
      paintOnboardPixels(pixels);
    } finally {
      if (renderer.setRenderTarget) renderer.setRenderTarget(prev);
      hidden.forEach((obj) => {
        obj.visible = true;
      });
    }
    return true;
  }

  function tick() {
    const sim = window.SEMIF_SIM;
    const world = window.SEMIF_WORLD;
    if (sim) {
      if (sim._fsdBound !== true) {
        sim._fsdBound = true;
        applyRawMode(window.SEMIF_RAW_MODE);
        const orig = sim.step.bind(sim);
        sim.step = function (dt) {
          sim._pdDt = dt;
          try {
            applyLaneKeepReference(sim);
          } catch (_err) {
            /* PD must not kill the drive loop */
          }
          let out;
          try {
            out = orig(dt);
          } catch (err) {
            console.warn("semif: sim.step", err);
            return out;
          }
          try {
            injectFrustumEvents(sim, Math.min(dt || 0.016, 0.05));
          } catch (_err) {
            /* HUD ghosts must not kill the drive loop */
          }
          try {
            requestEgoReplan(sim, dt);
          } catch (_err) {
            /* replan must not kill the drive loop */
          }
          return out;
        };
      }
      const seed = sim.world && sim.world.seed;
      if (seed != null && document.activeElement !== seedInput) seedInput.value = String(seed);
      drawBoxes(sim, world);
    }
    try {
      if (renderOnboard(world) && pipFps) pipFps.textContent = pipFpsText(performance.now());
    } catch (_err) {
      /* The onboard view must not kill the drive loop */
    }
    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  function grabFrame() {
    const world = window.SEMIF_WORLD;
    if (!renderOnboard(world) || !pipCanvas) return null;
    return pipCanvas.toDataURL("image/jpeg", 0.55);
  }

  window.SEMIF_GRAB_FRAME = grabFrame;

  async function visionTick() {
    if (!visionOn) {
      visionEl.textContent = "VISION off";
      return;
    }
    const tGrab = performance.now();
    const dataUrl = grabFrame();
    const grabMs = performance.now() - tGrab;
    if (!dataUrl) {
      visionEl.textContent = "VISION waiting";
      return;
    }
    try {
      const tVis = performance.now();
      const res = await origFetch("/v1/vision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const data = await res.json();
      const rttMs = performance.now() - tVis;
      if (!res.ok || data.error) {
        visionEl.textContent = "VISION error";
        return;
      }
      const vis = data.vision || data;
      window.SEMIF_VISION = vis;
      if (telemetry) {
        const encode = Number(
          data.vision_encode_ms != null ? data.vision_encode_ms : vis.latency_ms
        );
        if (Number.isFinite(encode)) {
          telemetry.recordVision({
            encode_ms: encode,
            rtt_ms: rttMs,
            grab_ms: grabMs,
          });
          refreshLatencyHud();
        }
      }
      const sig = vis.signal || "unknown";
      visionEl.textContent = vis.event
        ? ["VISION", vis.event].join(" · ")
        : [
            "VISION",
            vis.backend || "stub",
            "sig " + sig,
            "ped " + (vis.pedestrian != null ? Number(vis.pedestrian).toFixed(2) : "—"),
            "veh " + (vis.vehicle != null ? Number(vis.vehicle).toFixed(2) : "—"),
          ].join(" · ");
    } catch (_err) {
      visionEl.textContent = "VISION error";
    }
  }

  if (visionOn) {
    setInterval(visionTick, 700);
    setTimeout(visionTick, 1500);
  }
})();
