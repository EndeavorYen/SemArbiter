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
        <span class="fsd-pip-title">CAMERA: Front Main (60° FOV)</span>
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
      <span id="fsd-intent">SemIf</span>
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
  const PIP_GROUND_Y = 0.05;
  const PIP_RIBBON = "rgba(34, 211, 238, 0.45)";
  const PIP_PED = "#22c55e";
  const PIP_CUT = "#f5c518";
  const PIP_LIGHT_RED = "#ef4444";
  const PIP_HORIZON_S = 3;
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

  const STEER_LIMIT = 0.85;
  const LATERAL_KP = 0.45;
  const LATERAL_KD = 0.15;
  const LATERAL_PD_LIMIT = 0.20;
  const DETOUR_STEER = 0.28;
  const LANE_KEEP_OFFSET_M = 1.4;
  const LOOKAHEAD_MIN_M = 4.0;
  const LOOKAHEAD_MAX_M = 8.0;
  const LOOKAHEAD_S = 0.40;
  const YAW_KD = 0.08;
  const STEER_SLEW = 0.9;

  function clipSteer(u) {
    return Math.max(-STEER_LIMIT, Math.min(STEER_LIMIT, u));
  }

  function lateralPd(uSelected, offsetM, offsetDot) {
    const u = Number(uSelected) || 0;
    if (Math.abs(u) > DETOUR_STEER) return clipSteer(u);
    let pd = LATERAL_KP * offsetM + LATERAL_KD * offsetDot;
    if (pd > LATERAL_PD_LIMIT) pd = LATERAL_PD_LIMIT;
    else if (pd < -LATERAL_PD_LIMIT) pd = -LATERAL_PD_LIMIT;
    return clipSteer(u - pd);
  }

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

  function dampenStanleySteer(uSelected, yawRate, prevU, dt) {
    let u = (Number(uSelected) || 0) - YAW_KD * (Number(yawRate) || 0);
    const step = Math.max(0.001, Number(dt) || 0.016);
    if (prevU != null && Number.isFinite(prevU)) {
      const maxDu = STEER_SLEW * step;
      const du = u - prevU;
      if (du > maxDu) u = prevU + maxDu;
      else if (du < -maxDu) u = prevU - maxDu;
    }
    return clipSteer(u);
  }

  function applySteerCommand(uA, offsetM, offsetDot, yawRate, prevU, dt) {
    return dampenStanleySteer(lateralPd(uA, offsetM, offsetDot), yawRate, prevU, dt);
  }

  function laneOffsetM(sim) {
    const lane =
      (sim.lastDecisionState && sim.lastDecisionState.lane) ||
      (sim.lastPlan && sim.lastPlan.lane);
    if (lane && typeof lane.offset_m === "number" && Number.isFinite(lane.offset_m)) {
      return lane.offset_m;
    }
    return null;
  }

  function realtimeLaneOffsetM(sim) {
    const p = sim && sim.player;
    const route = p && p.route && p.route.points;
    if (!p || !route || route.length < 2) {
      const snap = laneOffsetM(sim);
      return snap == null ? 0 : snap;
    }
    const s0 = Number.isFinite(p.s) ? p.s : null;
    let best = null;
    let bestD2 = Infinity;
    for (let i = 0; i < route.length - 1; i++) {
      const a = route[i];
      const b = route[i + 1];
      if (s0 != null && a.s != null && a.s < s0 - 12) continue;
      if (s0 != null && a.s != null && a.s > s0 + 32) break;
      const abx = b.x - a.x;
      const abz = b.z - a.z;
      const len2 = abx * abx + abz * abz || 1;
      let t = ((p.x - a.x) * abx + (p.z - a.z) * abz) / len2;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      const qx = a.x + abx * t;
      const qz = a.z + abz * t;
      const dx = p.x - qx;
      const dz = p.z - qz;
      const d2 = dx * dx + dz * dz;
      if (d2 < bestD2) {
        bestD2 = d2;
        const h = Number.isFinite(a.heading) ? a.heading : Math.atan2(abx, -abz);
        best = dx * Math.cos(h) + dz * Math.sin(h);
      }
    }
    if (best == null) {
      const snap = laneOffsetM(sim);
      return snap == null ? 0 : snap;
    }
    return best;
  }

  window.SEMIF_APPLY_STEER = function (player, u) {
    const sim = window.SEMIF_SIM;
    if (!sim || !sim.autopilot || sim.paused || sim.crash || !player) return u;
    const dt = Math.min(Math.max(Number(sim._pdDt) || 0.016, 0.008), 0.05);
    const offset = realtimeLaneOffsetM(sim);
    const prevOff = sim._lastOffset;
    const offsetDot = prevOff == null ? 0 : (offset - prevOff) / dt;
    sim._lastOffset = offset;
    const heading = Number(player.heading) || 0;
    let yaw = 0;
    if (sim._pdHeading != null && Number.isFinite(sim._pdHeading)) {
      let d = heading - sim._pdHeading;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      yaw = d / dt;
    }
    sim._pdHeading = heading;
    const out = applySteerCommand(u, offset, offsetDot, yaw, sim._pdU, dt);
    sim._pdU = out;
    return out;
  };

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
    const intent = meta.tier1_maneuver || "SemIf";
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

  function invertMat4(m) {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    det = 1 / det;
    return [
      (a11 * b11 - a12 * b10 + a13 * b09) * det,
      (a02 * b10 - a01 * b11 - a03 * b09) * det,
      (a31 * b05 - a32 * b04 + a33 * b03) * det,
      (a22 * b04 - a21 * b05 - a23 * b03) * det,
      (a12 * b08 - a10 * b11 - a13 * b07) * det,
      (a00 * b11 - a02 * b08 + a03 * b07) * det,
      (a32 * b02 - a30 * b05 - a33 * b01) * det,
      (a20 * b05 - a22 * b02 + a23 * b01) * det,
      (a10 * b10 - a11 * b08 + a13 * b06) * det,
      (a01 * b08 - a00 * b10 - a03 * b06) * det,
      (a30 * b04 - a31 * b02 + a33 * b00) * det,
      (a21 * b02 - a20 * b04 - a23 * b00) * det,
      (a11 * b07 - a10 * b09 - a12 * b06) * det,
      (a00 * b09 - a01 * b07 + a02 * b06) * det,
      (a31 * b01 - a30 * b03 - a32 * b00) * det,
      (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ];
  }

  function mulMat4(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[r] * b[c * 4] +
          a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] +
          a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }

  function applyMat(m, x, y, z, w) {
    return {
      x: m[0] * x + m[4] * y + m[8] * z + m[12] * w,
      y: m[1] * x + m[5] * y + m[9] * z + m[13] * w,
      z: m[2] * x + m[6] * y + m[10] * z + m[14] * w,
      w: m[3] * x + m[7] * y + m[11] * z + m[15] * w,
    };
  }

  function unprojectGround(camera, px, py, width, height) {
    if (!camera || !camera.matrixWorldInverse || !camera.projectionMatrix) return null;
    const inv = invertMat4(mulMat4(camera.projectionMatrix.elements, camera.matrixWorldInverse.elements));
    if (!inv) return null;
    const nx = (px / width) * 2 - 1;
    const ny = 1 - (py / height) * 2;
    function at(ndcZ) {
      const p = applyMat(inv, nx, ny, ndcZ, 1);
      if (!p.w) return null;
      return { x: p.x / p.w, y: p.y / p.w, z: p.z / p.w };
    }
    const a = at(-1);
    const b = at(1);
    if (!a || !b) return null;
    const dy = b.y - a.y;
    if (Math.abs(dy) < 1e-8) return null;
    const t = (PIP_GROUND_Y - a.y) / dy;
    return {
      x: a.x + (b.x - a.x) * t,
      y: PIP_GROUND_Y,
      z: a.z + (b.z - a.z) * t,
    };
  }

  function fitContain(srcW, srcH, dstW, dstH) {
    const sw = Math.max(1, srcW);
    const sh = Math.max(1, srcH);
    const scale = Math.min(dstW / sw, dstH / sh);
    const w = sw * scale;
    const h = sh * scale;
    return { x: (dstW - w) / 2, y: (dstH - h) / 2, w: w, h: h, scale: scale };
  }

  function mapPip(px, py, srcW, srcH, fit) {
    return {
      x: fit.x + (px / srcW) * fit.w,
      y: fit.y + (py / srcH) * fit.h,
    };
  }

  function egoRel(player, x, z) {
    const h = Number(player && player.heading) || 0;
    const dx = x - (Number(player && player.x) || 0);
    const dz = z - (Number(player && player.z) || 0);
    return {
      rel_x: dx * Math.cos(h) + dz * Math.sin(h),
      rel_z: dx * Math.sin(h) - dz * Math.cos(h),
    };
  }

  function ribbonSpeed(player, maneuver) {
    if (maneuver && Number.isFinite(Number(maneuver.velocity_mps))) {
      return Math.max(0, Number(maneuver.velocity_mps));
    }
    if (maneuver && Number.isFinite(Number(maneuver.end_speed_mps))) {
      return Math.max(0, Number(maneuver.end_speed_mps));
    }
    return Math.max(0, Number(player && player.speed) || 0);
  }

  function pointOnRoute(points, s) {
    let i = 0;
    if (s > points[0].s) {
      i = points.length - 2;
      for (let k = 0; k < points.length - 1; k++) {
        if (s <= points[k + 1].s) {
          i = k;
          break;
        }
      }
    }
    const a = points[i];
    const b = points[i + 1];
    const span = (b.s - a.s) || 1;
    let t = (s - a.s) / span;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    return {
      x: a.x + (b.x - a.x) * t,
      z: a.z + (b.z - a.z) * t,
      heading: Math.atan2(b.x - a.x, -(b.z - a.z)),
    };
  }

  function ribbonPoints(player, maneuver) {
    const speed = ribbonSpeed(player, maneuver);
    const dist = Math.max(0.5, speed * PIP_HORIZON_S);
    const offset = maneuver && Number.isFinite(Number(maneuver.lane_offset_m))
      ? Number(maneuver.lane_offset_m)
      : 0;
    const look = Math.max(1, Number(maneuver && maneuver.lookahead_m) || 6);
    const route = player && player.route && player.route.points;
    const originS = Number(player && player.s);
    const pts = [];
    for (let i = 0; i <= 8; i++) {
      const along = (dist * i) / 8;
      const lat = offset * Math.min(1, along / look);
      if (route && route.length >= 2 && Number.isFinite(originS)) {
        const pose = pointOnRoute(route, originS + along);
        const moved = aheadOf(pose, 0, lat);
        moved.heading = Number.isFinite(Number(pose.heading)) ? Number(pose.heading) : 0;
        pts.push(moved);
      } else {
        const moved = aheadOf(player, along, lat);
        moved.heading = Number(player.heading) || 0;
        pts.push(moved);
      }
    }
    return pts;
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

  function pipScene(sim) {
    if (!sim) return null;
    return sim.lastDecisionState || null;
  }

  function paintPip(sim, world, now) {
    if (!pipCanvas || !pipRoot || pipRoot.classList.contains("fsd-pip-hidden")) return;
    const ctx = pipCanvas.getContext("2d");
    if (!ctx) return;
    if (pipFps) pipFps.textContent = pipFpsText(now);
    const src = world && world.canvas;
    const srcW = src ? (src.width || src.clientWidth || PIP_W) : PIP_W;
    const srcH = src ? (src.height || src.clientHeight || PIP_H) : PIP_H;
    const fit = fitContain(srcW, srcH, PIP_W, PIP_H);
    ctx.clearRect(0, 0, PIP_W, PIP_H);
    if (src) ctx.drawImage(src, fit.x, fit.y, fit.w, fit.h);
    const camera = world && world.camera;
    const player = sim && sim.player;
    if (!camera || !player) return;
    const maneuver = player.maneuver;
    const center = ribbonPoints(player, maneuver);
    const left = [];
    const right = [];
    for (const pt of center) {
      const pose = {
        x: pt.x,
        z: pt.z,
        heading: Number.isFinite(pt.heading) ? pt.heading : (player.heading || 0),
      };
      const l = aheadOf(pose, 0, -0.9);
      const r = aheadOf(pose, 0, 0.9);
      const pl = project(camera, l.x, PIP_GROUND_Y, l.z, srcW, srcH);
      const pr = project(camera, r.x, PIP_GROUND_Y, r.z, srcW, srcH);
      if (!pl || !pr) continue;
      left.push(mapPip(pl.x, pl.y, srcW, srcH, fit));
      right.push(mapPip(pr.x, pr.y, srcW, srcH, fit));
    }
    if (left.length >= 2 && right.length >= 2) {
      ctx.beginPath();
      ctx.fillStyle = PIP_RIBBON;
      ctx.moveTo(left[0].x, left[0].y);
      for (let i = 1; i < left.length; i++) ctx.lineTo(left[i].x, left[i].y);
      for (let i = right.length - 1; i >= 0; i--) ctx.lineTo(right[i].x, right[i].y);
      ctx.closePath();
      ctx.fill();
    }
    const marks = []
      .concat(sim.pedestrians || [])
      .concat(sim._fsdAgents || [])
      .concat((sim.traffic || []).filter((obj) => obj && (obj.hazardLights || obj._fsd === "cutin")));
    for (const obj of marks) {
      const d = dist2(obj, player);
      if (d > 55) continue;
      const top = project(camera, obj.x, obj.height || 1.6, obj.z, srcW, srcH);
      const bot = project(camera, obj.x, PIP_GROUND_Y, obj.z, srcW, srcH);
      if (!top || !bot) continue;
      const topM = mapPip(top.x, top.y, srcW, srcH, fit);
      const botM = mapPip(bot.x, bot.y, srcW, srcH, fit);
      const boxH = Math.max(8, Math.abs(botM.y - topM.y));
      const boxW = Math.max(6, boxH * 0.45);
      const leftX = topM.x - boxW / 2;
      const topY = Math.min(topM.y, botM.y);
      const ped = obj.type === "pedestrian" || obj._fsd === "pedestrian";
      const cut = obj._fsd === "cutin";
      const tau = Number.isFinite(Number(obj._cut)) ? Number(obj._cut) : d / Math.max(0.5, Number(obj.speed) || 1);
      let stroke = ped ? PIP_PED : PIP_CUT;
      if (cut && tau < 1.5) stroke = PIP_LIGHT_RED;
      ctx.strokeStyle = stroke;
      ctx.strokeRect(leftX, topY, boxW, boxH);
      const contactX = leftX + boxW / 2;
      const contactY = Math.max(topM.y, botM.y);
      ctx.beginPath();
      ctx.moveTo(contactX - 4, contactY);
      ctx.lineTo(contactX + 4, contactY);
      ctx.moveTo(contactX, contactY - 4);
      ctx.lineTo(contactX, contactY + 4);
      ctx.stroke();
      const srcContactX = ((contactX - fit.x) / fit.w) * srcW;
      const srcContactY = ((contactY - fit.y) / fit.h) * srcH;
      const ground = unprojectGround(camera, srcContactX, srcContactY, srcW, srcH);
      const rel = ground ? egoRel(player, ground.x, ground.z) : egoRel(player, obj.x, obj.z);
      const where = "(" + rel.rel_x.toFixed(1) + ", " + rel.rel_z.toFixed(1) + ")";
      ctx.fillStyle = stroke;
      if (ped) ctx.fillText("PED " + d.toFixed(1) + "m", leftX, topY - 4);
      else if (cut) ctx.fillText("CUT-IN " + d.toFixed(1) + "m · tau " + tau.toFixed(1) + "s", leftX, topY - 4);
      else ctx.fillText("VEH " + d.toFixed(1) + "m", leftX, topY - 4);
      ctx.fillText(where, contactX + 6, contactY);
    }
    const scene = pipScene(sim);
    const inter = scene && scene.scene && scene.scene.intersection;
    const line = inter && inter.stop_line_position;
    if (line && inter.visible !== false && inter.signal) {
      const sig = String(inter.signal).toLowerCase();
      let color = "#22c55e";
      if (sig === "red") color = PIP_LIGHT_RED;
      else if (sig === "amber" || sig === "yellow") color = "#f5c518";
      const lamp = project(camera, Number(line.x) || 0, 4.2, Number(line.z) || 0, srcW, srcH);
      if (lamp) {
        const m = mapPip(lamp.x, lamp.y, srcW, srcH, fit);
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.arc(m.x, m.y, 6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  window.SEMIF_PIP = {
    project: project,
    unprojectGround: unprojectGround,
    fitContain: fitContain,
    egoRel: egoRel,
    ribbonPoints: ribbonPoints,
    paintPip: paintPip,
  };

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
      try {
        paintPip(sim, world, performance.now());
      } catch (_err) {
        /* PIP must not kill the drive loop */
      }
    }
    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  function grabFrame() {
    const world = window.SEMIF_WORLD;
    const canvas = (world && world.canvas) || document.getElementById("world-canvas");
    if (!canvas || !canvas.toDataURL) return null;
    const tmp = document.createElement("canvas");
    const w = 320;
    const h = Math.max(64, Math.round((canvas.height / Math.max(canvas.width, 1)) * w));
    tmp.width = w;
    tmp.height = h;
    tmp.getContext("2d").drawImage(canvas, 0, 0, w, h);
    return tmp.toDataURL("image/jpeg", 0.55);
  }

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
