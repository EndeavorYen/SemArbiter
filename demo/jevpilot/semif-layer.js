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
    <div id="fsd-status" aria-live="polite">
      <span id="fsd-seed"></span>
      <span id="fsd-intent">SemIf</span>
      <span id="fsd-vision">VISION off</span>
    </div>
  `;
  document.body.appendChild(chrome);

  const halo = document.getElementById("fsd-halo");
  const boxes = document.getElementById("fsd-boxes");
  const seedEl = document.getElementById("fsd-seed");
  const visionEl = document.getElementById("fsd-vision");
  const intentEl = document.getElementById("fsd-intent");
  const visionOn = params.get("vision") !== "0";
  window.SEMIF_VISION = null;

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
        const player = window.SEMIF_SIM && window.SEMIF_SIM.player;
        if (player && typeof player.x === "number") {
          body.state = body.state || {};
          body.state.lateral_offset_m = player.x;
        }
        if (window.SEMIF_VISION) {
          body.state = body.state || {};
          body.state.vision = window.SEMIF_VISION;
        }
        opts = Object.assign({}, opts, { body: JSON.stringify(body) });
      } catch (_err) {
        /* leave request unchanged */
      }
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

  function tick() {
    const sim = window.SEMIF_SIM;
    const world = window.SEMIF_WORLD;
    if (sim) {
      if (sim._fsdBound !== true) {
        sim._fsdBound = true;
        applyRawMode(window.SEMIF_RAW_MODE);
        const orig = sim.step.bind(sim);
        sim.step = function (dt) {
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
          return out;
        };
      }
      const seed = sim.world && sim.world.seed;
      seedEl.textContent = seed != null ? `seed ${seed}` : "";
      drawBoxes(sim, world);
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
    const dataUrl = grabFrame();
    if (!dataUrl) {
      visionEl.textContent = "VISION waiting";
      return;
    }
    try {
      const res = await origFetch("/v1/vision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const data = await res.json();
      const vis = data.vision || data;
      window.SEMIF_VISION = vis;
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
