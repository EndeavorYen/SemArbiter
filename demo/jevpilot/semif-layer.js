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
    <div id="fsd-gear" aria-label="Gear">
      <span data-gear="P">P</span>
      <span data-gear="R">R</span>
      <span data-gear="N">N</span>
      <span data-gear="D" data-active="true">D</span>
    </div>
    <div id="fsd-pill" role="status"></div>
    <div id="fsd-halo" aria-hidden="true"></div>
    <div id="fsd-boxes"></div>
    <button id="fsd-raw-toggle" type="button" aria-pressed="false">Raw decision</button>
    <button id="fsd-drawer-toggle" type="button">Decision</button>
    <aside id="fsd-drawer" data-open="false">
      <h2>Decision</h2>
      <div class="fsd-intent" id="fsd-intent">SAFE_CRUISE</div>
      <p id="fsd-directive">Waiting for the first SemIf decision.</p>
      <div id="fsd-probs"></div>
    </aside>
    <div id="fsd-seed"></div>
    <div id="fsd-vision" aria-label="Vision">VISION off</div>
  `;
  document.body.appendChild(chrome);

  const pill = document.getElementById("fsd-pill");
  const halo = document.getElementById("fsd-halo");
  const boxes = document.getElementById("fsd-boxes");
  const drawer = document.getElementById("fsd-drawer");
  const rawBtn = document.getElementById("fsd-raw-toggle");
  const seedEl = document.getElementById("fsd-seed");
  const visionEl = document.getElementById("fsd-vision");
  const visionOn = params.get("vision") !== "0";
  window.SEMIF_VISION = null;
  const intentEl = document.getElementById("fsd-intent");
  const directiveEl = document.getElementById("fsd-directive");
  const probsEl = document.getElementById("fsd-probs");

  document.getElementById("fsd-drawer-toggle").addEventListener("click", () => {
    drawer.dataset.open = drawer.dataset.open === "true" ? "false" : "true";
  });

  function applyRawMode(on) {
    window.SEMIF_RAW_MODE = !!on;
    rawBtn.setAttribute("aria-pressed", on ? "true" : "false");
    rawBtn.textContent = on ? "Raw decision ON" : "Raw decision";
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

  rawBtn.addEventListener("click", () => applyRawMode(!window.SEMIF_RAW_MODE));

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
        if (window.SEMIF_VISION) {
          body.state = body.state || {};
          body.state.vision = window.SEMIF_VISION;
        }
        opts = Object.assign({}, opts, { body: JSON.stringify(body) });
      } catch (_err) {
        /* leave request unchanged */
      }
      return origFetch(url, opts).then((res) => {
        const clone = res.clone();
        clone.json().then((data) => {
          if (data && data.meta) renderDecision(data);
        }).catch(() => {});
        return res;
      });
    }
    return origFetch(url, opts);
  };

  function renderDecision(data) {
    const meta = data.meta || {};
    const intent = meta.tier1_maneuver || "SAFE_CRUISE";
    intentEl.textContent = intent;
    directiveEl.textContent = meta.tier1_directive || "";
    const show = intent && intent !== "SAFE_CRUISE" && intent !== "GEOMETRIC_HEURISTIC" && intent !== "CRUISE";
    pill.textContent = String(intent).replace(/_/g, " ");
    pill.dataset.show = show ? "true" : "false";
    const vector = (data.answers && data.answers.vector) || {};
    const probs = vector.probabilities || {};
    probsEl.innerHTML = Object.keys(probs)
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, 6)
      .map((id) => {
        const pct = Math.round((probs[id] || 0) * 100);
        return `<div class="fsd-prob"><b>${id}</b><span class="fsd-bar"><i style="width:${pct}%"></i></span><span>${pct}%</span></div>`;
      })
      .join("");
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
        const ped = {
          id: `frustum-ped-${sim.time.toFixed(2)}`,
          type: "pedestrian",
          x: pose.x,
          z: pose.z,
          heading: player.heading + Math.PI / 2,
          walking: true,
          crossing: true,
          jaywalker: true,
          progress: 0,
          direction: 1,
          speed: 1.4,
          width: 0.6,
          depth: 0.6,
          height: 1.7,
          nodeId: "frustum",
          _fsd: "pedestrian",
          walkPath: { start: pose, heading: player.heading + Math.PI / 2, length: 14 },
        };
        sim.pedestrians.push(ped);
        sim._fsdAgents.push(ped);
      } else if (roll < 0.75) {
        const pose = aheadOf(player, 22, 3.1);
        const parked = {
          id: `frustum-park-${sim.time.toFixed(2)}`,
          type: "car",
          x: pose.x,
          z: pose.z,
          heading: player.heading,
          speed: 0,
          width: 1.8,
          depth: 4.4,
          height: 1.5,
          hazardLights: true,
          _fsd: "parked",
        };
        sim.traffic.push(parked);
        sim._fsdAgents.push(parked);
      } else {
        const pose = aheadOf(player, 24, 3.4);
        const cut = {
          id: `frustum-cut-${sim.time.toFixed(2)}`,
          type: "car",
          x: pose.x,
          z: pose.z,
          heading: player.heading,
          speed: Math.max(6, (player.speed || 12) * 0.7),
          width: 1.8,
          depth: 4.4,
          height: 1.5,
          _fsd: "cutin",
          _cut: 1.6,
        };
        sim.traffic.push(cut);
        sim._fsdAgents.push(cut);
      }
    }

    for (const agent of sim._fsdAgents) {
      if (agent._fsd === "pedestrian" && agent.walking) {
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
      .concat((sim.traffic || []).filter((v) => v.hazardLights || v._fsd));
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

  function syncGear(sim) {
    const speed = sim && sim.player ? sim.player.speed : 0;
    const gear = speed < -0.4 ? "R" : speed < 0.35 ? "P" : "D";
    for (const el of document.querySelectorAll("#fsd-gear span")) {
      el.dataset.active = el.dataset.gear === gear ? "true" : "false";
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
          injectFrustumEvents(sim, Math.min(dt || 0.016, 0.05));
          return orig(dt);
        };
      }
      const seed = sim.world && sim.world.seed;
      seedEl.textContent = seed != null ? `seed ${seed}` : "";
      syncGear(sim);
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
      visionEl.textContent = [
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
