# Decision readout versus generated tokens

Open `index.html` directly in a current browser. It has no dependencies, network requests, or model execution.

The replay shows one measured request on each side: the same frozen BF16 Qwen3.5-4B, owned state, 21 binary criteria, and RTX 3090.

| Path | Completion | Output |
|---|---:|---:|
| Direct typed logits | 1.023 s median | 21 probability pairs; 0 generated tokens |
| Compact JSON array | 5.332 s median | Valid 21-value array; 111 generated tokens |

The generative path emitted its first token at a median 0.489 seconds and then visibly streams its recorded answer. The direct output appears together at its measured completion point. Short labels make the questions readable. Open `index.html` for the interactive replay; the repository media are static previews of that page.

Included project-owned media:

- `assets/semif-phase1-replay.gif` — GitHub README preview.
- `assets/semif-phase1-replay.webm` — 1280 × 720 VP9 preview.
- `assets/semif-phase1-replay-poster.png` — poster frame.
- `assets/semif-social-preview.png` — 1200 × 630 social preview.

The visual compares output paths, not semantic correctness. See the main [results](../docs/RESULTS.md) and [method](../docs/METHOD.md) for quality and scope.

---

# JevPilot: 3D Autonomous Driving Simulator & Live Decision Server

Inspired by [`featherless-ai/simple-jev`](https://github.com/featherless-ai/simple-jev) and `jevpilot`, this interactive demo provides a real-time 3D highway simulation powered by SemIf's sub-15ms semantic decision engine.

### Highlights
- **Zero-Build 3D Client** (`demo/jevpilot/index.html`): Pure Three.js from CDN with dynamic cyber highway, procedural road curves, obstacle traffic, and full vehicle dynamics.
- **WebSocket Streaming Server** (`demo/server.py`): Real-time `/stream-decide` bi-directional JSON streaming with Qwen2.5-3B-Instruct, Sliced LM Head, and CUDA Graphs.
- **Three Pilot Modes**:
  1. **Manual [1]**: WASD / Arrow key player control.
  2. **Raw LLM [2]**: Demonstrates catastrophic Option-A prior logit bias (75% off-track drift).
  3. **SemIf Autopilot [3]**: Prior-calibrated logits, smooth lane keeping, overtaking, and obstacle deceleration.
- **Helmholtz Free Energy OOD Safety Gate [4]**: Press Chaos Monkey (`[4]`) to inject sensor corruption. SemIf detects the free-energy spike ($E > -20.0$) and immediately activates the Autonomous Emergency Brake fail-safe!

### Quick Start
```bash
# 1. Launch the server (Zero-GPU Mock mode)
python demo/server.py --mock --port 8000

# Or launch with real Qwen2.5-3B-Instruct on CUDA GPU:
python demo/server.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --port 8000

# 2. Open in your browser
# Navigate to: http://localhost:8000/jevpilot/
# Replay a world: http://localhost:8000/jevpilot/?seed=42
# Vision HUD (canvas → /v1/vision → state.vision): default on; ?vision=0 to skip
# Raw decision mode: ?raw=1
# Or open demo/jevpilot/index.html directly (features automatic offline fallback engine)
```

Closed-loop mock benchmark with a fixed seed matrix:

```bash
python benchmarks/benchmark_jevpilot_hierarchical.py --mock --episodes 2 --seed 42 --raw-mode \
  --output results/jevpilot-sdi-mock.json
```

### Closed-Loop Benchmark Results (20 Episodes per Mode)
| Mode | Completion Rate | Collisions | Off-Track Departures | OOD Anomaly Recall | Latency P50 |
|---|---:|---:|---:|---:|---:|
| **Heuristic Rule-Based** | 85.0% | 0.0% | 15.0% | 100.0% | 0.0 ms |
| **Raw Qwen2.5-3B** | 0.0% | 25.0% | 75.0% | 0.0% | 23.7 ms |
| **SemIf Enhanced Qwen2.5-3B** | **25.0%** | 25.0% | **50.0%** | **100.0%** | **39.3 ms** |
