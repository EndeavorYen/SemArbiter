# Repository instructions

- Run commands from the repository root in an isolated environment installed with `pip install -e '.[test]'`.
- Validate changes with `pytest -q`, `(cd results/raw && sha256sum -c SHA256SUMS)`, and `python benchmarks/verify_published.py`.
- Benchmark outputs are create-only. Use a new output path and expose exactly one CUDA GPU per scorer process.
- Do not change headline claims or `results/phase1-summary.json` without committing the supporting row-level evidence, regenerating the relevant raw report, updating `results/raw/SHA256SUMS`, and updating the method/results text.
- Preserve exact model and source revisions. Use `benchmarks/fetch_sources.py` only for its listed redistributable inputs; do not commit model weights, caches, or third-party raw records.
- `webgpu-demo/` is static and has no build step. Preserve `_headers`, runtime version pins, browser-only inference, and the explicit probability limitations.

# How to write here

Lead with the answer. Then say why, in short sentences. A longer sentence is fine after the point is already clear. Define a project term the first time it appears. Do not stack labels, tables, or file dumps in front of the conclusion. Do not dress a guess as a measured result.

# Evaluation honesty

JevPilot executors are Heuristic and Flat SemIf only. Both see the same sampled trajectories this frame. SemIf uses sliced-head readout and n-way prior. Hierarchical-as-pruning was retired for JevPilot (insight in docs/tutorials/10_jev_vs_classifier_io.md). Do not delete trajectories with a coarse tree. Constrain output schema, not the candidate set. Web and the Python loop share the option contract (no signal-injected candidates, six-column vectors, `VECTOR_INSTRUCTIONS`); they do not share a world. Official scores are the closed loop.

Keep geometric slow and stop trajectories in the pool for every mode. Do not inject a stop trajectory because the light is red. Do not strip brake as an action to make Heuristic look worse.

Headline driving metric is clean completion (finished with no collision, red-light, pedestrian hit, or off-track). OOD recall only counts true sensor corruption. OOD false positives must be zero. Deprecated SDI is not the ranking key.

Mock code is not GPU evidence. A claim about model driving needs a CUDA run written to a new path, with `device: cuda` and no `MockDecisionEngine`.
