# Repository instructions

- Run commands from the repository root in an isolated environment installed with `pip install -e '.[test]'`.
- Validate changes with `pytest -q`, `(cd results/raw && sha256sum -c SHA256SUMS)`, and `python benchmarks/verify_published.py`.
- Benchmark outputs are create-only. Use a new output path and expose exactly one CUDA GPU per scorer process.
- Do not change headline claims or `results/phase1-summary.json` without committing the supporting row-level evidence, regenerating the relevant raw report, updating `results/raw/SHA256SUMS`, and updating the method/results text.
- Preserve exact model and source revisions. Use `benchmarks/fetch_sources.py` only for its listed redistributable inputs; do not commit model weights, caches, or third-party raw records.
- `webgpu-demo/` is static and has no build step. Preserve `_headers`, runtime version pins, browser-only inference, and the explicit probability limitations.

# How to write here

Lead with the answer. Then say why, in short sentences. A longer sentence is fine after the point is already clear. Define a project term the first time it appears. Do not stack labels, tables, or file dumps in front of the conclusion. Do not dress a guess as a measured result.

# What this project is for

This repository exists to remove failure modes of a Jev-style decision model on the path a product calls. That path is `DecisionEngine.classify_jev` and `DecisionEngine._score_neural_options` behind `/v1/classifier` and `/v1/systemone`. An application sends this step's evidence and option ids. The path returns a choice aligned by option id. A failure is fixed only when a test calls that live function, fails while the failure is still present, and the observable decision changes: argmax aligned by option id, or abstain.

A CLI flag, a tutorial, or a row in `docs/RESULTS.md` does not close the failure. Describe option order, calibration, confidence, abstention, or invariance as solved only when that live path runs the change at its default settings. Rewrite a claim that says otherwise. A benchmark that cannot change a live decision is not a deliverable.

# Applications

Applications live in their own repositories and call the door over HTTP. The driving app is [JevPilot-Vision](https://github.com/EndeavorYen/JevPilot-Vision). It owns its world, camera, SigLIP, trajectory sampling, and any veto applied after the choice. Its closed loop produces the official driving scores.

This repository does not import an application, accept pixels on the door, or change a choice with an application's physics. A fix here must not assume one application's option layout, such as six-column trajectory vectors.

`results/phase5-jevpilot-*.json` is the driving record from before the split. Keep those files as they are. Do not add new driving scores here.

# Evaluation honesty

Mock code is not model evidence. The mock engine returns the first option. A claim about a model needs a CUDA run, or a run on the named Apple hardware, written to a new path with the device recorded and no mock engine.

# How to change code

Write the failing test first. Watch it fail for the reason you care about. Then write the minimum production code. If you did not see the test fail, it does not test the bug.

The test must call the live branch: the function the runtime actually invokes. A helper that an early return never reaches is dead. Delete it, or move the work onto the live path. Name a function for what it still does.

A string grep (`"foo" in file`) is only for hooks and cache pins. It is not proof a branch executes. Duplicate constants across languages must be locked by a test that reads both sources.

Do not claim a fix from reading the file. Run `pytest -q` on the tests that encode the acceptance. A browser claim needs a browser check. A CUDA claim needs a CUDA run to a new path. If the test cannot fail when the bug is present, it is not a test of the bug.
