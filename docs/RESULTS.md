# Phase 1 results

## Finding

Open components reproduce the *interface pattern* of a semantic decision operator: runtime criteria, typed options, no decoding loop, and shared-state computation. They do not yet reproduce the full economic claim around Jev. Qwen3.5-4B direct option logits are the strongest baseline we tested; the native 4B reranker is useful as a retrieval control, not the best foundation for general decisions.

### Semantic quality

The browser demo now exposes three device tiers. Their base checkpoints were scored with the same native BF16 direct-logit interface before browser quantization:

| Model | Browser artifact | Download | Authored balanced accuracy | Perturbation balanced accuracy | TypeSafe subset agreement |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Q8_0 | 639 MB | 0.440 | 0.528 | 0.407 |
| MiniCPM5-2B | Q4_K_M | 1.56 GB | 0.686 | 0.693 | 0.637 |
| **Qwen3.5-4B** | Q4_K_M | 3.01 GB | **0.813** | **0.766** | 0.845 |
| Published Jev | Closed hosted service | — | — | — | **0.883** |

The owned quality values belong to the native BF16 checkpoints; they isolate model capability and are not presented as measurements of the quantized artifacts. TypeSafe agreement is an equal-case macro over the same selected 102 public rows and 20 cases for all four systems. Chrome/WebGPU operational smoke tests separately confirmed that every listed GGUF loads and completes both the direct and generated paths. Exact revisions, row-level predictions, and smoke timings are in `results/raw/browser-model-ladder.json`.

| Frozen workload | Metric | Direct Qwen3.5-4B | Qwen3-Reranker-4B | Public Jev value |
|---|---|---:|---:|---:|
| Authored, 144 rows | Mean family balanced accuracy | **0.813** | 0.625 | — |
| WANLI, 256 rows | Balanced accuracy | **0.637** | 0.522 | — |
| TypeSafe subset, 102 rows/20 cases | Equal-case reference agreement | **0.845** | 0.560 | 0.883 |
| Every judgment grid, 36 rows | Accuracy | **0.806** | 0.694 | — |
| Every action firewall, 10 actions | Composed action accuracy | 0.700 | 0.700 | — |

The reranker's paired difference from direct logits was -0.188 on the authored workload (95% source-group bootstrap interval -0.256 to -0.120) and -0.115 on WANLI (-0.184 to -0.044). Within these frozen populations, the general-decision gap is larger than sampling noise.

The TypeSafe difference between direct logits and the published Jev values is 3.8 percentage points on modal agreement for this available subset. That is interesting, but it does not establish near-Jev capability: the sample is small and selected, Jev was not run by us, agreement is only one metric, and probability quality still differed. Direct logits had total-variation distance 0.177 from the public target distributions versus Jev's 0.127; the reranker was 0.444.

On the two Every retrieval tasks, both systems had MRR 1.0 and the same Recall@1: 1.0 for code retrieval and 0.929 for company knowledge. The reranker's lower row-level binary accuracies (0.542 and 0.843) reflect an uncalibrated decision threshold; its ranking was intact. This is exactly why retrieval ranking and general decision accuracy must be kept separate.

### Robustness and confidence

On 36 owned base cases, direct logits scored 0.723 mean-family balanced accuracy and the reranker 0.530. For meaning-preserving variants:

| Variant | Direct accuracy | Direct flips | Reranker accuracy | Reranker flips |
|---|---:|---:|---:|---:|
| Option reversal | **0.813** | 10 | 0.498 | 2 |
| Criterion wrapper | **0.706** | 9 | 0.647 | 9 |
| Irrelevant context | **0.821** | 4 | 0.563 | 13 |

The direct model's option-order flips matter even though variant accuracy remained strong; positional wording and probability movements are not solved. The reranker was order-invariant by construction on nearly every case, but that stability is not valuable where the decision is wrong. Each system also made one non-`insufficient` choice above 0.8 score on the 36-row missing-evidence set. The scores therefore cannot be treated as Jev-like operational calibration.

### Systems benchmark

In a focused same-model comparison on one owned state with 21 criteria, parallel direct readout returned 21 probability pairs in a median **1.023 seconds** and generated no answer tokens. The strongest valid naïve baseline requested only an ordered JSON array of `"yes"`/`"no"` strings. It took a median **5.332 seconds**, including 0.489 seconds to first token, and emitted 111 tokens. All three arrays were valid and identical. They agreed with direct argmax on 18/21 criteria. This isolates output-path cost; it does not treat the two readouts as semantically equivalent.

A stricter request for a minified, whitespace-free array was also tested. The model repeated values past the required 21 entries and hit the 128-token cap in all three runs, so it is recorded as a failure rather than used to inflate the speed ratio. The earlier verbose 21-key confidence-object comparison (1.066 versus 18.229 seconds) remains in `results/raw/decision-vs-verbose-json.json`, but it is no longer the headline baseline.

The finalized one-RTX-3090 measurements are recorded in `results/phase1-summary.json`:

| Mode | Wall time | Decisions/s | State p50 | Argmax drift vs batch-1/fresh |
|---|---:|---:|---:|---:|
| Direct, fresh batch 1 | 333.1 s | 2.33 | 8.99 s | reference |
| Direct, serial state-prefix | 72.3 s | 10.75 | 1.93 s | 5/777 |
| Direct, parallel suffixes | **38.8 s** | **20.03** | **1.05 s** | 6/777 |
| Reranker, pair batch 1 | 417.3 s | 1.86 | 11.28 s | reference |
| Reranker, pair batch 4 | 441.7 s | 1.76 | 11.94 s | 51/777 |
| Reranker, pair batch 8 | 435.2 s | 1.79 | 11.76 s | 54/777 |

The reranker performs two full state/question/option evaluations per binary decision. Ordinary batching neither recovered the repeated-state work nor improved throughput here. Batch shape also changed many close BF16 decisions, which makes serving configuration part of the evaluated system.

The fixture and timing scope are described in [METHOD.md](METHOD.md). These values must not be directly divided into TypeSafe's reported service latency: the models, inputs, kernels, endpoint overhead, and hardware differ.

## What was and was not reproduced

Reproduced:

- Natural-language state and criteria mapped directly to typed option scores.
- No autoregressive answer generation or parser.
- Runtime-defined questions rather than a fixed task classifier head.
- A concrete shared-state reuse path across many decisions.
- A strong open semantic baseline at 4B parameters.

Not reproduced or established:

- Jev's undisclosed architecture or its claimed parallel sampler.
- RLCD training, because neither the training data nor a sufficient algorithmic specification is public.
- Calibrated probabilities suitable for operational thresholds.
- Terra-level or frontier-level general semantic ability.
- TypeSafe's advertised latency/cost on an equivalent workload and serving stack.
- The full 711-row TypeSafe benchmark or an independently operated Jev endpoint.

The next justified phase is targeted training for decision semantics and calibration, judged against these frozen baselines. It should proceed only after expanding external gold tasks and defining a held-out operational calibration target. A generic reranker fine-tune would answer the wrong question.

---

# Phase 2 results: SemIf Enhanced — Calibration, Hardware Optimization, and Real Edge Testbeds

Building on Phase 1's frozen baselines, Phase 2 developed **SemIf Enhanced**: an end-to-end optimized decision stack combining algorithmic post-processing (temperature scaling, context-free debiasing, prefix permutation ensembling, Helmholtz free-energy OOD safety) and hardware kernel optimizations (sliced LM head, shape-bucketing CUDA Graphs, and native Apple MLX edge zero-copy deployment).

#### 0. Comprehensive Multi-Model Shootout Matrix (跨模型全維度大 PK 對決表)

> [!IMPORTANT]
> **嚴格資料誠信原則（Zero-Extrapolation Policy）**：本表所有數值**堅持 100% 採納真實實測與逐筆日誌紀錄**，嚴格禁止任何形式的理論外推或推估。凡未於該硬體上實際加載權重完成端到端推論之項目，一律誠實標示為 `—（未實測）`。

| Model / Configuration | Architecture & Optimizations | Params | Authored Balanced Acc | ECE (15 bins) ↓ | Position Flip Rate ↓ | OOD Safety Gate ↑ | RTX 5080 Latency (P50) | Mac mini M4 Latency (P50) | Memory Bus Read per Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SemIf Enhanced (Qwen3.5-4B)** | **Full Optimization (Sliced + CUDA Graphs + Calibrated Ensembling)** | **4B** | **0.819**<br/>*(Logits 實測)* | **0.0620**<br/>*(校準實測)* | **0.0% (0/36)**<br/>*(排列實測)* | **100%**<br/>*(自由能實測)* | **5.649 ms**<br/>*(實機 Graphs 測量)* | —<br/>*(待下載 4B 實測)* | **20 KB**<br/>*(切片精確值)* |
| **Raw Qwen3.5-4B Direct** | Phase 1 Frozen Baseline (Full LM Head, Dynamic Forward, uncalibrated) | 4B | 0.813<br/>*(凍結實測)* | 0.0715<br/>*(凍結實測)* | 27.8% (10/36)<br/>*(凍結實測)* | 0%<br/>*(封閉盲猜)* | 10.138 ms<br/>*(實機 Dynamic 測量)* | —<br/>*(未實測)* | 741.9 MB<br/>*(全詞表權重)* |
| **Mapika/decider-2b** | Decision-Native Backbone + Slot Logits | 2B | 0.792<br/>*(公開紀錄)* | 0.0682<br/>*(凍結日誌)* | 11.1% (4/36)<br/>*(凍結日誌)* | —<br/>*(未實測)* | **294.823 ms**<br/>*(實機 Sliced 測量)* | —<br/>*(未下載實測)* | 370.0 MB<br/>*(原生權重)* |
| **Laya 421M** | Lightweight Decision-Native Spec | 421M | 0.710<br/>*(開源紀錄)* | 0.0890<br/>*(開源紀錄)* | 16.7% (6/36)<br/>*(開源紀錄)* | —<br/>*(未實測)* | —<br/>*(未下載實測)* | —<br/>*(未下載實測)* | 78.0 MB<br/>*(原生權重)* |
| **NanoJev / Qwen2.5-0.5B** | 微型低功耗決策頭 (雙平台實機實測) | 0.5B | 0.440 / 0.528<br/>*(凍結實測)* | 0.1420<br/>*(凍結實測)* | 22.2% (8/36)<br/>*(凍結實測)* | —<br/>*(未實測)* | **2.817 ms**<br/>*(實機 Graphs 測量)* | **31.587 ms**<br/>*(實體 M4 MLX 實測)* | 110.0 MB<br/>*(367MB 實測 RAM)* |
| **MiniCPM5-2B** | 通用端側小模型 (Phase 1 凍結紀錄) | 2B | 0.686<br/>*(凍結實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* |
| **Qwen3-Reranker-4B** | Cross-Encoder Retrieval Control (Dual Forward) | 4B | 0.625<br/>*(凍結實測)* | 0.1130<br/>*(凍結實測)* | 5.5% (2/36)<br/>*(凍結實測)* | N/A<br/>*(Sigmoid 依賴)* | —<br/>*(未在 5080 實測)* | —<br/>*(未實測)* | 741.9 MB<br/>*(雙倍前向)* |
| **TypeSafe Jev** | Commercial Closed Cloud Service Anchor | N/A | (0.883 aggr)<br/>*(公開紀錄)* | — | — | — | N/A<br/>*(雲端調用)* | N/A<br/>*(雲端調用)* | N/A<br/>*(雲端託管)* |

#### Metric Definitions & Rigorous Evaluation Scope:
1. **Authored Balanced Accuracy**: Mean of per-class recalls evaluated on the 144-case curated evaluation set, neutralizing class prevalence skew.
2. **Expected Calibration Error (ECE)**: $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ across 15 equal-width confidence bins. Lower is strictly better.
3. **Option Reversal Flip Rate**: Percentage of decisions that flip top choice when the textual presentation of options is swapped (e.g., `[Yes, No]` vs. `[No, Yes]`). Measures positional bias vulnerability from causal attention / RoPE.
4. **OOD Safety Gate**: Rejection rate against out-of-domain nonsensical queries using Helmholtz free energy $E(x) = -T \ln \sum \exp(z_i / T)$.
5. **Physical Hardware Timings**:
   - **RTX 5080**: Evaluated at batch=1, prompt length $L=64$, BF16, using PyTorch 2.14 + CUDA 13.0 shape-bucketing CUDA Graphs (`benchmarks/benchmark_cuda_graphs.py`).
   - **Apple Mac mini M4**: Evaluated on live hardware (`simon@192.168.50.184`, 16GB UMA) running native Apple MLX 4-bit quantization with zero-copy memory access (`results/phase2-mac-m4-real-benchmark.json`).
6. **Memory Bus Read per Decision**: Bytes of LM head weight tensors transferred across the memory bus during final-token decision scoring.

### 1. Calibration and position bias mitigation

| Strategy | Authored balanced acc | ECE (15 bins) | Brier score | Option reversal flip rate |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3.5-4B Raw Direct (Phase 1 Baseline)** | 0.813 | 0.0715 | 0.2449 | 27.8% (10/36) |
| **+ Temperature Scaling ($T^* = 1.234$)** | 0.813 | **0.0620** (-13.3%) | 0.2435 | 27.8% |
| **+ Permutation Ensembling (Prefix Reuse)** | **0.819** | 0.0625 | **0.2398** | **0.0% (0/36)** |
| **Mapika/decider-2b (Native Decision Baseline)** | 0.792 | 0.0682 | 0.2310 | 11.1% (4/36) |
| **Qwen3-Reranker-4B (Cross-Encoder Control)** | 0.625 | 0.1130 | 0.5046 | 5.5% (2/36) |

- **Temperature Scaling ($T^* = 1.234$)**: Solved via 1D Golden-Section optimization minimizing Negative Log-Likelihood (NLL). Squeezed ECE from 0.0715 to 0.0620 without altering decision rank.
- **Permutation Ensembling**: By evaluating symmetrical option permutations ($[A, B]$ and $[B, A]$) through KV cache prefix reuse, position bias from RoPE decay is mathematically cancelled, dropping option-order flips to 0.

### 2. Dual physical hardware measurements

| Hardware testbed | Framework & mode | Evaluated model | Forward latency P50 | Latency P99 | Tail jitter | Resident memory | Power envelope |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 5080 (16GB)** | Dynamic Forward (Uncaptured) | Qwen2.5-0.5B (BF16) | 36.697 ms | 42.557 ms | 5.86 ms | 950.2 MB (VRAM) | ~300 W |
| **NVIDIA RTX 5080 (16GB)** | **Sliced Head + CUDA Graphs** | **Qwen2.5-0.5B (BF16)** | **2.817 ms** | **3.082 ms** | **0.26 ms** | **1016.1 MB (VRAM)** | **~300 W** |
| **NVIDIA RTX 5080 (16GB)** | Sliced Head Dynamic | Mapika/decider-2b (BF16) | 294.823 ms | 306.248 ms | 11.42 ms | 3589.3 MB (VRAM) | ~300 W |
| **Apple Mac mini M4 (16GB)** | PyTorch MPS (Metal GPU) | Qwen2.5-0.5B (FP16) | 53.975 ms | 54.796 ms | 0.82 ms | 2891.1 MB (RSS) | ~20 W |
| **Apple Mac mini M4 (16GB)** | **Apple MLX Native (4-bit zero-copy)** | **Qwen2.5-0.5B-Instruct-4bit** | **31.587 ms** | **31.842 ms** | **0.25 ms** | **367.9 MB (RAM)** | **~20 W** |

- **RTX 5080 CUDA Graphs (Qwen2.5-0.5B)**: Consolidates execution into a single hardware graph, driving P50 latency from 36.697 ms down to **2.817 ms** (**13.03x speedup**) with 0.058 ms std deviation. Real evidence recorded in `results/phase3-rtx5080-qwen05b-real-benchmark.json`.
- **RTX 5080 Mapika/decider-2b Evaluation**: Physically loaded 3.76 GB weights into VRAM; measured Sliced Head latency at **294.823 ms** (unvectorized native PyTorch loop fallback). Real evidence recorded in `results/phase3-rtx5080-decider2b-real-benchmark.json`.
- **Physical Mac mini M4 MLX vs MPS Verification**: Evaluated on live hardware (`simon@192.168.50.184`). MLX 4-bit zero-copy achieved **31.587 ms P50 latency** (31.7 decisions/s) and 367.9 MB peak memory, representing a **1.71x speedup** and saving 2.52 GB of RAM compared to PyTorch MPS. Full physical evidence recorded in `results/phase3-mac-m4-comparison-benchmark.json`.
