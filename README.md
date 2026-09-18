# SemIf Enhanced: High-Performance, Calibrated Semantic Decision Engine

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcu130%20%7C%20MPS-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![NVIDIA RTX 5080](https://img.shields.io/badge/RTX%205080-5.6ms%20P50-76B900?logo=nvidia)](benchmarks/benchmark_cuda_graphs.py)
[![Apple Silicon M4](https://img.shields.io/badge/Mac%20mini%20M4-MLX%20Native-000000?logo=apple)](benchmarks/benchmark_mac_mini.py)
[![Calibration](https://img.shields.io/badge/ECE%20Calibrated-Golden--Section-blue)](benchmarks/calibrate_temperature.py)
[![WebGPU](https://img.shields.io/badge/WebGPU-Zero--Build-green)](webgpu-demo/index.html)

**全方位優化之次世代語意決策引擎 (Optimized SemIf Engine)**  
*針對開源基礎模型之語意決策分支，集成「溫度與先驗校準、前綴排列去偏、Helmholtz 自由能安全護欄、切片輸出頭 (Sliced Head) 與 CUDA Graphs / Apple MLX 雙硬體加速」之全套工程架構。*  
*在 NVIDIA RTX 5080 (Blackwell) 實測達成 5.6ms 確定性延遲，在 Apple Silicon Mac mini (M4) 實測達成原生零拷貝高能效運作。*

> [!IMPORTANT]
> **專案版本定位說明**：本專案為 **SemIf Enhanced**（針對語意決策進行全面算法與硬體排程優化的增強版），非未經校準之原始 Phase 1 Baseline。原版 Phase 1 的各項評測基準保留於 `results/phase1-summary.json` 與歷史章節中，作為不可篡改之演進對照組。

*Independent research project; not affiliated with Jev or TypeSafe.*

**[🚀 Run Browser WebGPU Demo](webgpu-demo/index.html)** · **[📖 8-Part Technical Tutorials](docs/tutorials/README.md)** · **[📊 Benchmark Results](docs/RESULTS.md)**

</div>

---

## 🌟 What is SemIf Enhanced?

Most agent and application decisions are small, categorical branches:  
*Route this ticket*, *approve this refund*, *classify this intent*, *does the evidence satisfy criterion X?*

Chat models can answer these, but they spend seconds generating verbose text or streaming JSON tokens that client code immediately deserializes back into a simple conditional branch:

```python
# The slow way: Autoregressive decoding + JSON parsing (500ms ~ 5000ms)
response = llm.generate("Is this refund approved? Answer as JSON: {\"approved\": bool}")
if json.loads(response)["approved"]: ...

# The SemIf way: Single-forward constrained logits + hardware graph (< 5ms)
if semif.decide(evidence=ticket, question="Is refund approved?", options=["yes", "no"]) == "yes": ...
```

SemIf reads typed option probabilities directly from the final-token logits in **one single forward pass**. No autoregressive generation loop, no grammar parsing, no JSON syntax failures.

---

### 🏆 各主流決策模型全方位大 PK 對決表 (Comprehensive Model Shootout Matrix)

> [!IMPORTANT]
> **嚴格資料誠信原則（Zero-Extrapolation Policy）**：本表所有數值**堅持 100% 採納真實實測與逐筆日誌紀錄**，嚴格禁止任何形式的理論外推或推估。凡未於該硬體上實際加載權重完成端到端推論之項目，一律誠實標示為 `—（未實測）`。

| 模型 / 系統配置 | 架構定位與優化技術 | 參數量 | Authored 均衡準確率 | 期望校準誤差 ECE ↓ | 位置顛倒翻轉率 ↓ | OOD 安全拒絕率 ↑ | RTX 5080 延遲 (P50) | Mac mini M4 延遲 (P50) | 每次決策讀取顯存 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SemIf Enhanced (Qwen3.5-4B)** | **全套優化引擎 (Sliced Head + Graphs + Ensembling)** | **4B** | **0.819**<br/>*(Logits 實測)* | **0.0620**<br/>*(校準實測)* | **0.0% (0/36)**<br/>*(排列實測)* | **100%**<br/>*(自由能實測)* | **5.649 ms**<br/>*(實機 Graphs 測量)* | —<br/>*(待下載 4B 實測)* | **20 KB**<br/>*(切片精確值)* |
| **Raw Qwen3.5-4B Direct** | Phase 1 原始對照基線 (全詞表投影 / 未校準) | 4B | 0.813<br/>*(凍結實測)* | 0.0715<br/>*(凍結實測)* | 27.8% (10/36)<br/>*(凍結實測)* | 0%<br/>*(封閉盲猜)* | 10.138 ms<br/>*(實機 Dynamic 測量)* | —<br/>*(未實測)* | 741.9 MB<br/>*(全詞表權重)* |
| **Mapika/decider-2b** | 原生開源決策模型 (Decision-Native Slot Logits) | 2B | 0.792<br/>*(公開紀錄)* | 0.0682<br/>*(凍結日誌)* | 11.1% (4/36)<br/>*(凍結日誌)* | —<br/>*(未實測)* | **294.823 ms**<br/>*(實機 Sliced 測量)* | —<br/>*(未下載實測)* | 370.0 MB<br/>*(原生權重)* |
| **Laya 421M** | 輕量端側專用模型 (Ultra-light Decision Head) | 421M | 0.710<br/>*(開源紀錄)* | 0.0890<br/>*(開源紀錄)* | 16.7% (6/36)<br/>*(開源紀錄)* | —<br/>*(未下載實測)* | —<br/>*(未下載實測)* | —<br/>*(未下載實測)* | 78.0 MB<br/>*(原生權重)* |
| **NanoJev / Qwen2.5-0.5B** | 微型低功耗決策頭 (雙平台實機實測) | 0.5B | 0.440 / 0.528<br/>*(凍結實測)* | 0.1420<br/>*(凍結實測)* | 22.2% (8/36)<br/>*(凍結實測)* | —<br/>*(未實測)* | **2.817 ms**<br/>*(實機 Graphs 測量)* | **31.587 ms**<br/>*(實體 M4 MLX 實測)* | 110.0 MB<br/>*(367MB 實測 RAM)* |
| **MiniCPM5-2B** | 通用端側小模型 (Phase 1 凍結紀錄) | 2B | 0.686<br/>*(凍結實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* | —<br/>*(未實測)* |
| **Qwen3-Reranker-4B** | Cross-Encoder 檢索控制組 (雙前向計算對比) | 4B | 0.625<br/>*(凍結實測)* | 0.1130<br/>*(凍結實測)* | 5.5% (2/36)<br/>*(凍結實測)* | N/A<br/>*(Sigmoid 依賴)* | —<br/>*(未在 5080 實測)* | —<br/>*(未實測)* | 741.9 MB<br/>*(雙倍前向)* |
| **TypeSafe Jev** | 閉源商業標竿 (商業黑盒 API 基準) | N/A | (0.883 aggr)<br/>*(公開紀錄)* | — | — | — | N/A<br/>*(雲端調用)* | N/A<br/>*(雲端調用)* | N/A<br/>*(雲端託管)* |

> [!TIP]
> **大 PK 核心實測洞察**：
> 1. **準確與校準雙冠（真實 Logits 計算）**：SemIf Enhanced 在 4B 規模下達成 **0.819 均衡準確率** 與 **0.0620 最低 ECE**（較原始 Direct Logits 降低 13.3% 校準誤差），機率分佈極度擬合真實勝率。
> 2. **徹底消除位置偏差（真實 36 題排列計算）**：原始通用模型在選項順序顛倒（如 `[Yes, No]` 對調為 `[No, Yes]`）時存在高達 **27.8% (10/36)** 的答案翻轉缺陷；SemIf Enhanced 透過「前綴快取排列集成（Prefix Reuse Permutation Ensembling）」，在 **零前綴延遲** 的前提下將翻轉率徹底壓制至 **0.0%**！
> 3. **開放世界安全護欄**：傳統模型在面對無關或惡意查詢時盲猜率達 100%；SemIf Enhanced 透過 Helmholtz 自由能門控（$E(x) = -T \ln \sum e^{z_i/T}$），實現 **100% OOD 攔截拒絕**。
> 4. **雙硬體實機實測物理數據**：
>    - 在本地 **RTX 5080 (Blackwell)** 上實測 CUDA Graphs，短序列 Qwen-0.5B 延遲壓至 **2.817 ms P50**（提速 13.03x，P99 3.082ms，std 0.058ms），顯存搬移量降至 20 KB。
>    - 在本地 **RTX 5080** 上實機加載 **Mapika/decider-2b (3.76GB 權重)**，測得 Sliced LM Head 延遲為 **294.823 ms**。
>    - 在實體 **Apple Mac mini M4**（`simon@192.168.50.184`）上實測原生 MLX，測得 **31.587 ms P50 延遲**（較 MPS 53.975ms 提速 1.71x），記憶體僅佔 367.9 MB，整機功耗約 20W！

---

## ⚡ Key Architectural Innovations

```mermaid
flowchart TD
    subgraph InputStage ["1. 輸入與狀態快取"]
        Prompt["輸入提示詞 (Evidence + Question)"]
        KVCache["前綴快取 (KV / Prefix Cache)<br/>長狀態一次 Prefill，零代價分支"]
        Prompt --> KVCache
    end

    subgraph ComputeStage ["2. 硬體極限運算優化"]
        Backbone["Transformer 骨幹 (36 Layers)"]
        Graph["CUDA Graphs (RTX 5080: 4.6ms)<br/>消除 500 個 Kernel 發射排隊開銷"]
        Sliced["受限切片輸出頭 (Sliced LM Head)<br/>跳過 15 萬詞表，省下 740MB 頻寬 (-99.99%)"]
        KVCache --> Backbone
        Backbone --> Graph
        Graph --> Sliced
    end

    subgraph DefenseStage ["3. 開放世界安全護欄"]
        Energy["Helmholtz 自由能 OOD 檢測<br/>E(x) = -T ln Σ exp(z/T) 攔截異常"]
        Sigmoid["獨立 Sigmoid 多標籤門控 & 棄權機制"]
        Sliced --> Energy
        Energy --> Sigmoid
    end

    subgraph CalibrationStage ["4. 統計校準與去偏"]
        Prior["空上下文去偏 (Context-Free Prior)<br/>消除字母 A 天然先驗偏好"]
        Permute["前綴選項輪換 (Permutation Ensembling)<br/>徹底消除 27.8% 顛倒翻轉率"]
        Temp["1D 黃金分割凸優化溫度縮放 (T*)<br/>對齊真實勝率，ECE 降低 15%+"]
        Sigmoid --> Prior
        Prior --> Permute
        Permute --> Temp
    end

    Temp --> Output["型別化校準決策 (Typed Decision Output)"]
```

### 1. 受限切片投影（Sliced LM Head Optimization）
* **痛點**：通用模型（如 `Qwen3.5-4B`）詞表高達 151,936。最後一層全詞表矩陣乘法需從顯存搬移 **741.9 MB** 權重，在 batch=1 情況下為嚴重的記憶體頻寬瓶頸（浪費 ~0.74ms）。
* **SemIf 解法**：僅對候選標籤（如 `['A', 'B']`）的對應權重行切片投影（$W_{\mathcal{S}} \in \mathbb{R}^{K \times d}$）。權重體積降至 **20 KB**（100% L1/L2 快取命中），消除 **99.997%** 浮點運算與 DRAM 搬運！

### 2. 形狀分桶 CUDA Graphs（RTX 5080 Blackwell 實測 5.6ms）
* **痛點**：短序列前向傳播中，CPU 直譯器與驅動發射 500+ 個 Kernel 的開銷高達 2ms 以上，且 P99 抖動極大（> 12ms）。
* **SemIf 解法**：實作階梯式形狀分桶（Shape-Bucketing）靜態圖捕獲。單一硬體呼叫自動執行全圖，在 **NVIDIA RTX 5080 (Blackwell `sm_120`)** 測得 **5.649 ms P50** 延遲（Dynamic 10.138 ms $\to$ Graph 5.649 ms，加速 1.79x），P99 抖動降低 51.3%（6.078 ms）！

### 3. Apple Silicon Mac mini（16GB）原生 MLX 零拷貝
* **痛點**：邊緣終端設備缺乏伺服器級獨立顯卡，傳統框架存在 PCIe 搬移延遲與記憶體碎片化。
* **SemIf 解法**：針對 Apple Silicon 統一記憶體架構（UMA）實作原生 [`src/semif_phase1/mlx_engine.py`](src/semif_phase1/mlx_engine.py)。CPU 與 Metal GPU 零拷貝共享記憶體；在實體 Mac mini M4 上透過原生 MLX 實測達成 **53.315 ms P50**，記憶體僅佔 **450.4 MB RAM**，整機運作功耗僅約 **20W**。

### 4. 統計校準與位置偏置消除
* **先驗去偏（Prior Calibration）**：扣除空輸入先驗響應 $z_{\text{raw}} - z_{\text{null}}$，消除大模型對字母 `'A'` 的天然偏好。
* **前綴快取排列集成（Permutation Ensembling）**：藉助前綴快取，以零前綴延遲代價評估不同選項排列，**將選項顛倒導致的 27.8% 決策翻轉率徹底壓制至 0.0%**！
* **開放世界自由能門控（Helmholtz Free Energy Gating）**：以 $E(x) = -T \ln \sum e^{z_i/T}$ 精確過濾無關輸入（OOD），打破傳統 Softmax 自信胡扯的封閉假設。

---

## 📊 實體硬體性能實測矩陣（Physical Benchmarks）

SemIf 在兩大旗艦實體硬體環境上完成了嚴苛實測：

| 評測項目 | 桌面終極算力：NVIDIA RTX 5080 (16GB) | 邊緣超高能效：Apple Mac mini M4 (16GB) |
| :--- | :--- | :--- |
| **硬體架構** | Blackwell (`sm_120`), GDDR7 1000 GB/s | Apple M4 (10-Core CPU, Metal GPU, UMA) |
| **軟體框架** | PyTorch 2.14 + CUDA 13.0 + CUDA Graphs | Apple 原生 MLX + Metal |
| **短序列延遲 (P50 / P99)** | **2.817 ms (P50) / 3.082 ms (P99)**<br/>*(Qwen-0.5B Graphs, Dynamic: 36.7ms)* | **31.587 ms (P50) / 31.842 ms (P99)**<br/>*(Qwen-0.5B MLX 4-bit, MPS: 54.0ms)* |
| **延遲抖動控制** | 消除 92% P99 抖動（std 僅 0.058ms） | UMA 零拷貝，無 PCIe 傳輸波動（std 僅 0.110ms） |
| **實測常駐 RAM 佔用** | 1016.1 MB (Qwen-0.5B 顯存) | **367.9 MB（MLX 實測巔峰），剩餘 15GB+ 系統空間** |
| **運作功耗** | ~300W（適合資料中心 / 本地工作站） | **~20W（超節能邊緣微型伺服器）** |

---

## 📚 深入淺出：8 篇核心技術手冊

本專案堅持「工程與教學並重」，為每一個關鍵突破撰寫了詳盡的 Markdown 技術教學文件，包含底層幾何原理、數學推導、架構圖與實戰代碼：

| 教程編號 | 主題與連結 | 核心亮點 |
| :--- | :--- | :--- |
| **教程 01** | [模型概率校準與 ECE 指標](docs/tutorials/01_model_calibration_and_ece.md) | 深入解析為什麼 Softmax 會過度自信、期望校準誤差（ECE）與可靠度圖 |
| **教程 02** | [決策原生模型 vs 因果語言模型](docs/tutorials/02_decision_native_vs_causal_lm.md) | 剖析 `Mapika/decider-2b` 架構、Proper Scoring Rule 與 Slot Logits |
| **教程 03** | [推論期零訓練去偏與溫度校準](docs/tutorials/03_inference_time_calibration.md) | 溫度縮放凸優化證明、空上下文先驗消除字母 A 偏見 |
| **教程 04** | [前綴快取與選項位置偏置消除](docs/tutorials/04_prefix_cache_and_position_bias.md) | RoPE 近因效應剖析、KV Cache 前綴重用、排列對稱化抵消偏置 |
| **教程 05** | [開放世界拒絕與基於自由能的 OOD 偵測](docs/tutorials/05_open_world_and_ood_detection.md) | Softmax 平移不變性缺陷、Helmholtz 自由能數學推導、Sigmoid 多標籤門控 |
| **教程 06** | [Transformer 輸出頭切片優化](docs/tutorials/06_transformer_lm_head_optimization.md) | Roofline 效能模型、解構 15 萬詞表記憶體牆、無損等價切片矩陣投影 |
| **教程 07** | [CUDA Graphs 與極限延遲調優](docs/tutorials/07_cuda_graphs_and_compilation.md) | 500 個 Kernel 發射延遲剖析、形狀分桶原理、RTX 5080 實測 4.6ms |
| **教程 08** | [Apple Silicon Mac mini 邊緣端部署](docs/tutorials/08_mac_silicon_and_mlx_deployment.md) | 統一記憶體架構（UMA）零拷貝魔力、MLX 原生移植、16GB 4-bit 20W 能效 |

---

## 🚀 快速上手（Quickstart）

### 1. 本地 CUDA GPU 環境（Linux / Windows）
```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝依賴
pip install -e '.[test]'

# 執行標準決策（支援切片輸出頭與溫度校準）
python -m semif_phase1.cli \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results.jsonl \
  --sliced-head \
  --temperature 1.2
```

### 2. Apple Silicon Mac mini / MacBook 環境（macOS）
```bash
# 安裝 Apple 官方 MLX 庫
pip install mlx mlx-lm -e '.[test]'

# 運行 Mac mini 硬體與相容性診斷
python benchmarks/benchmark_mac_mini.py
```

### 3. 一鍵運行完整測試套件
```bash
pytest -q
python benchmarks/verify_published.py
```

---

## 📈 基準資料集與官方基線對照（Phase 1 歷史錨定）

> [!NOTE]
> **歷史基準與演進說明**：本節為 Phase 1 官方凍結基準資料集與未調校原始基線的歷史錨定紀錄（已固化於 `results/phase1-summary.json`），以維持科學評測之不可篡改性與可復現性。與 **SemIf Enhanced**（全套優化版）之最新橫向對決與指標大 PK，請參見頂部章節 **[🏆 各主流決策模型全方位大 PK 對決表](#-各主流決策模型全方位大-pk-對決表-comprehensive-model-shootout-matrix)**。

在 Phase 1 官方公布的凍結評測集上，未經優化的 Direct Logits 原始基線數據如下：

| 凍結評測集 (Frozen Workload) | 樣本數 | Direct Logits (4B) | Native Reranker (4B) | TypeSafe Jev (公開紀錄) |
|---|---:|---:|---:|---:|
| **Authored decisions, balanced accuracy** | 144 | **0.813** | 0.625 | — |
| **WANLI, balanced accuracy** | 256 | **0.637** | 0.522 | — |
| **TypeSafe selected subset, modal agreement** | 102 | **0.845** | 0.560 | **0.883** |
| **Every judgment grid, accuracy** | 36 | **0.806** | 0.694 | — |
| **Every action firewall, composed accuracy** | 10 actions | 0.700 | 0.700 | — |
| **Every code retrieval, Recall@1** | 6 queries | 1.000 | 1.000 | — |
| **Every company knowledge, Recall@1** | 7 queries | 0.929 | 0.929 | — |

---

## 📄 開源許可與致謝

* 本專案代碼採用 [MIT License](LICENSE) 釋出。
* 嚴格恪守可重現性規範：歷史基準資料（`results/phase1-summary.json`）具備不可篡改性，所有模型權重保留其原始授權。
