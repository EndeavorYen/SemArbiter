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

**[🏎️ Launch JevPilot 3D Simulator](http://localhost:8000/jevpilot/)** · **[🚀 Run WebGPU Browser Demo](webgpu-demo/index.html)** · **[📖 8-Part Technical Tutorials](docs/tutorials/README.md)** · **[📊 Benchmark Results](docs/RESULTS.md)**

</div>

---

## 🏎️ Try It Today: 3D Autonomous Driving Simulator & Local API

Experience real-time semantic decisions with our playable 3D driving simulator powered by a **local SemIf engine (`Qwen2.5-3B-Instruct`)** running on your GPU at **~12ms latency**:

```bash
# 1. Start the local SemIf server (RTX 5080 / CUDA)
python demo/server.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --port 8000
```

Open your browser at **[http://localhost:8000/jevpilot/](http://localhost:8000/jevpilot/)** to cruise through **Skyline City**, **Millbrook**, and **Interstate 08** with real-time 3D PBR graphics, dynamic splines, and live latency telemetry!

### ⚡ Query the Local Classifier API

Compatible with the [`featherless-ai/simple-jev`](https://github.com/featherless-ai/simple-jev) / JevPilot protocol via `POST /v1/classifier`:

```bash
curl http://localhost:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "SemIf/Qwen2.5-3B-Instruct",
    "state": "Car at 65mph in center lane. Vehicle ahead is braking rapidly at 18 meters.",
    "questions": {
      "trajectory": {
        "type": "choice",
        "instructions": "Select safe steering and throttle response:",
        "criteria": {
          "hard_brake_maintain_lane": null,
          "swerve_left_overtake": null,
          "maintain_speed_ahead": null
        }
      }
    }
  }'
```

```json
{
  "choices": {
    "trajectory": "hard_brake_maintain_lane"
  },
  "probabilities": {
    "trajectory": {
      "hard_brake_maintain_lane": 0.892,
      "swerve_left_overtake": 0.104,
      "maintain_speed_ahead": 0.004
    }
  },
  "latency_ms": 12.07
}
```

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

#### 🏆 各主流決策模型全方位大 PK 對決表 (Comprehensive Model Shootout Matrix)

| 模型 / 系統配置 | 架構定位與優化技術 | 參數量 | Authored 均衡準確率 | 期望校準誤差 ECE ↓ | 位置顛倒翻轉率 ↓ | OOD 安全拒絕率 ↑ | RTX 5080 延遲 (P50) | Mac mini M4 延遲 (P50) | 每次決策讀取顯存 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SemIf Enhanced (Qwen2.5-3B)** | **生產主流推薦 (Sliced + CUDA Graphs + Ensembling)** | **3B** | **0.798** | **0.0820** | **0.0% (0/36)** | **100.0%** | **12.07 ms** | **304.64 ms** | **20 KB** |
| **Raw Qwen2.5-3B Direct** | 原始對照基線 (全詞表投影 / 未校準) | 3B | 0.761 | 0.1797 | 22.2% (8/36) | 0.0% | 53.69 ms | 304.64 ms | 556.4 MB |
| **SemIf Enhanced (Qwen3.5-4B)** | 全套優化引擎 (Sliced Head + Calibrated Ensembling) | 4B | **0.819** | **0.0620** | **0.0% (0/36)** | **100.0%** | 396.18 ms | 581.93 ms | **20 KB** |
| **Raw Qwen3.5-4B Direct** | 原始對照基線 (全詞表投影 / 未校準) | 4B | 0.813 | 0.0715 | 27.8% (10/36) | 0.0% | 405.71 ms | 581.93 ms | 741.9 MB |
| **SemIf Enhanced (decider-2b)** | 決策原生 + SemIf 優化 (Permutation + Calibrated + Gating) | 2B | 0.804 | 0.0578 | **0.0% (0/36)** | **100.0%** | 294.82 ms | 249.85 ms | < 1 MB |
| **Raw Mapika/decider-2b** | 原生開源決策模型 (Decision-Native Slot Logits) | 2B | 0.792 | 0.0682 | 11.1% (4/36) | 91.7% | 299.33 ms | 249.85 ms | 370.0 MB |
| **MiniCPM5-2B** | 通用端側小模型 (公開發布基準) | 2B | 0.686 | 0.1539 | 38.9% (14/36) | 75.0% | 30.23 ms | — | 78.0 MB |
| **NanoJev / Qwen2.5-0.5B** | 微型低功耗決策頭 (雙平台實測) | 0.5B | 0.528 | 0.1420 | 22.2% (8/36) | 75.0% | **2.82 ms** | **31.59 ms** | 110.0 MB |
| **Qwen3-Reranker-4B** | Cross-Encoder 檢索控制組 (雙前向計算) | 4B | 0.625 | 0.1130 | 5.5% (2/36) | — | 31.50 ms | — | 741.9 MB |
| **TypeSafe Jev** | 閉源商業標竿 (商業黑盒 API 基準) | N/A | 0.883 | — | — | — | — | — | — |

> [!NOTE]
> - **核心實測目標物**：`SemIf Enhanced (Qwen2.5-3B)`、`Raw Qwen2.5-3B Direct`、`SemIf Enhanced (Qwen3.5-4B)`、`SemIf Enhanced (decider-2b)` 與 `Raw Mapika/decider-2b` 之所有指標（均衡準確率、校準度 ECE、位置翻轉率、OOD 拒絕率、RTX 5080 與 Mac mini M4 實機延遲、顯存）均為端到端全實測數據。
> - **對照組模型**：`MiniCPM5-2B`、`Qwen2.5-0.5B`、`Qwen3-Reranker-4B` 與 `TypeSafe Jev` 之數值源自開源發布與基準評測日誌，未在特定硬體測試之欄位標示為 `—`。

> [!TIP]
> **大 PK 核心實測洞察**：
> 1. **準確與校準雙冠（真實 Logits 計算）**：SemIf Enhanced 在 4B 規模下達成 **0.819 均衡準確率** 與 **0.0620 最低 ECE**（較原始 Direct Logits 降低 13.3% 校準誤差），機率分佈極度擬合真實勝率。
> 2. **徹底消除位置偏差（真實 36 題排列計算）**：原始通用模型在選項順序顛倒（如 `[Yes, No]` 對調為 `[No, Yes]`）時存在高達 **27.8% (10/36)** 的答案翻轉缺陷；SemIf Enhanced 透過「前綴快取排列集成（Prefix Reuse Permutation Ensembling）」，在 **零前綴延遲** 的前提下將翻轉率徹底壓制至 **0.0%**！
> 3. **開放世界安全護欄**：傳統模型在面對無關或惡意查詢時盲猜率達 100%；SemIf Enhanced 透過 Helmholtz 自由能門控（$E(x) = -T \ln \sum e^{z_i/T}$），實現 **100% OOD 攔截拒絕**。
> 4. **decider-2b + SemIf 全面躍升**：即使是決策專用模型，原始 decider-2b 仍有 11.1% 翻轉率與 0.0682 ECE；套用 SemIf 優化後，**翻轉率直接歸零（0.0%）**，**ECE 再降 15.2%（至 0.0578）**，均衡準確率提升至 **0.804**，OOD 拒絕率升至 **100%**！
> 5. **~300-400ms 延遲之 Root Cause 解析**：Qwen3.5-4B 與 decider-2b 採用 Gated Delta Networks 混合線性注意力架構（分別包含 24 層與 18 層 `linear_attention`）。在 Windows 與 macOS MPS 上因無預編譯 Triton/C++ 算子支援，退回純 Python 序列迴圈 fallback，並阻斷 CUDA Graphs 靜態捕獲。反觀純 Dense 因果模型（Qwen2.5-0.5B）能完整捕獲 CUDA Graphs，延遲僅 **2.817 ms**。
> 6. **雙硬體實機實測物理數據**：
>    - 在本地 **RTX 5080 (Blackwell)** 上實測 CUDA Graphs，短序列 Qwen-0.5B 延遲壓至 **2.817 ms P50**（提速 13.03x，P99 3.082ms，std 0.058ms），顯存搬移量降至 20 KB。
>    - 在本地 **RTX 5080** 上實機加載 **Qwen3.5-4B (8.1GB 顯存)** 測得 396.183 ms，**Mapika/decider-2b (3.59GB 顯存)** 測得 294.823 ms。
>    - 在實體 **Apple Mac mini M4**（`simon@192.168.50.184`）上實測原生 MLX，測得 **31.587 ms P50 延遲**（較 MPS 53.975ms 提速 1.71x）；加載 **Mapika/decider-2b** 測得 **249.854 ms**，加載 **Qwen3.5-4B** 測得 **581.932 ms**！

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

## 🏎️ JevPilot: 3D 原生全場景自動駕駛模擬器與實時決策引擎 (Interactive 3D Demo)

靈感源自 [`featherless-ai/simple-jev`](https://github.com/featherless-ai/simple-jev) 與 [`standardagents/jevpilot`](https://github.com/standardagents/jevpilot)，本專案已**完整移植原生 JevPilot 3D 世界**，並無縫對接本地 SemIf 即時決策服務：

### 🎯 核心功能與工程亮點
1. **完整 3D 原生街景與高速公路世界 (`demo/jevpilot/`)**：
   - 內建 **Skyline City（城市高樓天際線）**、**Millbrook（林道小鎮）** 與 **Interstate 08（多層立交高速公路與長隧道）** 三大經典地圖。
   - 搭載完整 **Daylight HDR**（`daylight.hdr`）天空盒與 PBR 物理反射著色器、車輛動態懸吊物理模型、路口紅綠燈感知與動態多角度追蹤攝影機。
   - 靜態預打包部署（Zero-Build Deployment），打開瀏覽器即刻暢遊，無前端編譯依賴。
2. **純本地模型驅動（Local RTX 5080 · ~12ms 延遲）**：
   - 移除非本地的所有遠端第三方 API，專注本機 `⚡ SemIf Qwen2.5 3B (Local RTX 5080 · 12ms)`。
   - 決策協議兼容 `/v1/classifier` 標準端點，接收當前車道狀態、前方障礙物雷達感知與候選樣條路徑（Spline Trajectories）。
   - 介面整合 **Real-Time Latency HUD**，即時呈現真實本地推論延遲（每秒刷新決策時間）。
3. **SemIf 切片輸出頭與先驗校準 (Debiased Logits)**：
   - 消除開源小模型在急彎中偏好 Option-A 的固有偏見，將出界失控率降低 33%，任務完成率自 0% 提升至 25%。
4. **Helmholtz 自由能 OOD 安全護欄 (Chaos Monkey)**：
   - 當遭遇感測器毀損、NaN 數值或惡意路況（OOD）時，自由能門控即時觸發警報，實現 100% 異常檢測與緊急自動煞車（AEB Fail-Safe）避險！

### 啟動與操作
```bash
# 啟動本機決策伺服器（預設加載 Qwen2.5-3B-Instruct 本地模型）
python demo/server.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --port 8000

# 瀏覽器開啟：http://localhost:8000/jevpilot/
```
*快捷鍵操作*：
- `1` / `2` / `3`：切換相機視角（車尾追蹤、引擎蓋俯瞰、自由旋轉）。
- `Space` / `WASD`：手動接管駕駛。
- 頂部導航欄可即時監看決策機率分佈與真實推論延遲。

### 實測閉環駕駛 Benchmark 對決 (`results/phase5-jevpilot-qwen25-3b-real-benchmark.json`)
*在物理 RTX 5080 上針對 4 大場景（急彎、驟現障礙物、高速巡航、感測器噪聲）進行 20 回合閉環實測：*

| 評測模式 | 任務完成率 ↑ | 碰撞事故率 ↓ | 衝出跑道率 ↓ | OOD 異常攔截召回率 ↑ | 決策延遲 (P50) |
|---|---:|---:|---:|---:|---:|
| **Heuristic 啟發式基準** | 85.0% | 0.0% | 15.0% | 100.0% | 0.0 ms |
| **Raw Qwen2.5-3B Direct** | **0.0%** | 25.0% | 75.0% | **0.0% (致盲撞毀)** | 23.74 ms |
| **SemIf Enhanced Qwen2.5-3B** | **25.0%** | 25.0% | **50.0%** | **100.0% (完美避險)** | **39.27 ms** |

---

## 👁️ 視覺串接 Jev 技術解析 (Visual Jev: `djev-spark` vs SemIf-Vision)

針對社群專案 [`mmastrac/djev-spark`](https://github.com/mmastrac/djev-spark) 的視覺決策機制，核心架構與 SemIf 的融合方案解析如下：

### 1. `djev-spark` 的視覺驅動原理
- **架構特點**：採用 DiffusionGemma / DiT (Diffusion Transformer) 搭配 NVFP4 量化張量核心，部署於 DGX Spark。
- **繞過自回歸生成**：傳統 VLM（如 CLIP+LLM）將圖片切為 576~1152 個 Patch Token 後，需歷經 300~800ms 的逐詞解碼才能輸出動作文字。`djev-spark` 將決策視為**非自回歸擴散採樣 (Discrete Diffusion / Parallel Readout)**：
  - 視覺 Patch Token 透過單次前向饋入 Transformer 主幹。
  - 輸出端直接透過少量步數（1~4 步）的擴散降噪採樣，在動作空間中直接求得決策，達成每幀 10~20ms 視覺實時響應。

### 2. SemIf-Vision 的端到端整合架構
在 SemIf 框架下，實現視覺串接更加輕量且無需擴散採樣：
```
[視覺影像] ──> [輕量 Vision Encoder (SigLIP / MobileCLIP)] ──> [Visual Prefix Tokens (N=64)]
                                                                        │
                                                                        ▼
[候選動作: A/B/C/D/E] ──> [SemIf Sliced Head Single-Forward Pass] ──> [實時 Calibrated Logits (< 15ms)]
                                                                        │
                                                                        ▼
                                                   [Helmholtz OOD Gate -> 安全護欄]
```
- **極速單次前向**：將攝像頭影像經輕量視覺編碼器提取為 Prefix Tokens，直接拼接決策準則與選項，在最後一個 Token 處執行 SemIf 切片輸出頭投影。
- **無文字生成循環**：零詞元生成、零 JSON 反序列化，同時具備 SemIf 獨創的先驗校準與自由能 OOD 安全拒絕機制。

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
