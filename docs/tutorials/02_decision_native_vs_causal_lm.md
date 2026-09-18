# 教程 02：通用因果語言模型（Causal LM）與專門決策模型（Decision-Native Model）解析

> **模組對應**：`src/semif_phase1/decider.py`、`benchmarks/evaluate_decider.py`、`tests/test_decider.py`  
> **關聯任務**：[#3 [Sub-Issue 2] 評測環境整合 Mapika/decider-2b 對照基準](https://github.com/EndeavorYen/SemIf/issues/3)  
> **前置知識**：[教程 01：模型校準與 ECE 解析](01_model_calibration_and_ece.md)、Hugging Face 基礎概念。

---

## 導讀：2B 小模型真能擊敗 4B 大模型嗎？

在軟體工程與 AI Agent 架構中，有一個歷久彌新的原則：
> **通用大模型（Generalist LLM）適合探索與開放式對話；專門小型模型（Specialist SLM）適合特定任務的高速執行。**

當我們需要讓 AI 在每秒幾十甚至上百次的頻率下做條件判斷（If-Else 決策、安全過濾、客服路由）時，我們究竟應該：
- **方案 A（SemIf 原生路線）**：拿一個通用的開源 4B 因果語言模型（如 `Qwen/Qwen3.5-4B`），透過巧妙的 Prompt 與 Logits 提取當作分類器？
- **方案 B（Decider-2B 路線）**：拿一個參數量更小（1.9B），但專門用 100 萬筆決策資料與嚴格評分規則（Proper Scoring Rule）微調訓練的專門模型（如 `Mapika/decider-2b`）？

本篇教學將深入解構這兩種架構思維的本質差異，並展示如何在 SemIf 中建立對照適配器（Adapter）。

---

## 一、 架構本質剖析：Causal LM vs Decision-Native

```mermaid
flowchart TD
    subgraph CausalLM ["通用因果模型 (Qwen3.5-4B Direct)"]
        C1["輸入: Chat 模板 (JSON 結構)"] --> C2["Backbone: 4B 因果 Transformer"]
        C2 --> C3["LM Head: 151,936 全詞表矩陣乘法"]
        C3 --> C4["輸出: 提取單個 Token Logits (未校準)"]
    end

    subgraph DecisionNative ["專門決策模型 (Mapika/decider-2b)"]
        D1["輸入: Context / Question / Options / Answer 前綴"] --> D2["Backbone: 2B 特化微調 Transformer"]
        D2 --> D3["專門 Answer Slot 投影 + 溫度縮放 T=1.05"]
        D3 --> D4["輸出: 高校準機率 (ECE < 0.03) + 原生拒絕能力"]
    end
```

### 1. 通用因果語言模型（Causal LM）的特徵與侷限
- **核心目標**：自回歸續寫文字，詞表龐大（Qwen3.5 詞表超過 15 萬）。
- **優勢**：蘊含廣泛的世界知識與推理能力（MMLU、常識推理、程式碼理解均較強）。
- **缺點**：
  1. **計算浪費**：每做一次決策，都要計算一次 $[1, 151936]$ 的大矩陣投影，即使我們只需要 2 個選項的機率。
  2. **未校準（Uncalibrated）**：交叉熵損失讓它對預測過度自信，不能把 Softmax 機率直接當作信心門檻。
  3. **位置與先驗偏置**：模型天生對常見字母（如 `A`）有頻率偏好；選項前後顛倒容易改變選擇（翻轉率高）。

### 2. 專門決策模型（Decision-Native Model）的特徵與優勢
以開源的 [`Mapika/decider-2b`](https://huggingface.co/Mapika/decider-2b) 為例：
- **微調規模**：以 `Qwen3.5-2B-Base` 為基底，混合 **69 個決策資料集**、共 **94.2 萬筆樣本**，涵蓋意圖識別、工單路由、安全審查、NLI 自然語言推理與工具選擇。
- **訓練目標**：採用嚴格評分規則（Proper Scoring Rule，如 Brier Score），迫使模型輸出的 Softmax 機率嚴格對齊真實準確率。
- **內建拒絕機制（Abstention Augmentation）**：在 10% 的訓練題目中故意放入「無匹配選項（Off-topic options）」並強制標註「以上皆非（None of the above）」，從根本上解決了 Softmax 在遇到未見狀態時盲目猜測的缺陷。
- **校準品質飛躍**：
  - Zero-shot 4B 基模的 ECE 約在 **0.09 ~ 0.12**，Brier 分數約 **0.405**。
  - `decider-2b` 的 ECE 直接降至 **0.028 ~ 0.037**，Brier 分數大幅降至 **0.248**！

---

## 二、 顯存與延遲精算：為什麼 2B 是邊緣與本機推論的甜蜜點？

| 比較項目 | Qwen3.5-4B (SemIf 原生) | Mapika/decider-2b |
| :--- | :--- | :--- |
| **參數量** | 約 4.0B | 約 1.9B |
| **BF16 靜態顯存** | 約 8.0 GB | **約 3.9 GB** |
| **RTX 5080 (16GB) 剩餘容量** | 剩餘 ~ 8.0 GB (足夠) | **剩餘 ~ 12.0 GB (極為寬裕)** |
| **Mac mini (16GB) 適用度** | 原生運行可用，餘裕較緊 | **極佳，系統記憶體毫無負擔** |
| **推論延遲 (PyTorch Eager)** | 約 40 ~ 50 ms | 約 40 ~ 49 ms |
| **推論延遲 (CUDA Graphs + Compile)** | 未優化 (~40ms) | **壓榨至 4.0 ms！** |

2B 模型的靜態顯存不到 4GB，這意味著在你的 **RTX 5080 (16GB)** 上，有充裕的空間可以建立巨大的動態前綴快取池（Prefix Cache Pool）；而在 **16GB 的 Mac mini** 上，也能在背景無感運行，完全不影響日常多工操作。

---

## 三、 在 SemIf 中實作 Decider-2B 適配器

為了讓 SemIf 既有的資料集與評測管線能無縫評估 `decider-2b`，我們在 [`src/semif_phase1/decider.py`](file:///D:/Code/SemIf/src/semif_phase1/decider.py) 中建立了專屬適配器。

### 1. Prompt 模板適配
`decider-2b` 訓練時使用的是特定的三段式 Prompt：

```
Context:
{state 非結構化上下文}

Question: {question 決策問題}
Options:
(A) {option_A_description}
(B) {option_B_description}
Answer: (
```

適配核心函式：
```python
def format_decider_prompt(row: dict) -> str:
    validate_row(row)
    state = row["state"]
    state_text = json.dumps(state, ensure_ascii=False, indent=2) if isinstance(state, (dict, list)) else str(state).strip()

    options_lines = [f"({LETTERS[i]}) {opt['description'].strip()}" for i, opt in enumerate(row["options"])]
    options_block = "\n".join(options_lines)
    question = row["question"].strip()

    return f"Context:\n{state_text}\n\nQuestion: {question}\nOptions:\n{options_block}\nAnswer: ("
```

### 2. 溫度縮放 Softmax（Temperature-Scaled Softmax）
`decider-2b` 官方建議推論時施加 $T=1.05 \sim 1.30$ 的溫度係數：

```python
# 提取 slot tokens (A, B, ...) 對應的 logits
selected = vocabulary[slots].cpu().tolist()
# 施加溫度縮放：z_i / T
scaled_logits = [val / temperature for val in selected]
# 計算平滑機率
probabilities = softmax(scaled_logits)
```

### 3. CLI 一鍵切換
現在你可以直接透過 `semif-score` 調度 `decider` 模式：
```bash
semif-score \
  --mode decider \
  --model Mapika/decider-2b \
  --revision <commit_hash> \
  --temperature 1.05 \
  --input examples/decisions.jsonl \
  --output results_decider.jsonl
```

---

## 四、 橫向對比評測工具：`benchmarks/evaluate_decider.py`

我們提供了專門的對照評測腳本 [`benchmarks/evaluate_decider.py`](file:///D:/Code/SemIf/benchmarks/evaluate_decider.py)，能同時載入 Gold 標準答案、Decider 預測輸出、以及 Qwen3.5-4B 基準輸出：

```bash
python benchmarks/evaluate_decider.py \
  --gold manifests/authored_decisions.jsonl \
  --decider-predictions results/decider_authored.jsonl \
  --baseline-predictions results/raw/authored-decisions-direct.predictions.jsonl \
  --output results/decider_comparison_report.json
```

該腳本將自動計算：
1. **平衡準確率（Balanced Accuracy）** 與 **Macro F1**。
2. **預期校準誤差（ECE 15 bins）** 與 **Brier Score**。
3. **Paired Bootstrap 差異顯著性**（與通用 4B 基準相比，差異是否具備統計顯著性）。

---

## 五、 結論與架構啟示

1. **專門決策模型是「高信賴度」的典範**：
   - 如果你的任務是關鍵流程路由、自動化重試或風控，`decider-2b` 這種經過 Proper Scoring Rule 訓練的模型，其機率值具備真實的統計信賴度（Confidence），遠優於通用大模型的原始 Logits。
2. **通用大模型是「泛化推理」的底座**：
   - 面對從未見過的複雜多步驟推理題目時，4B 模型憑藉更大的參數量與更深的世界知識儲備，依然有其獨特優勢。
3. **下一步研究路徑**：
   - 能否「既要又要」？這正是我們下一課 **[Sub-Issue 3]** 要探索的：**如何對通用 4B 模型施加零訓練的推論期校準（溫度縮放 + 空先驗去偏），讓 4B 大模型也能擁有媲美 2B 專用模型的校準精度！**
