# 教程 01：模型校準（Model Calibration）與 ECE（預期校準誤差）解析

> **模組對應**：`benchmarks/evaluate.py`、`tests/test_calibration_metrics.py`  
> **關聯任務**：[#2 [Sub-Issue 1] 擴充校準度量工具鏈 (ECE, Brier 分數, 排列翻轉率)](https://github.com/EndeavorYen/SemIf/issues/2)  
> **前置知識**：基本的機率統計概念、Softmax 函數、基本的 Python 語法。

---

## 導讀：為什麼我們不只要「選對」，更要「知己知彼」？

在傳統的自然語言處理（NLP）評測中，大家最常看的指標是 **準確率（Accuracy）**：
- 「這個分類問題模型猜對了沒有？」
- 只要模型猜 `A`，正解也是 `A`，就算成功。

但在現代 **AI Agent（智慧體工作流）** 與 **軟體控制流（Control Flow）** 中，單純的「準確率」是遠遠不夠的。試想以下兩個關鍵工業場景：

1. **工單自動路由（Ticket Routing）**：
   - 如果模型判斷這張工單屬於「退款部門」的信心高達 **98%**，軟體可以直接放行自動退款。
   - 如果模型的判斷信心只有 **51%**（在退款與售後之間猶豫不決），軟體應該將這筆工單轉給「人工客服審查（Human-in-the-loop）」。
2. **自動駕駛或醫療診斷等安全護欄（Safety Guardrails）**：
   - 系統需要知道「模型自己何時不確定」。若模型明明不確定卻盲目給出 99% 的信心，軟體系統將無法設計有效的安全備援防線。

這就是 **模型校準（Model Calibration）** 的核心命題：
> **如果模型在預測某個選項時給出了 80% 的置信度（Confidence），那麼在統計上，這類預測真實正確的機率就應該正好是 80%。**

---

## 一、 現代 LLM 與 Softmax 的通病：嚴重的「過度自信（Overconfidence）」

### 1. 交叉熵損失（Cross-Entropy）的訓練原罪
目前主流的生成式語言模型（如 LLaMA、Qwen、DeepSeek 等）都是以自回歸（Autoregressive）預測下一個 Token 為目標訓練出來的。其訓練損失函數為標準的交叉熵（Cross-Entropy）：

$$\mathcal{L}_{\text{CE}} = - \log P(w_t \mid w_{<t}) = - \log \left( \frac{e^{z_{\text{gold}}}}{\sum_j e^{z_j}} \right)$$

為了讓損失 $\mathcal{L}$ 最小化（趨近於 0），反向傳播梯度會強烈促使模型把正確選項的 Logit $z_{\text{gold}}$ 推得非常大，把其他干擾選項的 Logits 壓得非常低。

### 2. 未校準的 Softmax 尖銳現象
當我們直接拿未經決策特化微調的通用模型（如 Qwen3.5-4B）提取選項 Logits，並通過標準 Softmax 歸一化時：

$$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

Logits 之間哪怕只有 4~5 的微小數值差距，經過指數函數運算後，機率分佈就會變得極端尖銳：
- $z_A = 10, z_B = 5 \implies e^{10} / (e^{10} + e^5) \approx 99.3\% \text{ vs } 0.7\%$
- 系統動輒輸出 99% 的信心，即使模型根本答錯了！這種現象在機器學習界被稱為 **過度自信（Overconfidence）**。

---

## 二、 社群五大流派定位：我們身在何處？

在開源社群復刻 TypeSafe Jev 的浪潮中，出現了五種不同取向的模型架構：

```mermaid
flowchart LR
    Reflex["Reflex (SemIf 原型)<br>通用 4B + Direct Logits<br>未經校準"] --> SysOne["System-One 4B<br>專注溫度與先驗去偏<br>機率具統計信賴度"]
    SysOne --> Nano["NanoJev 0.6B<br>專屬 Decision Head<br>極致低延遲與節能"]
    Nano --> Decider["Decider-2B<br>Proper Scoring 訓練<br>端到端高精度"]
    Decider --> Laya["Laya 421M<br>原生 Mac (MLX) 支援<br>端側極致輕量"]
```

目前 SemIf 原始專案處於 **Reflex** 階段（直接提取 raw logits）。而我們現在推進的 **Sub-Issue 1、3、4**，正是帶領專案邁向 **System-One 4B**（具備統計學意義的真實置信度）的關鍵基石！

---

## 三、 校準度量指標深度解析

要改進校準，第一步是「**如何精確衡量校準品質**」。我們在 `benchmarks/evaluate.py` 引入了以下核心度量：

### 1. 預期校準誤差（Expected Calibration Error, ECE）

ECE 是目前學術界與工業界評估神經網絡置信度最經典的指標（出自 Guo et al., ICML 2017《On Calibration of Modern Neural Networks》）。

#### 計算步驟：
1. **區間分箱（Binning）**：將 $[0, 1]$ 的置信度區間等分為 $M$ 個區間（Bins），常用 $M=10$ 或 $M=15$。
   - 例如 $M=10$ 時，各分箱為 $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$。
2. **樣本落入分箱**：令 $B_m$ 表示所有「模型最高預測置信度落在第 $m$ 個區間」的樣本集合。
3. **計算各箱的平均置信度與平均正確率**：
   - 區間平均置信度：
     $$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
   - 區間實際正確率：
     $$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
4. **加權平均得出 ECE**：
   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   - 其中 $N$ 為總樣本數。
   - **數值意義**：ECE 介於 $0$ 到 $1$ 之間。**ECE 越接近 0，表示模型越完美校準**。

---

### 2. 最大校準誤差（Maximum Calibration Error, MCE）

$$ \text{MCE} = \max_{m=1, \dots, M} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| $$
- 衡量在所有分箱中，**偏差最嚴重的那一個分箱的誤差**。
- 這在安全至上的關鍵場景（High-stakes decisions）特別重要，用來確認模型是否在某個特定的信心區間存在巨大的系統性盲點。

---

### 3. 布萊爾分數（Brier Score）

布萊爾分數是預測科學中的一種 **嚴格評分規則（Proper Scoring Rule）**，計算預測機率向量與真實 One-Hot 標籤之間的均方誤差（MSE）：

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (p_{ik} - y_{ik})^2$$

- **直觀特點**：如果正解是選項 A，模型預測 $p_A = 0.99$，Brier 分數會非常小（接近 0，極好）；若模型預測 $p_A = 0.51$，即使最終 Argmax 選對，Brier 分數依然會因為「信心不足」而受到懲罰；若模型預測 $p_A = 0.01$（猜錯），Brier 分數會遭到嚴厲重罰。
- 因此，Brier 分數能同時衡量「**分類準確性**」與「**機率校準品質**」。

---

### 4. 可靠度圖（Reliability Diagram）

可靠度圖將各分箱的 `mean_confidence` 作為 X 軸，`accuracy` 作為 Y 軸繪製成長條圖：
- **對角線（$y=x$）**：代表完美的校準（Perfect Calibration）。當信心為 70% 時，準確率剛好是 70%。
- **長條低於對角線（$\text{acc} < \text{conf}$）**：代表 **過度自信（Overconfidence）**。模型自以為有 90% 把握，實測卻只有 60% 準確率。
- **長條高於對角線（$\text{acc} > \text{conf}$）**：代表 **過於保守（Underconfidence）**。

---

## 四、 程式碼實踐與解構

在本次改進中，我們在 [`benchmarks/evaluate.py`](file:///D:/Code/SemIf/benchmarks/evaluate.py) 中實作了高效率的 `compute_ece()` 函式：

```python
def compute_ece(valid_rows, n_bins=15):
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and per-bin statistics."""
    if not valid_rows:
        return dict(ece=None, mce=None, n_bins=n_bins, bins=[])
    n_total = len(valid_rows)
    bins = []
    weighted_gap_sum = 0.0
    max_gap = 0.0

    for b in range(n_bins):
        # 將樣本按最高置信度分入對應 bin [b/n_bins, (b+1)/n_bins)
        part = [r for r in valid_rows if min(n_bins - 1, int(r['confidence'] * n_bins)) == b]
        if part:
            count = len(part)
            mean_conf = sum(r['confidence'] for r in part) / count
            acc = sum(r['correct'] for r in part) / count
            gap = abs(acc - mean_conf)
            weighted_gap_sum += (count / n_total) * gap
            if gap > max_gap:
                max_gap = gap
            bins.append(dict(
                bin=b,
                lower=b / n_bins,
                upper=(b + 1) / n_bins,
                n=count,
                fraction=count / n_total,
                mean_confidence=mean_conf,
                accuracy=acc,
                gap=gap,
            ))
    return dict(ece=weighted_gap_sum, mce=max_gap, n_bins=n_bins, evaluated_samples=n_total, bins=bins)
```

### 重點設計考量：
1. **安全邊界處理**：使用 `min(n_bins - 1, int(r['confidence'] * n_bins))`，確保當 `confidence == 1.0` 時不會陣列越界，而是平滑落入最後一個頂級分箱。
2. **多尺度報告**：系統同時計算 $M=10$（傳統標準）與 $M=15$（`Mapika/decider-2b` 論文標準），使後續對比具備高度可比性。
3. **無損向後相容**：保留原有的 `reliability_bins` 輸出，保證既有的測試驗證腳本與 Checksum 驗證 100% 通過。

---

## 五、 單元測試驗證

在 [`tests/test_calibration_metrics.py`](file:///D:/Code/SemIf/tests/test_calibration_metrics.py) 中，我們為此度量工具撰寫了嚴密的單元測試，涵蓋：
- **完美校準情境**：驗證當準確率完全對齊信心時，`ECE == 0.0`。
- **完全過度自信情境**：10 筆樣本皆給出 95% 信心但全數答錯，驗證 `ECE == 0.95`。
- **多分箱加權情境**：手動計算加權理論值，驗證演算法數值精度完全吻合。

執行指令：
```bash
pytest -q
```
輸出：
```
....................                                                     [100%]
20 passed in 0.05s
```

---

## 六、 課後思考與下一階段預告

### 自我檢驗思考題：
1. 如果一個二元分類模型對所有樣本一律無腦輸出 50% 信心，而測試集中正負樣本各佔一半，此時該模型的準確率是多少？它的 ECE 是多少？（提示：思考完美平緩與無資訊決策）。
2. 為什麼在決策模型中，單看 Brier Score 比單看 Accuracy 更能反映模型是否具備「工業上線可靠度」？

### 下一課預告：
- **[Sub-Issue 2 (#3)]**：我們將把開源專門決策模型 **Mapika/decider-2b** 接入評測管線，實測對比通用 4B 與專門 2B 的 ECE 差距！
