# 教程 10：Jev 與傳統分類器——輸入／輸出維度，以及為何 JevPilot 拿掉 Hierarchical

> **模組對應**：`src/semif_phase1/trajectory_sampler.py`、`demo/server.py`、`benchmarks/driving_quality.py`  
> **關聯任務**：[ #80 移除 JevPilot Hierarchical](https://github.com/EndeavorYen/SemIf/issues/80)、[ #79 樹只解釋不刪軌](https://github.com/EndeavorYen/SemIf/issues/79)、[ #71 動態採樣](https://github.com/EndeavorYen/SemIf/issues/71)  
> **前置知識**：[教程 02：決策原生 vs 因果 LM](02_decision_native_vs_causal_lm.md)、[教程 09：動態候選與仲裁](09_dynamic_candidates_and_arbitration.md)

---

## 導讀

傳統分類器問的是：「在一個**固定、事先列好的標籤表**上，這筆輸入屬於哪一類？」  
Jev／SemIf 問的是：「在**這一刻才存在的候選集合**上，選哪一個，且輸出必須符合型別契約？」

兩者都可以做成 letter-slot／argmax。差在 **in／out 的維度從哪裡來**，以及限制的是 **schema** 還是 **語意可能**。

JevPilot 閉環上，Hierarchical 樹曾經把本幀軌砍成 halt／lateral／lane 再打分。那是把 Jev 退化成傳統粗分類器。GPU 上它輸給 Flat。執行路徑已移除；下列數字與檔案留作 insight。

---

## 一、輸入維度（In）

| | 傳統分類器 | Jev 仲裁 |
| :--- | :--- | :--- |
| 標籤空間 | 訓練時固定（貓／狗／車） | 不固定；候選由採樣器／規劃器當場生成 |
| 輸入 | 特徵 → 固定 K 類 | **state**（幾何＋語意）＋ **本幀 candidates** |
| 視覺 | 像素直接進分類頭 | 視覺應先變成 state／可供性；Jev 不自由生成轉角文字 |

JevPilot 的 in：`speed_mps`、路口號誌、行人、障礙，加上 `candidates: { t00: [speed, steer, …], … }`。號誌在 state，不在採樣器。

---

## 二、輸出維度（Out）

要限制 output，限制的是**控制層級的契約**，不是把世界收成三個死按鈕。

```text
❌ 不限制：自由文字 "I think I should steer a bit left…"
   → 解析失敗、無法 clamp、自回歸延遲

❌ 低級限制：永遠 ["TURN_LEFT","TURN_RIGHT","BRAKE"]
   → 查表，不需要 Jev

✅ 高級限制：必須回傳
   { "selected_trajectory_id": "t03" }   // id ∈ 本幀 candidates
   可選 target_speed_mps ∈ [0, vmax]
```

K 很小（約 8–16 條軌）正是 letter-slot 的甜區。再把它切成 2～3 個粗類，是在 **out 維度上二次壓縮**，精簡掉的是語意可能，不是噪音。

---

## 三、為何 Hierarchical 在 JevPilot 不好

採樣器已經做過物理可行性。Jev 該做的是對這盤小菜單一次仲裁（Flat SemIf）。

樹若**刪軌**：路邊臨停時 lateral 桶把「微減速、幾乎不轉」的安全軌丟掉。  
RTX 5080、Qwen2.5-3B、seed 42、raw_mode：

| 檔案 | Hierarchical | Flat | 臨停 |
| :--- | ---: | ---: | :--- |
| `results/phase5-jevpilot-2.0-dq-rtx5080-dynsample.json`（刪軌） | 75% 乾淨完成 | 85% | 0/2 vs 2/2 |
| `results/phase5-jevpilot-2.0-dq-rtx5080-explain-tree.json`（只解釋） | 85% | 85% | 都 2/2 |

刪軌輸在語意集合；只解釋則與 Flat 對齊，只多延遲（P50 ~457ms）。因此 JevPilot **執行路徑不再跑 Hierarchical**。`src/semif_phase1/hierarchical.py` 仍留給「葉子本來就很多」的任務，不掛在這台車上。

---

## 四、現在 JevPilot 的 output action

閉環落地的是本幀軌 id。向量六欄：`[speed, steer, route_error, offroad, collision, stop_at_line]`。`stop_at_line` 是幾何，不是紅燈按鈕。Classifier 回 `answers.vector.choice`（例如 `t03`）。`motion: drive|stop` 不是執行器輸入。

Demo 策略只留 **SemIf（flat typed trajectory）** 與 **Heuristic**。`mode=semif_hierarchical` 若仍被送出，伺服器當成 flat。

---

## 五、對照

傳統分類器：固定 K、固定標籤、argmax。  
Jev：動態 K、動態描述、schema 鎖死、語意開放。  
不要用粗樹去模仿前者的 K。
