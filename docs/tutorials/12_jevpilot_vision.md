# 教程 12：JevPilot 視覺輸入

> **模組**：`src/semif_phase1/vision.py`、`demo/server.py`（`POST /v1/vision`）、`demo/jevpilot/semif-layer.js`、`benchmarks/benchmark_jevpilot_vision.py`  
> **Issue**：[ #90](https://github.com/EndeavorYen/SemIf/issues/90) [ #48](https://github.com/EndeavorYen/SemIf/issues/48) 傘 [#89](https://github.com/EndeavorYen/SemIf/issues/89)

視覺認路況，Jev 選本幀軌 `tXX`。不准自由文字轉角。不准因視覺標籤在採樣器裡塞停車軌。

---

## 驗收怎麼開

```bash
python demo/server.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --port 8000
```

`http://localhost:8000/jevpilot/?seed=42` → 左下 `VISION · … · sig …`。關視覺：`?vision=0`。

```bash
# 測試（不下載 CLIP）
pytest -q tests/test_vision.py

# 閉環 A/B：遙測 vs 合成視覺（mock）
python benchmarks/benchmark_jevpilot_vision.py --mock --episodes 1 --seed 42 \
  --output results/phase5-jevpilot-vision-mock.json

# CUDA（與其他 JevPilot 報告同一套誠實規則）
python benchmarks/benchmark_jevpilot_vision.py --episodes 2 --seed 42 \
  --output results/phase5-jevpilot-vision-cuda.json
```

第一次真實視覺會下載 **SigLIP** `google/siglip-base-patch16-224`（#48）。視覺主要角色為**慢迴圈可供性提取**：編碼器將路況提煉為 `red/pedestrian` 等短分數，注入 `state.vision` 與 HUD，快迴圈維持純文字 Sliced Head + 512 桶 CUDA Graph 進行動態選軌。

實驗性 Patch 前綴（不作預設）：`SEMIF_VISION_PREFIX=1`。未訓練投影，會關 CUDA Graph。尚未 CUDA 閉環（詳見 [教程 13](13_semif_vision_routes_and_tradeoffs.md)）。`compact_jev_state` 不放 `backend` / `prefix_tokens`，避免撐破 512 桶。

---

## 資料流

```mermaid
flowchart LR
    Canvas["3D canvas JPEG"] --> Vision["/v1/vision SigLIP (慢迴圈 10-20Hz)"]
    Vision --> Evidence["state.vision.event (一句相機事件)"]
    Sampler["幾何採樣器 tXX"] --> Cands["candidates"]
    Evidence --> State["compact_jev_state"]
    State --> SemIf["SemIf Sliced Head (快迴圈 50Hz)<br/>512 桶 CUDA Graph"]
    Cands --> SemIf
    SemIf --> Pick["選中軌跡 ID"]
```

像素不直接作為未訓練 token 注入決策迴圈。`compact_jev_state` 只留 `vision.event` 與 `signal`。事件來自**相鄰兩張圖的像素框**（變大＝靠近、橫移向中心＝切入、上一幀沒有＝出現），不讀世界座標，不把示意幀尺度寫成「TTC 2.0s」。沒有框時才退回 CLIP 分數差。1024 桶可接受。

閉環沒有 3D 相機。官方 CUDA 成績用 **編碼器看 PIL 示意幀**，`vision_mode=clip`（名稱沿用；backend 可能是 SigLIP）。載入失敗直接中止，不准退回合成標籤。

`vision_mode=synthetic` 只給 pytest／`--mock`，不得當 CLIP 準確率。

---

## 契約

| 層 | 做 | 不做 |
| :--- | :--- | :--- |
| CLIP／SigLIP | 畫面 → 短證據 | 輸出轉角 |
| 採樣器 | 幾何軌 | 因 `vision.signal=red` 多塞停車軌 |
| SemIf | 選本幀 id | 自由文字 |

同一 seed：加不加 `state.vision`，`candidates` 的鍵必須相同。
