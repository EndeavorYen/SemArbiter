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

第一次真實視覺會下載 CLIP ViT-B/32。換成 SigLIP：`SEMIF_VISION_MODEL=google/siglip-base-patch16-224`。

---

## 資料流

```mermaid
flowchart LR
    Canvas["3D canvas JPEG"] --> Vision["/v1/vision CLIP"]
    Vision --> Evidence["state.vision"]
    Sampler["幾何採樣器 tXX"] --> Cands["candidates"]
    Evidence --> SemIf
    Cands --> SemIf["classify_jev Flat"]
    SemIf --> Pick["selected id"]
```

像素不進 letter-slot prompt。`compact_jev_state` 只留 `vision.signal / red / pedestrian / …`。

閉環沒有真實相機：`vision_mode=synthetic` 依場景名填合成證據（紅燈→red 高）。這是夾具，**不是** CLIP 成績。CUDA 報告若 `backend=synthetic` 不得寫成「視覺編碼器準確率」。

---

## 契約

| 層 | 做 | 不做 |
| :--- | :--- | :--- |
| CLIP／SigLIP | 畫面 → 短證據 | 輸出轉角 |
| 採樣器 | 幾何軌 | 因 `vision.signal=red` 多塞停車軌 |
| SemIf | 選本幀 id | 自由文字 |

同一 seed：加不加 `state.vision`，`candidates` 的鍵必須相同。
