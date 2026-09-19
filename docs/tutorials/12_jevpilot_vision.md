# 教程 12：JevPilot 視覺輸入（明早驗收）

> **模組**：`src/semif_phase1/vision.py`、`demo/server.py`（`POST /v1/vision`）、`demo/jevpilot/semif-layer.js`  
> **Issue**：[ #90](https://github.com/EndeavorYen/SemIf/issues/90) [ #48](https://github.com/EndeavorYen/SemIf/issues/48) 傘 [#89](https://github.com/EndeavorYen/SemIf/issues/89)

視覺認路況，Jev 仍選本幀軌 `tXX`。不准自由文字轉角。不准因視覺標籤在採樣器裡塞停車軌。

---

## 明早怎麼開

```bash
# 有 GPU（決策用 3B；視覺預設 CPU CLIP）
python demo/server.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --port 8000

# 沒 GPU 也能看 HUD 管線（視覺可能是 stub）
python demo/server.py --mock --port 8000
```

瀏覽器：`http://localhost:8000/jevpilot/?seed=42`  
左下應出現 `VISION · openai/clip-vit-base-patch32 · sig …`  
Autopilot 選 SemIf。關視覺：`?vision=0`

第一次會下載 CLIP（約 600MB）。3B 與 CLIP 分開裝置：決策 CUDA、視覺預設 CPU（`SEMIF_VISION_DEVICE=cuda` 可改）。

換成 SigLIP／MobileCLIP：`SEMIF_VISION_MODEL=google/siglip-base-patch16-224`（須 transformers 吃得下該卡）。

---

## 資料流

1. 3D canvas 縮成 320px JPEG  
2. `POST /v1/vision` → `{signal, red, green, pedestrian, vehicle, construction}`  
3. 下一次 `POST /v1/classifier` 的 `state.vision` 帶這些欄（compact，不含像素）  
4. 軌仍來自 planner；SemIf 選 id  

測試用 `synthetic_vision()`，不打權重。
