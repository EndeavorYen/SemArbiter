# 教程 11：JevPilot 延遲與正確度的取捨（壓進 512 桶）

> **模組對應**：`src/semif_phase1/trajectory_sampler.py`（`compact_jev_state`、`vector_option_tag`）、`demo/server.py`、`benchmarks/profile_classify_latency.py`、`benchmarks/ablate_option_tags.py`  
> **關聯任務**：[ #86 為何 50ms+](https://github.com/EndeavorYen/SemIf/issues/86)  
> **前置知識**：[教程 07：CUDA Graphs](07_cuda_graphs_and_compilation.md)、[教程 09：採樣 vs 仲裁](09_dynamic_candidates_and_arbitration.md)

---

## 導讀

教程 07 的 ~4.6ms、README 曾寫的 12ms，是**短 prompt、少選項**的微基準。JevPilot 閉環把整包 `candidates` 再塞進 `state`，prompt 到 **3108 token**，超過 CUDA Graph 最大桶 2048，變成 eager forward。再問一次沒用的 `motion`，P50 來到 **~440 ms**。

縮 state、拿掉 motion、把選項寫短，可以進 **512 桶**。但同一 seed 的閉環證明：**包裝不是免費的，正確度會變。**

---

## 一、Root cause

`classify_jev` → `encode_prompt` → Graph bucket → sliced lm_head。

| 題 | tokens | Graph | 總延遲 |
| :--- | ---: | :--- | ---: |
| 閉環完整 obs（軌進 evidence） | 3108 | 無 | ~286 ms 單次；閉環兩題 ~440 ms |
| 16 個 id、幾乎沒描述 | 336 | 512 replay | 46 ms |
| 短 5 選 | 142 | 256 replay | 22 ms |

16 個選項本身不貴。貴的是把軌描述複製進 evidence，序列超過 2048。

---

## 二、同一 seed 42 的正確度

採樣器不變。Heuristic 場景表完全相同。變的是 SemIf 的 prompt。

| 設定 | tokens | 桶 | 乾淨完成 | P50 | 闖紅燈 | 撞車 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| 長 prompt（軌進 state） | 3108 | 無 | 60% | 441 ms | **0** | 0 |
| csv `16.2,-0.14,0.00,0,1` | 500 | 512 | 35% | 48 ms | 2 | 9 |
| **words（預設）** `collision=yes halt=no` | **468** | **512** | **75%** | **49 ms** | 2 | 2 |
| verbose 英文長句 | 580 | 1024 | 85% | 89 ms | 2 | 0 |

檔案：`results/phase5-jevpilot-2.0-dq-rtx5080-planner-policy.json`（長 prompt）、`results/phase5-jevpilot-2.0-dq-rtx5080-prompt512.json`（csv）、`results/phase5-jevpilot-option-tag-ablation.json`（三風格）。

---

## 三、預設取捨

**預設 `OPTION_TAG_STYLE=words`。** 仍在 512 桶（~20 Hz），乾淨完成 75%，高於長 prompt 的 60% 與 Heuristic 的 65%。英文關鍵字要留，不能收成純 `0,1`。

代價：紅燈長 prompt 曾是 0 違規，壓縮後三種都是 2。`SEMIF_OPTION_TAG=verbose` 可換 85% 完成、約 90ms，仍進不了 512，紅燈也不會回到 0。

程式：`compact_jev_state` 不把軌放進 evidence；閉環不發 `motion`；Graph 桶 `(256, 512, 1024)`。

---

## 四、不要再搞混的數字

- 4.6ms / 12ms ≠ JevPilot 閉環。
- 壓 token ≠ 正確度不變。要講正確度，必須同一 seed 再跑閉環。
- 1024 桶會把 627 token pad 滿，可能比 eager 還慢。目標是進 512，不是盲目加大桶。
