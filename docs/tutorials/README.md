# SemIf 核心技術教學系列（Technical Tutorials）

本系列教學文檔旨在配合 SemIf 語意決策引擎的各階段架構演進，深入剖析大語言模型在語意決策、概率校準、推論加速與邊緣端部署的底層幾何與演算法原理。

---

## 教程目錄

| 篇章 | 主題 | 核心概念與模組 | 關聯 Issue |
| :--- | :--- | :--- | :--- |
| **[教程 01](01_model_calibration_and_ece.md)** | 模型概率校準與 ECE 指標 | 預測置信度與真實準確率對齊、期望校準誤差（ECE）、可靠性曲線（Reliability Diagram） | [#2](https://github.com/EndeavorYen/SemIf/issues/2) |
| **[教程 02](02_decision_native_vs_causal_lm.md)** | 決策原生模型 vs 因果語言模型 | Decider-2B 架構、Slot 概率解碼、Zero-shot 決策對比評估 | [#3](https://github.com/EndeavorYen/SemIf/issues/3) |
| **[教程 03](03_inference_time_calibration.md)** | 推論期零訓練去偏與溫度校準 | 溫度縮放（Temperature Scaling）、先驗無效提示詞校準（Prior Calibration）、黃金分割凸優化 | [#4](https://github.com/EndeavorYen/SemIf/issues/4) |
| **[教程 04](04_prefix_cache_and_position_bias.md)** | 前綴快取與選項位置偏置消除 | RoPE 近因效應、因果注意力非對稱性、排列組合集成（Permutation Ensembling）、KV Cache 重用 | [#5](https://github.com/EndeavorYen/SemIf/issues/5) |
| **[教程 05](05_open_world_and_ood_detection.md)** | 開放世界拒絕與基於自由能的 OOD 偵測 | Softmax 盲點與平移不變性、Helmholtz 自由能理論、獨立 Sigmoid 多標籤門控與棄權機制 | [#6](https://github.com/EndeavorYen/SemIf/issues/6) |
| **[教程 06](06_transformer_lm_head_optimization.md)** | Transformer Sliced LM Head 投影優化 | 詞表矩陣切片、RTX 5080 記憶體頻寬解耦、跳過 150k 全詞表解碼、無損等價性驗證 | [#7](https://github.com/EndeavorYen/SemIf/issues/7) |
| **[教程 07](07_cuda_graphs_and_compilation.md)** | CUDA Graphs 與 torch.compile 極限延遲優化 | Kernel Launch 開銷消除、形狀分桶（Shape Bucketing）、RTX 5080 實測 4.6ms 延遲 | [#8](https://github.com/EndeavorYen/SemIf/issues/8) |
| **[教程 08](08_mac_silicon_and_mlx_deployment.md)** | Apple Silicon Mac mini 邊緣端部署 | MLX 原生架構、統一記憶體（Unified Memory）零拷貝決策執行、16GB 4-bit 20W 超能效部署 | [#9](https://github.com/EndeavorYen/SemIf/issues/9) |
| **[教程 09](09_dynamic_candidates_and_arbitration.md)** | 動態候選空間與語意仲裁（Sampler vs Arbiter） | 物理幾何採樣與語意仲裁解耦、點質量動力學 Rollout、評測誠實性規範、乾淨完成率（Clean Completion） | [#71](https://github.com/EndeavorYen/SemIf/issues/71) [#72](https://github.com/EndeavorYen/SemIf/issues/72) [#73](https://github.com/EndeavorYen/SemIf/issues/73) |
| **[教程 10](10_jev_vs_classifier_io.md)** | Jev 與傳統分類器（輸入／輸出維度與階層樹退役實證） | Schema 約束 vs 語意解空間、Letter-slot 甜區、Hierarchical 硬剪枝失敗歸因（RTX 5080 GPU 實測）、Typed Action 契約 | [#80](https://github.com/EndeavorYen/SemIf/issues/80) [#79](https://github.com/EndeavorYen/SemIf/issues/79) |
