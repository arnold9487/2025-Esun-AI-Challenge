# 模型訓練與預測 (Model Training & Prediction)

本資料夾包含 **Positive-Unlabeled (PU) Learning** 的核心實作。透過迭代擴展的方式，從大量的未標記資料中挖掘出可靠的負樣本 (RN)，並訓練最終的 LightGBM 模型。

## 📋 執行順序與功能說明

請確保 `Preprocess` 階段已完成，或已準備好對應的輸入資料 (`alert_features.csv`, `sampled_unrelate_shuffled.csv` 等)。

| 順序 | 檔案名稱 | 功能詳細說明 |
| :--- | :--- | :--- |
| **7** | `7_find_rn_lgbm.py` | **可靠負樣本篩選 (RN Discovery)**<br>利用 **Spy Technique** 策略：<br>1. 將部分 Alert 樣本混入未標記資料作為「間諜 (Spy)」。<br>2. 訓練初步模型，觀察間諜樣本的預測分數。<br>3. 設定閾值 (Threshold) 為間諜樣本的最低分。<br>4. 將分數低於此閾值的未標記樣本視為 **Reliable Negatives (RN)** 並輸出。 |
| **8** | `8_iterative_training.py` | **迭代擴展訓練 (Iterative Expansion)**<br>這是 PU Learning 的核心迴圈：<br>1. 初始訓練集 = Alert + RN。<br>2. **動態守門員 (Dynamic Gatekeeper)**：每輪迭代均動態劃分驗證集，決定該輪的高信心閾值。<br>3. 從剩餘的 Unrelated 資料中，挖掘高於/低於閾值的樣本加入訓練集。<br>4. 最終輸出擴展後的完整訓練資料集 (`final_P`, `final_N`)。 |
| **9** | `9_predict_final.py` | **最終預測 (Final Inference)**<br>使用步驟 8 生成的最終擴展資料集進行全量訓練 (Full Training)。<br>針對 `predict_features.csv` 進行預測，並依據比賽要求的 Top-K% (2%, 5%, 10%...) 門檻，生成多份提交檔案。 |

---

## ⚙️ 模型參數與設定

* **模型選擇**：LightGBM (Gradient Boosting Decision Tree)
* **關鍵參數**：
    * `objective`: binary
    * `metric`: auc
    * `learning_rate`: 0.05
    * `scale_pos_weight`: 動態計算 (N/P) 以處理類別不平衡
* **特徵處理**：
    * 在訓練前會自動移除 **靜態特徵 (Static Features)** (如 `total_count`, `amount_std` 等)，強迫模型學習交易行為模式而非單純的交易量級，以提升泛化能力。

## 📂 輸出檔案

執行完畢後，預測結果將儲存於 `../Data/` (或 `../Data_原版/`) 目錄下，檔名格式為：
`predictions_official_top{K}percent.csv`
最後請測top5的檔案，也就是我當初比賽時的提交檔