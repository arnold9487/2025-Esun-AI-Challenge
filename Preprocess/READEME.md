# 資料前處理與特徵工程 (Data Preprocessing & Feature Engineering)

本資料夾包含從原始交易紀錄到生成最終訓練特徵的所有程式碼。流程設計重點在於處理極大量的未標記資料 (Unlabeled Data)，以及確保特徵計算的數值穩定性。

## 📋 執行順序與功能說明

請依照檔名編號順序執行：

| 順序 | 檔案名稱 | 功能詳細說明 |
| :--- | :--- | :--- |
| **1** | `1_generate.py` | **資料分離**<br>讀取原始 `acct_transaction.csv`，將其拆解為 `Alert` (警示)、`Predict` (預測) 與 `Unrelated` (雙非) 三類帳戶的交易資料。<br>*(註：此步驟涉及 `set` 運算，若需完全復現後續分割順序，請使用 `Data_原版`)* |
| **2** | `2_split_unrelated.py` | **資料切分 (Chunking)**<br>由於 `Unrelated` 帳戶數量龐大，此腳本將其切分為多個小檔案 (Chunks)，存入 `unrelated_chunks/`，以利後續批次處理並避免記憶體溢位。 |
| **3** | `3_feature_extraction.py` | **核心特徵提取 (Alert/Predict)**<br>針對 `Alert` 與 `Predict` 帳戶進行行為特徵提取。<br>**關鍵設計**：採用 **迴圈 (Loop)** 方式逐筆計算，而非向量化操作，以確保浮點數運算結果在不同環境下的一致性。 |
| **4** | `4_extract_unrelated_features.py` | **批次特徵提取 (Unrelated)**<br>讀取 `unrelated_chunks/` 中的每個分塊，套用與步驟 3 完全相同的特徵提取邏輯，生成特徵檔至 `unrelated_features/`。 |
| **5** | `5_sample_distribution.py` | **分佈採樣 (Distribution Sampling)**<br>為了避免負樣本數量過大導致模型偏差，此腳本對 `Unrelated` 特徵進行分層抽樣，使其交易次數 (`total_count`) 的分佈呈現與目標資料相似的鐘形分佈。 |
| **6** | `6_shuffle_and_finalize.py` | **資料打亂**<br>對採樣後的資料進行隨機打亂 (Shuffle)，生成最終的未標記資料池 `sampled_unrelate_shuffled.csv`，供模型訓練使用。 |

---

## ⚠️ 復現性提示
* **檔案補齊** : Data資料夾裡只有alert跟predict的帳戶列表，acct_transaction.csv檔案太大無法上傳，需要使用者直接從官網獲取並放入Data。
並且"Data_原版"資料夾對復現我們的比賽結果有必要性，請前往Gmail附上的雲端連結下載解壓縮並貼上。
* **特徵一致性**：步驟 3 與 4 的特徵提取邏輯完全共用，包含分母平滑化 (`+1`) 與自由度設定 (`ddof=1`)，請勿隨意更改以避免數值差異。
* **節省時間**：由於步驟3、4採取的方式為迴圈，總執行時間可能要兩小時甚至更多，可以在確認邏輯正確後直接把"Data_原版"資料夾裡的全部內容複製貼上到Data資料夾，因上述步驟 2、3、4 的產出物已預先生成在 `Data_原版/` 中，可直接跳至步驟5執行鐘形分布切樣。

* **unrelated帳戶順序**：因使用了 Python 的 `set` (無序集合) 來處理帳號名單，導致每次執行的帳號排列順序都會不同，當初寫這份專案時就已經是使用set來處理了，所以幾乎不可能復現結果，但是我們有保留當初的原始順序版本，也就是"Data_原始"資料夾裡的unrelated_chunks，保證都來自於官方原始檔案，unrelated_features則是由這份chunk執行步驟4得來的，附上只是為了節省時間。
