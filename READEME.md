# 2025 玉山人工智慧公開挑戰賽 - 初賽解決方案

本專案採用 **Positive-Unlabeled (PU) Learning** 框架，針對洗錢防制議題中「極度類別不平衡」與「大量未標記資料」的特性，設計了一套穩健的半監督學習流程。我們透過「動態守門員」機制與「Spy 策略」，有效地從未標記資料中挖掘出可靠的負樣本 (Reliable Negatives)。

## ⚠️ 復現性關鍵說明 (Crucial Note on Reproducibility)

為了確保評審能夠 **100% 精確復現** 我們在排行榜上的成績，本專案在程式碼設計與資料提供上做了以下兩點重要決策：

### 1. 堅持使用「迴圈式」特徵提取 (Loop-based Feature Extraction)
在特徵工程階段，我們選擇保留原始的 **迴圈迭代 (Iterative)** 實作方式，而非使用 Pandas/Numpy 的向量化 (Vectorization/Groupby) 優化。

* **原因**：我們在實驗中發現，雖然向量化運算速度較快，但在底層浮點數運算上會產生極微小的差異（約 $10^{-10}$ 等級）。
* **蝴蝶效應**：由於我們的 PU Learning 流程中，RN (Reliable Negative) 的篩選依賴於極為敏感的「最小值門檻 (Min-Threshold)」。這些微小的浮點數差異經過模型訓練放大後，會產生蝴蝶效應，導致篩選出的 RN 樣本集合發生改變，最終顯著影響預測分數。
* **結論**：為了保證特徵數值與我們訓練當時完全一致，我們**必須**使用原始的迴圈版本程式碼。

### 2. 提供 `Data_原版` 資料夾
在資料前處理的初期步驟中，因使用了 Python 的 `set` (無序集合) 來處理帳號名單，導致每次執行的帳號排列順序可能不同。這會影響後續的資料切分 (Chunking) 與隨機抽樣 (Sampling) 結果。

* **建議**：若您希望檢驗程式邏輯，可執行 `Preprocess` 中的程式碼重新生成資料。
* **強烈建議**：若您希望**完全復現**我們的最終預測機率，請直接使用 `Data_原版` 資料夾中的中間產物（特別是 `acct_unrelated.csv` 與其對應的特徵檔），這能消除所有隨機性與順序差異。(具體步驟會寫在Preprocess的README)

---

## 📂 專案資料夾結構 (Project Structure)

```text
.
├── Data/                   # 程式執行過程中生成的輸出資料與暫存檔
├── Data_原版/              # (重要) 用於確保 100% 復現結果的原始排序資料
├── Preprocess/             # 資料前處理與特徵工程程式碼
│   └── README.md           # 詳細執行順序請參閱此處
├── Model/                  # PU Learning 模型訓練與最終預測程式碼
│   └── README.md           # 詳細模型設定請參閱此處
├── README.md               # 本專案說明文件
└── requirements.txt        # 環境依賴套件清單

🛠️ 模組功能簡介
本專案將流程模組化為兩個主要階段：

1. Preprocess (前處理與特徵工程)
位於 Preprocess/ 資料夾中。此階段負責：

從原始數據中分離警示帳戶、預測帳戶與無標記(Unrelated)帳戶。

將龐大的無標記資料進行切分 (Chunking) 處理。

執行行為特徵提取 (Behavioral Feature Extraction)。

對無標記資料進行「鐘形分佈採樣 (Bell-curve Sampling)」，使其特徵分佈擬合目標資料，解決資料量級落差問題。

2. Model (模型訓練與預測)
位於 Model/ 資料夾中。此階段負責：

RN Discovery：利用 Spy 策略，從未標記資料中篩選出高可信度的負樣本。

Iterative Training：將篩選出的 RN 加入訓練集，進行多輪迭代擴展訓練。

Final Prediction：使用最終擴展的資料集訓練 LightGBM 模型，並根據 Top-K% 策略輸出提交檔案。

💻 環境需求 (Requirements)
* **Python Version**: 3.13.9

Packages:
pandas==2.3.2
numpy==2.3.3
lightgbm==4.6.0
scikit-learn==1.7.2
tqdm==4.67.1

安裝指令：
pip install -r requirements.txt
