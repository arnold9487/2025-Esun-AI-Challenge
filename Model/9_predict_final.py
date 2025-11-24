"""
Final Model Training and Prediction Module.

This module performs the final training phase using the fully expanded datasets
(Final P and Final N) generated from the iterative learning step. It trains
a LightGBM model on the complete dataset and predicts the probability of 
money laundering for the test accounts ('Predict' dataset).

The module then generates submission files by thresholding the prediction scores
at various top-k percentiles (e.g., Top 2%, 5%, 10%), as required by the 
competition submission format.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Input Paths (Final Training Sets)
P_FINAL_PATH = os.path.join(BASE_DATA_DIR, "final_P_all_features.csv")
N_FINAL_PATH = os.path.join(BASE_DATA_DIR, "final_N_all_features.csv")

# Input Path (Prediction Set)
PREDICT_FILE_PATH = os.path.join(BASE_DATA_DIR, "predict_features.csv")

# Output Base Path
PREDICT_OUTPUT_BASE = os.path.join(BASE_DATA_DIR, "predictions_official")

# ==========================================
# Parameters
# ==========================================
PERCENTAGES_TO_LABEL = [2, 5, 10, 15, 50]

# Features to drop (Must match training configuration)
FEATURES_TO_DROP = [
    'total_count', 'in_sum', 'amount_std', 'out_count', 'in_count', 
    'non_esun_ratio_amt', 'out_sum', 'out_mean', 'in_mean', 
    'currency_count', 'max_dup_group', 'alert_partner_ratio', 
    'active_days', 'active_days_ratio_30d', 'active_days_ratio_7d', 
    'partner_count'
]

lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l2": 2.0,
    "metric": "auc",
    "verbosity": -1,
    "seed": 42
}

def drop_features(df, drop_list):
    """
    Safely remove specified features from a DataFrame.
    """
    features_exist = [col for col in drop_list if col in df.columns]
    return df.drop(columns=features_exist)

def main():
    """
    Main execution function for final training and prediction.
    
    1. Loads the final expanded Positive (P) and Negative (N) datasets.
    2. Combines them into a full training set.
    3. Loads the test set (Predict features).
    4. Removes static/unstable features.
    5. Trains the final LightGBM model.
    6. Generates predictions and saves multiple submission files based on 
       top-k percentiles.
    """
    print("--- Starting Final Prediction Phase ---")

    if not all(os.path.exists(p) for p in [P_FINAL_PATH, N_FINAL_PATH, PREDICT_FILE_PATH]):
        print(f"[Error] Missing input files in {BASE_DATA_DIR}")
        return

    # 1. Load Final Training Data
    df_P = pd.read_csv(P_FINAL_PATH)
    df_N = pd.read_csv(N_FINAL_PATH)
    
    df_P['label'] = 1
    df_N['label'] = 0
    
    train_full = pd.concat([df_P, df_N], ignore_index=True)
    print(f"Loaded Final Train Set: P={len(df_P)}, N={len(df_N)}")

    # 2. Load Prediction Data
    df_predict = pd.read_csv(PREDICT_FILE_PATH)
    print(f"Loaded Predict Set: {len(df_predict)}")
    
    predict_accts = df_predict['acct'] if 'acct' in df_predict.columns else df_predict.index

    # 3. Preprocessing (Drop Static Features)
    print("Dropping static features...")
    train_behav = drop_features(train_full, FEATURES_TO_DROP)
    predict_behav = drop_features(df_predict, FEATURES_TO_DROP)

    # 4. Prepare Training
    numeric_cols = list(train_behav.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in ['label', 'score', 'pred_score']]
    print(f"Training features: {len(feature_cols)}")

    # Calculate Scale Weight
    if len(df_P) > 0 and len(df_N) > 0:
        lgb_params['scale_pos_weight'] = len(df_N) / len(df_P)
        print(f"Set scale_pos_weight: {lgb_params['scale_pos_weight']:.4f}")
    
    dtrain = lgb.Dataset(train_behav[feature_cols], label=train_behav['label'])

    # 5. Train Final Model
    print("Training LightGBM...")
    model = lgb.train(lgb_params, dtrain, num_boost_round=300)

    # 6. Predict
    print("Predicting...")
    predictions = model.predict(predict_behav[feature_cols])

    # 7. Generate Submission Files
    df_scores = pd.DataFrame({'acct': predict_accts, 'pred_score': predictions})
    
    print("\n--- Generating Submission Files ---")
    for p in PERCENTAGES_TO_LABEL:
        quantile = 1.0 - (p / 100.0)
        threshold = df_scores['pred_score'].quantile(quantile)
        
        print(f"Processing Top {p}% (Threshold: {threshold:.6f})")
        
        labels = np.where(df_scores['pred_score'] >= threshold, 1, 0)
        df_out = pd.DataFrame({'acct': df_scores['acct'], 'label': labels})
        
        out_name = f"{PREDICT_OUTPUT_BASE}_top{p}percent.csv"
        df_out.to_csv(out_name, index=False, encoding='utf-8-sig')
        print(f"  Saved: {out_name}")
        print(f"  Count: {df_out['label'].value_counts().to_dict()}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()