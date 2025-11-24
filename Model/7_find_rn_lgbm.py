"""
Reliable Negative (RN) Discovery Module.

This module implements the 'Spy' technique for PU Learning to identify Reliable 
Negatives (RN) from the unlabeled dataset.

The process involves:
1. Treating a subset of 'Alert' (Positive) samples as known positives.
2. Using the remaining 'Alert' samples as 'Spies' within the validation set.
3. Training a LightGBM classifier to distinguish between Alerts and Unlabeled data.
4. Determining a strict threshold based on the lowest prediction score assigned 
   to the 'Spy' alerts (implying that any unlabeled sample with a score lower 
   than the "easiest-to-miss" positive is likely a true negative).
5. Selecting the bottom 10% of these low-scoring candidates as the initial set 
   of Reliable Negatives (RN).
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Input Files
ALERT_PATH = os.path.join(BASE_DATA_DIR, "alert_features.csv")
UNREL_PATH = os.path.join(BASE_DATA_DIR, "sampled_unrelate_shuffled.csv")

# Output Files
OUTPUT_RN_CSV = os.path.join(BASE_DATA_DIR, "rn_candidates.csv")
OUTPUT_UNREL_REMAINING = os.path.join(BASE_DATA_DIR, "unrelate_remaining.csv")
OUTPUT_SCORES = os.path.join(BASE_DATA_DIR, "holdout_scores.csv")
OUTPUT_SPY_SCORES = os.path.join(BASE_DATA_DIR, "spy_alert_scores.csv")

# ==========================================
# Parameters
# ==========================================
SEED = 42
N_ALERT_TRAIN = 800
N_U_TRAIN     = 3000
NEG_WEIGHT    = 0.2666

# Feature Dropping Lists (To prevent overfitting on static/high-corr features)
HIGH_CORR_DROP = [
    "active_days_ratio_30d",
    "active_days_ratio_7d",
    "non_esun_ratio_amt"
]

STATIC_DROP = [
    "out_count",
    "in_count",
    "total_count",
    "out_sum",
    "in_sum",
    "out_mean",
    "in_mean",
    "amount_std",
    "partner_count",
    "alert_partner_ratio",
    "active_days",       
    "currency_count",
    "max_dup_group"
]

def find_rn():
    """
    Execute the Reliable Negative discovery process.

    Steps:
    1. Loads Alert features and sampled Unrelated data.
    2. Removes static and highly correlated features to force the model to learn 
       behavioral patterns rather than volume metrics.
    3. Splits data into Training and Holdout sets (Spies are in Holdout).
    4. Trains a LightGBM model (Weighted).
    5. Uses the minimum score of Holdout Alerts (Spies) to set a cutoff threshold.
    6. Identifies RN candidates from the Unrelated set that fall below this threshold.
    7. Selects the bottom 10% of candidates as the final RN set.
    8. Exports the RN set and the remaining Unrelated set for subsequent iterative training.
    """
    print("--- Starting RN Discovery (LightGBM) ---")
    
    # 1. Load Data
    if not os.path.exists(ALERT_PATH) or not os.path.exists(UNREL_PATH):
        print(f"[Error] Input files missing: {ALERT_PATH} or {UNREL_PATH}")
        return

    alert = pd.read_csv(ALERT_PATH)
    unrelate_original_full = pd.read_csv(UNREL_PATH)
    unrelate = unrelate_original_full.copy()

    print(f"Loaded Alert: {len(alert)}, Unrelated: {len(unrelate)}")

    # 2. Drop Features
    all_features_to_drop = list(set(HIGH_CORR_DROP + STATIC_DROP))
    print(f"Dropping {len(all_features_to_drop)} static/high-corr features...")
    
    for f in all_features_to_drop:
        if f in alert.columns:    alert.drop(columns=f, inplace=True)
        if f in unrelate.columns: unrelate.drop(columns=f, inplace=True)

    # 3. Split Train/Holdout
    # Sample Training Sets
    alert_train = alert.sample(n=N_ALERT_TRAIN, random_state=SEED)
    unrelate_train = unrelate.sample(n=N_U_TRAIN, random_state=SEED)
    
    # Create Holdout Sets (The rest become 'Spies' and validation samples)
    alert_hold  = alert.drop(alert_train.index)
    unrelate_hold  = unrelate.drop(unrelate_train.index)

    # Assign Labels for Training (1=Positive, 0=Unlabeled/Negative)
    alert_train = alert_train.copy();    alert_train["label"] = 1
    unrelate_train = unrelate_train.copy(); unrelate_train["label"] = 0
    train = pd.concat([alert_train, unrelate_train]) 

    # Assign Labels for Holdout
    alert_hold = alert_hold.copy();       alert_hold["label"] = 1
    unrelate_hold = unrelate_hold.copy(); unrelate_hold["label"] = 0
    holdout = pd.concat([alert_hold, unrelate_hold])

    # Prepare Features
    num_cols = list(train.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in num_cols if c != "label"]
    print(f"Training with {len(feature_cols)} features.")

    # Set Weights (Down-weight unlabeled samples)
    weights = train['label'].apply(lambda x: 1.0 if x == 1 else NEG_WEIGHT).values

    # 4. Train LightGBM
    params = {
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
        "seed": SEED
    }

    dtrain = lgb.Dataset(train[feature_cols], label=train["label"], weight=weights)
    model = lgb.train(params, dtrain, num_boost_round=300)

    # 5. Predict Holdout
    holdout = holdout.copy()
    holdout["pred_score"] = model.predict(holdout[feature_cols])

    # 6. Determine Threshold
    # Logic: Find the lowest score among the 'Spy' Alerts (Holdout P)
    # Any Unrelated sample scoring lower than this is very likely NOT a positive.
    alert_scores_series = holdout.loc[holdout["label"] == 1, "pred_score"]
    threshold = float(alert_scores_series.min()) if len(alert_scores_series) > 0 else 0.0
    
    print(f"Threshold (Min Score of Spy Alerts): {threshold:.6f}")

    # Export Spy Scores (Optional Debugging)
    spy_alerts = holdout[holdout["label"] == 1].copy()
    spy_alerts.to_csv(OUTPUT_SPY_SCORES, index=False, encoding="utf-8-sig")

    # 7. Filter & Select RN
    # Candidates: Unrelated samples with score < Threshold
    candidates = holdout[(holdout["label"] == 0) & (holdout["pred_score"] < threshold)].copy()
    print(f"Candidates (< threshold): {len(candidates)}")

    # Selection: Take the bottom 10% (most confident negatives)
    if len(candidates) > 0:
        cutoff = float(np.percentile(candidates["pred_score"], 10))
        RN = candidates[candidates["pred_score"] <= cutoff].copy()
    else:
        RN = pd.DataFrame()

    print(f"Selected RN (Bottom 10% of candidates): {len(RN)}")

    # 8. Export Results
    # Retrieve full features from the original dataframe (including dropped ones)
    if not RN.empty:
        RN_final_output = unrelate_original_full.loc[RN.index].copy()
    else:
        RN_final_output = pd.DataFrame(columns=unrelate_original_full.columns)

    RN_final_output.to_csv(OUTPUT_RN_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved RN to {OUTPUT_RN_CSV}")

    # Retrieve remaining Unrelated samples (Original - RN)
    unrelate_remaining = unrelate_original_full.drop(RN.index).copy()
    unrelate_remaining.to_csv(OUTPUT_UNREL_REMAINING, index=False, encoding="utf-8-sig")
    print(f"Saved Remaining Unrelated to {OUTPUT_UNREL_REMAINING}")

    # Save validation scores
    holdout.to_csv(OUTPUT_SCORES, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    find_rn()