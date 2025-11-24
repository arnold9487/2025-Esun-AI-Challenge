"""
Iterative PU Learning (Positive-Unlabeled) Training Module.

This module implements the core iterative expansion logic for Positive-Unlabeled learning.
It starts with an initial set of labeled positive samples (Alerts), reliable negatives (RNs),
and a large pool of unlabeled data (Unrelated). Through multiple iterations, it trains a
classifier to identify high-confidence positive and negative samples from the unlabeled pool,
gradually expanding the training set.

Key mechanisms include:
- **Dynamic Gatekeeper**: Splits the current training pool into 'Train' and 'Guard' sets 
  to prevent overfitting and establish reliable confidence thresholds.
- **Percentile Thresholding**: Uses the score distribution of the 'Guard' set to determine 
  adaptive thresholds for selecting new samples.
- **Safety Brake**: Stops expansion if the model becomes confused (i.e., when negative 
  and positive thresholds overlap).
- **Weighted Loss**: Applies class weights (scale_pos_weight) to handle class imbalance 
  dynamically during each iteration.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Input Paths
P_PATH = os.path.join(BASE_DATA_DIR, "alert_features.csv")         # Initial Positives (Alerts)
N_PATH = os.path.join(BASE_DATA_DIR, "rn_candidates.csv")          # Initial Negatives (RNs)
U_PATH = os.path.join(BASE_DATA_DIR, "unrelate_remaining.csv")     # Unlabeled Pool (Remaining Unrelated)

# Output Paths
OUTPUT_P_FINAL = os.path.join(BASE_DATA_DIR, "final_P_all_features.csv")
OUTPUT_N_FINAL = os.path.join(BASE_DATA_DIR, "final_N_all_features.csv")

# ==========================================
# Parameters
# ==========================================
SEED = 42
GUARD_SPLIT_RATIO = 0.20
EXPANSION_RATE = 0.10
MAX_ITERATIONS = 100

P_THRESHOLD_PERCENTILE = 75
N_THRESHOLD_PERCENTILE = 25

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
    "seed": SEED
}

def drop_features(df, drop_list):
    """
    Safely remove specified features from a DataFrame.

    Args:
        df (pd.DataFrame): The input dataframe.
        drop_list (list): List of column names to drop.

    Returns:
        pd.DataFrame: The dataframe with specified columns removed (if they existed).
    """
    features_exist = [col for col in drop_list if col in df.columns]
    return df.drop(columns=features_exist)

def main():
    """
    Main execution function for iterative PU learning.
    
    Orchestrates the iterative loop:
    1. Loads initial datasets.
    2. Prepares behavioral features (dropping static ones).
    3. Initializes training pools.
    4. Runs the expansion loop (Train -> Threshold -> Select -> Expand).
    5. Saves the final expanded Positive and Negative datasets.
    """
    print("--- Starting Iterative PU Learning ---")

    if not all(os.path.exists(p) for p in [P_PATH, N_PATH, U_PATH]):
        print("[Error] Missing input files.")
        return

    # 1. Load Data
    P_original = pd.read_csv(P_PATH)
    N_original = pd.read_csv(N_PATH)
    U_original = pd.read_csv(U_PATH)
    print(f"Loaded: P={len(P_original)}, N={len(N_original)}, U={len(U_original)}")

    # 2. Prepare Behavioral Features (for training)
    P_behav = drop_features(P_original, FEATURES_TO_DROP)
    N_behav = drop_features(N_original, FEATURES_TO_DROP)
    U_behav = drop_features(U_original, FEATURES_TO_DROP)

    numeric_cols = list(U_behav.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in ['label', 'score']]
    print(f"Training with {len(feature_cols)} features.")
    
    # Track columns
    full_cols = list(P_original.columns)
    behav_cols = list(P_behav.columns)

    # 3. Initialize Pools
    # Training pools (behavioral features only)
    P_pool_train = P_behav.copy()
    N_pool_train = N_behav.copy()
    
    # Expansion pool (Unlabeled)
    Pool_full = U_original.copy() # Keeps full features for final output
    Pool_behav = U_behav.copy()   # Used for prediction
    
    # Final Result Containers
    Final_P = P_original.copy()
    Final_N = N_original.copy()

    # 4. Iteration Loop
    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")
        
        # (a) Dynamic Gatekeeper Split
        if len(P_pool_train) < 10 or len(N_pool_train) < 10:
            Train_P, Guard_P = P_pool_train, P_pool_train
            Train_N, Guard_N = N_pool_train, N_pool_train
        else:
            Train_P, Guard_P = train_test_split(P_pool_train, test_size=GUARD_SPLIT_RATIO, random_state=SEED+i)
            Train_N, Guard_N = train_test_split(N_pool_train, test_size=GUARD_SPLIT_RATIO, random_state=SEED+i)

        # (b) Construct Training Set
        Train_P = Train_P.copy(); Train_P['label'] = 1
        Train_N = Train_N.copy(); Train_N['label'] = 0
        train_df = pd.concat([Train_P, Train_N], ignore_index=True)
        
        # (c) Balance Weights
        curr_params = lgb_params.copy()
        if len(Train_P) > 0 and len(Train_N) > 0:
            curr_params['scale_pos_weight'] = len(Train_N) / len(Train_P)
            print(f"  Training with scale_pos_weight: {curr_params['scale_pos_weight']:.4f}")
        else:
            print("  Training set empty/imbalanced. Stopping.")
            break

        # (d) Train Model
        dtrain = lgb.Dataset(train_df[feature_cols], label=train_df['label'])
        model = lgb.train(curr_params, dtrain, num_boost_round=300)

        # (e) Determine Thresholds using Gatekeepers
        scores_gp = model.predict(Guard_P[feature_cols])
        scores_gn = model.predict(Guard_N[feature_cols])
        
        thresh_p = np.percentile(scores_gp, P_THRESHOLD_PERCENTILE)
        thresh_n = np.percentile(scores_gn, N_THRESHOLD_PERCENTILE)
        print(f"  Thresholds -> P({P_THRESHOLD_PERCENTILE}%): {thresh_p:.6f}, N({N_THRESHOLD_PERCENTILE}%): {thresh_n:.6f}")

        # (f) Safety Brake
        if thresh_n >= thresh_p:
            print("  [Stop] Model confused (Thresh_N >= Thresh_P).")
            break

        # (g) Predict on Expansion Pool
        if Pool_behav.empty:
            print("  Pool empty.")
            break
            
        pool_scores = model.predict(Pool_behav[feature_cols])
        Pool_behav['score'] = pool_scores

        # (h) Select Candidates
        candidates_p = Pool_behav[Pool_behav['score'] > thresh_p]
        candidates_n = Pool_behav[Pool_behav['score'] < thresh_n]
        
        print(f"  Candidates > P_Thresh: {len(candidates_p)}, < N_Thresh: {len(candidates_n)}")
        
        n_add_p = int(len(candidates_p) * EXPANSION_RATE)
        n_add_n = int(len(candidates_n) * EXPANSION_RATE)
        
        if n_add_p == 0 and n_add_n == 0:
            print("  No new samples selected.")
            break
            
        # Select Top/Bottom Scores
        new_p_idx = candidates_p.nlargest(n_add_p, 'score').index
        new_n_idx = candidates_n.nsmallest(n_add_n, 'score').index
        
        print(f"  Adding {len(new_p_idx)} P and {len(new_n_idx)} N.")

        # (i) Update Final Output (Full Features)
        Final_P = pd.concat([Final_P, Pool_full.loc[new_p_idx]])
        Final_N = pd.concat([Final_N, Pool_full.loc[new_n_idx]])

        # (j) Update Training Pool (Behavioral Features)
        P_pool_train = pd.concat([P_pool_train, Pool_behav.loc[new_p_idx, behav_cols]])
        N_pool_train = pd.concat([N_pool_train, Pool_behav.loc[new_n_idx, behav_cols]])

        # (k) Remove from Expansion Pool
        processed_idx = new_p_idx.union(new_n_idx)
        Pool_full.drop(processed_idx, inplace=True)
        Pool_behav.drop(processed_idx, inplace=True)

    # 5. Output
    print("\n--- Iterative Training Complete ---")
    if 'label' in Final_P.columns: Final_P.drop(columns='label', inplace=True)
    if 'label' in Final_N.columns: Final_N.drop(columns='label', inplace=True)
    
    Final_P.to_csv(OUTPUT_P_FINAL, index=False, encoding='utf-8-sig')
    Final_N.to_csv(OUTPUT_N_FINAL, index=False, encoding='utf-8-sig')
    print(f"Saved Final P: {OUTPUT_P_FINAL} ({len(Final_P)})")
    print(f"Saved Final N: {OUTPUT_N_FINAL} ({len(Final_N)})")

if __name__ == "__main__":
    main()