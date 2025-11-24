"""
Unrelated Data Sampling Module.

This module samples from the extracted 'Unrelated' features to create a dataset
that statistically resembles the 'Alert' and 'Predict' populations. Specifically,
it aligns the distribution of transaction counts (`total_count`) to follow a similar
pattern (e.g., a bell curve), preventing the model from being biased by the sheer
volume of low-activity unrelated accounts.
"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
ALERT_FEAT_PATH = os.path.join(BASE_DATA_DIR, "alert_features.csv")
PREDICT_FEAT_PATH = os.path.join(BASE_DATA_DIR, "predict_features.csv")
UNRELATE_FOLDER = os.path.join(BASE_DATA_DIR, "unrelated_features")
OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "sampled_unrelate.csv")

# Parameters
COL = "total_count"  # Target column for distribution alignment
RN_TOTAL = 42016     # Target total number of unrelated samples
BIN_WIDTH = 2
SEED = 42

def sample_distribution():
    """
    Sample unrelated accounts to match the transaction count distribution of the target datasets.

    This function performs stratified sampling based on histograms:
    1. It calculates the histogram of `total_count` for the combined Alert and Predict datasets.
    2. It determines the target number of samples for each bin to match this distribution.
    3. It iterates through all unrelated feature files, sampling accounts from matching bins.
    4. Finally, it aggregates the samples and saves the balanced dataset.

    Returns:
        None
    """
    # 1. Calculate Target Distribution
    print("Reading Alert and Predict features...")
    if not os.path.exists(ALERT_FEAT_PATH) or not os.path.exists(PREDICT_FEAT_PATH):
        print("[Error] Feature files missing.")
        return

    alert = pd.read_csv(ALERT_FEAT_PATH)
    predict = pd.read_csv(PREDICT_FEAT_PATH)

    # Combine to form the target population
    ap = pd.concat([alert, predict], axis=0, ignore_index=True)

    # Create Histogram Bins
    max_val = ap[COL].max()
    bins = np.arange(1, max_val + BIN_WIDTH, BIN_WIDTH)
    hist, bin_edges = np.histogram(ap[COL], bins=bins)

    # Calculate Target Counts per Bin
    proportions = hist / hist.sum()
    target_counts = np.floor(proportions * RN_TOTAL).astype(int)

    print(f"Total bins: {len(bins)-1}")

    # 2. Sample from Unrelated Files
    unrelate_files = sorted(glob.glob(os.path.join(UNRELATE_FOLDER, "unrelated_feature_*.csv")))
    print(f"Found {len(unrelate_files)} unrelated feature files.")

    sampled_parts = []

    for file in tqdm(unrelate_files, desc="Sampling"):
        try:
            df = pd.read_csv(file)
            
            # Assign bins to current data
            df["_bin"] = np.digitize(df[COL], bin_edges) - 1

            # Stratified Sampling per Bin
            for i, n in enumerate(target_counts):
                if n <= 0: continue
                
                subset = df[df["_bin"] == i]
                if len(subset) == 0: continue
                
                # Distribute sampling quota across files
                k = int(np.ceil(n / len(unrelate_files)))
                k = min(k, len(subset))
                
                if k > 0:
                    sampled_parts.append(subset.sample(n=k, random_state=SEED))
        except Exception as e:
            print(f"[Warning] Error reading {file}: {e}")

    # 3. Aggregate and Save
    if not sampled_parts:
        print("[Error] No data sampled.")
        return

    sampled_unrelate = pd.concat(sampled_parts, axis=0, ignore_index=True).drop(columns=["_bin"])

    # Cap to exact target size if exceeded
    if len(sampled_unrelate) > RN_TOTAL:
        sampled_unrelate = sampled_unrelate.sample(n=RN_TOTAL, random_state=SEED)

    sampled_unrelate.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Sampling complete. Saved {len(sampled_unrelate)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    sample_distribution()