"""
Feature Extraction Module (Exact Reproduction Version).

This module calculates behavioral features for 'Alert' and 'Predict' accounts.
It uses a strictly sequential looping approach (account by account) to process 
transactions. While computationally slower than vectorized operations, this 
method guarantees exact reproducibility of floating-point arithmetic results, 
avoiding 'butterfly effects' caused by operation order differences in parallel 
computing.

Key Features Extracted:
- Transaction counts, sums, means (in/out).
- Statistical measures (std, cv, max/mean).
- Network features (partner count, alert partner ratio).
- Temporal features (active days, density, night activity).
- Channel and currency usage.
- Time-windowed statistics (7 days, 30 days).
"""

import os
import pandas as pd
import numpy as np

# ==========================================
# File Path and Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Configuration for different tasks (Alert vs Predict)
TASKS = [
    {
        "name": "Alert",
        "acct_path": os.path.join(BASE_DATA_DIR, "acct_alert.csv"),
        "txn_path":  os.path.join(BASE_DATA_DIR, "alert_data.csv"),
        "out_path":  os.path.join(BASE_DATA_DIR, "alert_features.csv"),
        "label": 1
    },
    {
        "name": "Predict",
        "acct_path": os.path.join(BASE_DATA_DIR, "acct_predict.csv"),
        "txn_path":  os.path.join(BASE_DATA_DIR, "predict_data.csv"),
        "out_path":  os.path.join(BASE_DATA_DIR, "predict_features.csv"),
        "label": 0
    }
]

# Reference file for calculating alert partner ratios
ALERT_REF_PATH = os.path.join(BASE_DATA_DIR, "acct_alert.csv")

# ==========================================
# Mappings
# ==========================================
channel_map = {
    "01": "ATM", "02": "臨櫃", "03": "行銀", "04": "網銀", "05": "語音",
    "06": "eATM", "07": "電子支付", "99": "系統排程交易", "UNK": "原始數據通路為空值"
}
channels = list(channel_map.keys())

rate_map = {
    "TWD": 1.0, "USD": 32.0, "JPY": 0.22, "CNY": 4.5, "EUR": 35.0,
    "HKD": 4.1, "AUD": 21.0, "GBP": 40.0, "CAD": 23.0, "NZD": 19.0,
    "THB": 0.9, "ZAR": 1.8, "SGD": 24.0, "CHF": 36.0, "SEK": 3.1, "MXN": 1.9
}

def extract_features_exact_loop(acct_path, txn_path, out_path, label, alert_accts_set):
    """
    Extract features using an exact reproduction looping method.

    This function iterates through each account in the account list, filters
    the corresponding transactions from the transaction data, and calculates
    a comprehensive set of behavioral features.

    Args:
        acct_path (str): Path to the account list CSV.
        txn_path (str): Path to the transaction data CSV.
        out_path (str): Path where the feature CSV will be saved.
        label (int): The label to assign to these accounts (1 for Alert, 0 for Normal/Predict).
        alert_accts_set (set): A set of known alert account IDs for ratio calculations.

    Returns:
        None
    """
    if not os.path.exists(acct_path) or not os.path.exists(txn_path):
        print(f"[Skip] File not found: {acct_path} or {txn_path}")
        return

    print(f"--- Processing: {os.path.basename(acct_path)} ---")

    # 1. Load Data
    acct_list = pd.read_csv(acct_path, dtype={"acct": str})
    df = pd.read_csv(txn_path, dtype={"from_acct": str, "to_acct": str})
    
    # 2. Preprocessing
    t = pd.to_datetime(df["txn_time"], format="%H:%M:%S", errors="coerce")
    df["hour"] = t.dt.hour + t.dt.minute / 60.0
    df["txn_date"] = df["txn_date"].astype(int)
    df["rate_twd"] = df["currency_type"].map(rate_map).fillna(1.0)
    df["txn_amt_twd"] = df["txn_amt"] * df["rate_twd"]

    features = []
    eps = 1e-9
    total_accts = len(acct_list)

    # 3. Main Loop (Account by Account)
    for i, acct in enumerate(acct_list["acct"]):
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_accts} accounts...", end='\r')

        # Filter transactions for the current account
        df_out = df[df["from_acct"] == acct]
        df_in  = df[df["to_acct"] == acct]

        df_out = df_out.assign(direction="out", counterparty_type=df_out["to_acct_type"])
        df_in  = df_in.assign(direction="in",  counterparty_type=df_in["from_acct_type"])
        df_all = pd.concat([df_out, df_in], ignore_index=True)

        if df_all.empty:
            features.append({"acct": acct, "label": label}) 
            continue

        # --- Basic Statistics ---
        out_count, in_count = len(df_out), len(df_in)
        total_count = len(df_all)
        out_sum, in_sum = df_out["txn_amt_twd"].sum(), df_in["txn_amt_twd"].sum()
        out_mean = df_out["txn_amt_twd"].mean() if out_count > 0 else 0
        in_mean = df_in["txn_amt_twd"].mean() if in_count > 0 else 0
        
        # Denominator smoothing (+1) for stability
        mean_all = df_all["txn_amt_twd"].mean()
        max_div_mean = df_all["txn_amt_twd"].max() / (mean_all + 1)
        amount_std = df_all["txn_amt_twd"].std(ddof=1) if total_count > 1 else 0
        amount_cv = amount_std / (mean_all + 1)

        # --- Partner & Network Features ---
        partners = set(df_out["to_acct"]).union(set(df_in["from_acct"]))
        partner_count = len(partners)
        alert_partner_ratio = len(partners & alert_accts_set) / (partner_count + 1)
        
        non_esun_txn = df_all[(df_all["counterparty_type"] == 2) & (df_all["is_self_txn"] != "Y")]
        denom_txn = total_count - len(df_all[df_all["is_self_txn"] == "Y"])
        non_esun_ratio_txn = len(non_esun_txn) / denom_txn if denom_txn > 0 else 0
        
        denom_amt = df_all.loc[df_all["is_self_txn"] != "Y", "txn_amt_twd"].sum()
        non_esun_ratio_amt = non_esun_txn["txn_amt_twd"].sum() / denom_amt if denom_amt > 0 else 0
        
        self_ratio = len(df_all[df_all["is_self_txn"] == "Y"]) / total_count if total_count > 0 else 0

        # --- Temporal Features ---
        active_days = df_all["txn_date"].nunique()
        active_span = int(df_all["txn_date"].max() - df_all["txn_date"].min())
        active_density = active_days / (active_span + 1) if active_days > 1 else 0
        daily_avg_txn = total_count / (active_days + 1)
        
        night_txn = df_all[(df_all["hour"] < 6) | (df_all["hour"] >= 22)]
        night_ratio = len(night_txn) / (total_count + 1)

        # --- Duplicate & Frequency Features ---
        dup_keys = ["from_acct", "to_acct", "txn_amt_twd", "txn_date", "hour"]
        dup_grp = df_all.groupby(dup_keys).size().reset_index(name="dup_count")
        dup_grp = dup_grp[dup_grp["dup_count"] > 1]
        max_dup_group = dup_grp["dup_count"].max() if not dup_grp.empty else 0
        dup_txn_ratio = dup_grp["dup_count"].sum() / total_count if not dup_grp.empty else 0

        df_all["time_block"] = df_all["txn_date"] * 24 * 12 + np.floor(df_all["hour"] * 12).astype(int)
        freq = df_all.groupby("time_block").size()
        max_txn_5min = freq.max() if not freq.empty else 0
        mean_txn_5min = freq.mean() if not freq.empty else 0
        ratio_high_5min = (freq >= 5).mean() if not freq.empty else 0

        # --- Channel & Currency ---
        channel_ratio = df_all["channel_type"].value_counts(normalize=True).to_dict()
        currency_count = df_all["currency_type"].nunique()

        # --- Composite Features ---
        has_out = 1 if out_count > 0 else 0
        inout_amt_ratio = in_sum / out_sum if has_out == 1 and out_sum > 0 else -1
        inout_cnt_ratio = in_count / (out_count + 1) if has_out == 1 else -1

        # --- Time-Windowed Features (7d, 30d) ---
        max_day = df_all["txn_date"].max()
        win_feats = {}

        for w in [7, 30]:
            subset = df_all[df_all["txn_date"] >= max_day - w + 1]
            txn_cnt_w = len(subset)
            mean_amt_w = subset["txn_amt_twd"].mean() if txn_cnt_w > 0 else 0
            std_w = subset["txn_amt_twd"].std(ddof=0) if txn_cnt_w > 1 else 0
            cv_amt_w = std_w / (mean_amt_w + eps) if txn_cnt_w > 1 else 0 

            partner_w = len(set(subset["to_acct"]).union(set(subset["from_acct"])))
            active_days_w = subset["txn_date"].nunique()
            active_span_w = int(subset["txn_date"].max() - subset["txn_date"].min()) if txn_cnt_w > 1 else 0
            active_density_w = active_days_w / (active_span_w + 1) if active_days_w > 0 else 0

            if "time_block" in subset.columns: 
                freq_w = subset.groupby("time_block").size()
                ratio_high_5min_w = (freq_w >= 5).mean() if not freq_w.empty else 0
            else:
                ratio_high_5min_w = 0

            win_feats[f"txn_cnt_ratio_{w}d"]        = txn_cnt_w / (total_count + 1)
            win_feats[f"mean_amt_ratio_{w}d"]       = mean_amt_w / (mean_all + eps)
            win_feats[f"cv_amt_ratio_{w}d"]         = cv_amt_w / (amount_cv + eps) if amount_cv > 0 else 0
            win_feats[f"partner_count_ratio_{w}d"]  = partner_w / (partner_count + 1)
            win_feats[f"active_days_ratio_{w}d"]    = active_days_w / (active_days + 1)
            win_feats[f"active_density_ratio_{w}d"] = active_density_w / (active_density + eps) if active_density > 0 else 0
            win_feats[f"ratio_high_5min_ratio_{w}d"] = ratio_high_5min_w / (ratio_high_5min + eps) if ratio_high_5min > 0 else 0

        # --- Assemble Features ---
        feature_row = {
            "acct": acct,
            "out_count": out_count, "in_count": in_count, "total_count": total_count,
            "out_sum": out_sum, "in_sum": in_sum, "out_mean": out_mean, "in_mean": in_mean,
            "max_div_mean": max_div_mean, "amount_std": amount_std, "amount_cv": amount_cv,
            "partner_count": partner_count, "alert_partner_ratio": alert_partner_ratio,
            "non_esun_ratio_txn": non_esun_ratio_txn, "non_esun_ratio_amt": non_esun_ratio_amt,
            "self_ratio": self_ratio, "active_days": active_days, "active_density": active_density,
            "daily_avg_txn": daily_avg_txn, "night_ratio": night_ratio,
            "currency_count": currency_count, "inout_amt_ratio": inout_amt_ratio,
            "inout_cnt_ratio": inout_cnt_ratio, "has_out": has_out,
            "max_dup_group": max_dup_group, "dup_txn_ratio": dup_txn_ratio,
            "max_txn_5min": max_txn_5min, "mean_txn_5min": mean_txn_5min,
            "ratio_high_5min": ratio_high_5min,
            **win_feats
        }

        for ch in channels:
            ch_name = channel_map.get(ch, "未知通路")
            feature_row[f"channel_ratio_{ch_name}"] = channel_ratio.get(ch, 0.0)

        feature_row["label"] = label
        features.append(feature_row)

    print(f"\nFinished processing {os.path.basename(acct_path)}")
    
    # 4. Save Output
    result = pd.DataFrame(features).fillna(0)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Features saved to {out_path}")

if __name__ == "__main__":
    # Load Alert Set for Reference
    alert_set = set()
    if os.path.exists(ALERT_REF_PATH):
        try:
            alert_set = set(pd.read_csv(ALERT_REF_PATH, dtype={"acct": str})["acct"])
        except Exception as e:
            print(f"[Warning] Failed to load alert reference: {e}")
    else:
        print(f"[Warning] Alert reference file not found at {ALERT_REF_PATH}")

    # Process Tasks
    for task in TASKS:
        extract_features_exact_loop(
            task["acct_path"],
            task["txn_path"],
            task["out_path"],
            task["label"],
            alert_set
        )