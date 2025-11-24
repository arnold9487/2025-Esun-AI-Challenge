"""
Unrelated Features Extraction Module.

This module is designed to extract behavioral features from the chunked 'Unrelated'
account data. It iterates through each chunk file generated in the previous splitting
step, applying the exact same feature engineering logic as used for the 'Alert' and
'Predict' datasets (see `3_feature_extraction.py`). This ensures consistency across
all datasets while handling large volumes of data efficiently.
"""

import os
import pandas as pd
import numpy as np

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
CHUNKS_DIR = os.path.join(BASE_DATA_DIR, "unrelated_chunks")
OUT_DIR = os.path.join(BASE_DATA_DIR, "unrelated_features")
ALERT_REF_PATH = os.path.join(BASE_DATA_DIR, "acct_alert.csv")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

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

def extract_features_for_chunks():
    """
    Extract behavioral features for all unrelated account chunks.

    This function scans the `unrelated_chunks` directory for account and transaction
    file pairs. For each chunk, it calculates a comprehensive set of features 
    (statistical, temporal, network, etc.) identical to the logic used for the 
    training and testing sets. The results are saved as individual CSV files in 
    the `unrelated_features` directory.

    The process includes:
    1. Loading the Alert reference list for network feature calculation.
    2. Iterating through sorted chunk files.
    3. Performing feature extraction with strict floating-point reproducibility.
    4. Saving the feature matrix for each chunk.
    """
    # 0. Prepare Alert Set (for alert_partner_ratio)
    alert_accts = set()
    if os.path.exists(ALERT_REF_PATH):
        try:
            df_alert = pd.read_csv(ALERT_REF_PATH)
            alert_accts = set(df_alert["acct"])
        except Exception as e:
            print(f"[Warning] Failed to read alert file: {e}")
    else:
        print(f"[Warning] Alert file not found at {ALERT_REF_PATH}")

    # 1. Scan for Chunk Files
    if not os.path.exists(CHUNKS_DIR):
        print(f"[Error] Chunks directory not found: {CHUNKS_DIR}")
        return

    # Find all unrelated_accts_chunk_X.csv files
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.startswith("unrelated_accts_chunk_") and f.endswith(".csv")]
    # Sort by chunk ID to ensure sequential processing
    chunk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"Found {len(chunk_files)} chunks to process.")

    # 2. Process Each Chunk
    for acct_file_name in chunk_files:
        # Parse Chunk ID
        sid = acct_file_name.split('_')[-1].split('.')[0]
        
        acct_file = os.path.join(CHUNKS_DIR, acct_file_name)
        data_file = os.path.join(CHUNKS_DIR, f"unrelated_txn_chunk_{sid}.csv")
        out_file = os.path.join(OUT_DIR, f"unrelated_feature_{sid}.csv")

        if not os.path.exists(acct_file) or not os.path.exists(data_file):
            print(f"[Skip] Missing file pair for chunk {sid}")
            continue
            
        print(f"--- Processing Chunk {sid} ---")

        # Load Data
        acct_list = pd.read_csv(acct_file, dtype={"acct": str})
        df = pd.read_csv(data_file, dtype={"from_acct": str, "to_acct": str})
        
        # Preprocessing
        t = pd.to_datetime(df["txn_time"], format="%H:%M:%S", errors="coerce")
        df["hour"] = t.dt.hour + t.dt.minute / 60.0
        df["txn_date"] = df["txn_date"].astype(int)
        df["rate_twd"] = df["currency_type"].map(rate_map).fillna(1.0)
        df["txn_amt_twd"] = df["txn_amt"] * df["rate_twd"]

        features = []
        eps = 1e-9

        # Feature Extraction Loop (Core Logic)
        for acct in acct_list["acct"]:
            df_out = df[df["from_acct"] == acct]
            df_in  = df[df["to_acct"] == acct]

            df_out = df_out.assign(direction="out", counterparty_type=df_out["to_acct_type"])
            df_in  = df_in.assign(direction="in",  counterparty_type=df_in["from_acct_type"])
            df_all = pd.concat([df_out, df_in], ignore_index=True)

            if df_all.empty:
                features.append({"acct": acct, "label": 0}) 
                continue

            # Basic Statistics
            out_count, in_count = len(df_out), len(df_in)
            total_count = len(df_all)
            out_sum, in_sum = df_out["txn_amt_twd"].sum(), df_in["txn_amt_twd"].sum()
            out_mean = df_out["txn_amt_twd"].mean() if out_count > 0 else 0
            in_mean = df_in["txn_amt_twd"].mean() if in_count > 0 else 0
            
            # Smoothing (+1)
            mean_all = df_all["txn_amt_twd"].mean()
            max_div_mean = df_all["txn_amt_twd"].max() / (mean_all + 1)
            amount_std = df_all["txn_amt_twd"].std(ddof=1) if total_count > 1 else 0
            amount_cv = amount_std / (mean_all + 1)

            # Partner Features
            partners = set(df_out["to_acct"]).union(set(df_in["from_acct"]))
            partner_count = len(partners)
            alert_partner_ratio = len(partners & alert_accts) / (partner_count + 1)
            
            non_esun_txn = df_all[(df_all["counterparty_type"] == 2) & (df_all["is_self_txn"] != "Y")]
            denom_txn = total_count - len(df_all[df_all["is_self_txn"] == "Y"])
            non_esun_ratio_txn = len(non_esun_txn) / denom_txn if denom_txn > 0 else 0
            
            denom_amt = df_all.loc[df_all["is_self_txn"] != "Y", "txn_amt_twd"].sum()
            non_esun_ratio_amt = non_esun_txn["txn_amt_twd"].sum() / denom_amt if denom_amt > 0 else 0
            
            self_ratio = len(df_all[df_all["is_self_txn"] == "Y"]) / total_count if total_count > 0 else 0

            # Temporal Features
            active_days = df_all["txn_date"].nunique()
            active_span = int(df_all["txn_date"].max() - df_all["txn_date"].min())
            active_density = active_days / (active_span + 1) if active_days > 1 else 0
            daily_avg_txn = total_count / (active_days + 1)
            
            night_txn = df_all[(df_all["hour"] < 6) | (df_all["hour"] >= 22)]
            night_ratio = len(night_txn) / (total_count + 1)

            # Frequency Features
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

            # Channel / Currency
            channel_ratio = df_all["channel_type"].value_counts(normalize=True).to_dict()
            currency_count = df_all["currency_type"].nunique()

            # Composite Features
            has_out = 1 if out_count > 0 else 0
            inout_amt_ratio = in_sum / out_sum if has_out == 1 and out_sum > 0 else -1
            inout_cnt_ratio = in_count / (out_count + 1) if has_out == 1 else -1

            # Window Features
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

            # Assembly
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

            feature_row["label"] = 0
            features.append(feature_row)

        # Output
        result = pd.DataFrame(features).fillna(0)
        result.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Saved chunk {sid} features to {out_file}")

if __name__ == "__main__":
    extract_features_for_chunks()