"""
Data Generation Module for 2025 E.SUN AI Open Competition.

This module is responsible for the initial data processing pipeline.
It loads the raw dataset, separates transactions into specific groups 
(Alert, Predict, and Unrelated), and generates the necessary intermediate 
CSV files for subsequent steps.

Flow:
1. Load raw data (acct_alert, acct_predict, acct_transaction).
2. Extract transactions specifically for Alert and Predict accounts.
3. Identify 'Unrelated' accounts (Type 1 accounts not in Alert or Predict sets).
4. Export the unrelated accounts list and their full transaction history.
"""

import os
import pandas as pd

# ==========================================
# File Path Configuration
# ==========================================
# Base data directory relative to this script
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
OUTPUT_DIR = BASE_DATA_DIR

def load_data(data_dir):
    """
    Load raw datasets from the specified directory.

    Args:
        data_dir (str): The relative path to the raw data directory.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
               (df_alert, df_predict, df_txn).

    Raises:
        FileNotFoundError: If any required CSV file is missing.
    """
    print(f"Loading data from {data_dir}...")
    alert_path = os.path.join(data_dir, 'acct_alert.csv')
    predict_path = os.path.join(data_dir, 'acct_predict.csv')
    txn_path = os.path.join(data_dir, 'acct_transaction.csv')

    if not all(os.path.exists(p) for p in [alert_path, predict_path, txn_path]):
        raise FileNotFoundError(f"Missing one of the required files in {data_dir}")

    df_alert = pd.read_csv(alert_path)
    df_predict = pd.read_csv(predict_path)
    df_txn = pd.read_csv(txn_path)
    
    return df_alert, df_predict, df_txn

def extract_transactions(df_txn, account_set, output_path):
    """
    Extract transactions related to a specific set of accounts and save to CSV.

    Filters the transaction dataframe to include only rows where either the 
    'from_acct' or 'to_acct' exists in the provided account set.

    Args:
        df_txn (pd.DataFrame): The dataframe containing all transactions.
        account_set (set): A set of account IDs to filter by.
        output_path (str): The file path where the filtered CSV will be saved.
    """
    print(f"Extracting transactions for {len(account_set)} accounts...")
    filtered_txn = df_txn[
        (df_txn['from_acct'].isin(account_set)) |
        (df_txn['to_acct'].isin(account_set))
    ]
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    filtered_txn.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_txn)} transactions to {output_path}")

def identify_unrelated_accounts(df_txn, alert_accts, predict_accts):
    """
    Identify 'Unrelated' accounts (Type 1 accounts).

    Finds accounts that are Type 1 (individual) but are NOT present in either
    the Alert set or the Predict set.

    Args:
        df_txn (pd.DataFrame): The transaction dataframe.
        alert_accts (set): A set of alert account IDs.
        predict_accts (set): A set of predict account IDs.

    Returns:
        set: A set of unique account IDs classified as unrelated.
    """
    print("Identifying unrelated accounts...")
    # Union of alert and predict sets to exclude
    excluded_accts = alert_accts.union(predict_accts)

    # Find all Type 1 accounts in both 'from' and 'to' columns
    from_accts_type1 = set(df_txn[df_txn['from_acct_type'] == 1]['from_acct'])
    to_accts_type1 = set(df_txn[df_txn['to_acct_type'] == 1]['to_acct'])
    total_type1_accts = from_accts_type1.union(to_accts_type1)

    # Subtract excluded accounts to find unrelated ones
    unrelated_accts = total_type1_accts - excluded_accts
    return unrelated_accts

def main():
    """
    Main execution function for the data generation process.
    """
    # 1. Load Data
    try:
        df_alert, df_predict, df_txn = load_data(BASE_DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        return
    
    alert_accts = set(df_alert['acct'])
    predict_accts = set(df_predict['acct'])

    # 2. Extract Alert & Predict Transactions
    # Saves filtered transactions for model input
    extract_transactions(df_txn, alert_accts, os.path.join(OUTPUT_DIR, "alert_data.csv"))
    extract_transactions(df_txn, predict_accts, os.path.join(OUTPUT_DIR, "predict_data.csv"))

    # 3. Identify Unrelated Accounts
    unrelated_accts = identify_unrelated_accounts(df_txn, alert_accts, predict_accts)
    
    # Save Unrelated Accounts List
    unrelated_df = pd.DataFrame(list(unrelated_accts), columns=['acct'])
    unrelated_acct_path = os.path.join(OUTPUT_DIR, "acct_unrelated.csv")
    unrelated_df.to_csv(unrelated_acct_path, index=False)
    print(f"Saved {len(unrelated_accts)} unrelated accounts to {unrelated_acct_path}")

    # 4. Extract Unrelated Transactions
    # Generates a large transaction file for the unrelated accounts, used for subsequent splitting
    extract_transactions(df_txn, unrelated_accts, os.path.join(OUTPUT_DIR, "unrelated_transaction.csv"))

if __name__ == "__main__":
    main()