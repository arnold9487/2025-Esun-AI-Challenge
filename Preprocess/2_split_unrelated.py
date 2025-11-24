"""
Unrelated Data Splitting Module.

This script divides the large dataset of 'Unrelated' accounts into smaller, 
manageable chunks. This step is crucial for efficient feature extraction 
in later stages, allowing for batch processing and reducing memory usage.
"""

import pandas as pd
import os
import math

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
ACCT_UNRELATED_PATH = os.path.join(BASE_DATA_DIR, "acct_unrelated.csv")
TOTAL_TXN_PATH = os.path.join(BASE_DATA_DIR, "unrelated_transaction.csv")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "unrelated_chunks")

# Optimal chunk size determined for performance
OPTIMAL_CHUNK_SIZE = 6833

def parse_and_extract_unrelated_data(acct_path, txn_path, output_dir, chunk_size=OPTIMAL_CHUNK_SIZE):
    """
    Split unrelated accounts into chunks and extract their transactions.

    This function reads the full list of unrelated accounts and the full 
    unrelated transaction log. It then iterates through the account list, 
    slicing it into chunks of size `chunk_size`, extracts the corresponding 
    transactions, and saves them as separate CSV files.

    Args:
        acct_path (str): Path to the CSV file containing unrelated account IDs.
        txn_path (str): Path to the CSV file containing all unrelated transactions.
        output_dir (str): Directory where the chunked files will be saved.
        chunk_size (int, optional): The number of accounts per chunk. Defaults to 6833.

    Returns:
        None
    """
    if not os.path.exists(acct_path) or not os.path.exists(txn_path):
        print(f"[Error] Input files not found: {acct_path} or {txn_path}")
        return

    # Read accounts
    try:
        df_accts = pd.read_csv(acct_path)
        acct_list = df_accts['acct'].tolist()
        total_accts = len(acct_list)
        print(f"Successfully loaded {total_accts} unrelated accounts.")
    except Exception as e:
        print(f"[Error] Failed to read account file: {e}")
        return

    # Calculate number of chunks
    num_chunks = math.ceil(total_accts / chunk_size)
    
    print(f"--- Splitting Configuration ---")
    print(f"Total Accounts: {total_accts}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Total Chunks: {num_chunks}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all transactions
    # Note: This requires sufficient memory.
    print("Loading transaction data (this might take a while)...")
    try:
        df_txn_all = pd.read_csv(txn_path)
        print(f"Successfully loaded {len(df_txn_all)} transactions.")
    except Exception as e:
        print(f"[Error] Failed to read transaction file: {e}")
        return

    # Process each chunk
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_accts)
        
        chunk_accts = acct_list[start_idx:end_idx]
        
        if not chunk_accts:
            continue

        # Define output filenames
        acct_output_name = f"unrelated_accts_chunk_{i+1}.csv"
        acct_output_path = os.path.join(output_dir, acct_output_name)
        txn_output_name = f"unrelated_txn_chunk_{i+1}.csv"
        txn_output_path = os.path.join(output_dir, txn_output_name)
        
        print(f"Processing Chunk {i+1}/{num_chunks} (Accounts: {len(chunk_accts)})...")
        
        # 1. Save Account Chunk
        pd.DataFrame({'acct': chunk_accts}).to_csv(acct_output_path, index=False)

        # 2. Filter Transactions for this Chunk
        df_chunk_txn = df_txn_all[
            df_txn_all['from_acct'].isin(chunk_accts) |
            df_txn_all['to_acct'].isin(chunk_accts)
        ].copy()

        # 3. Save Transaction Chunk
        df_chunk_txn.to_csv(txn_output_path, index=False)

    print("All chunks have been successfully processed and saved.")

if __name__ == "__main__":
    parse_and_extract_unrelated_data(ACCT_UNRELATED_PATH, TOTAL_TXN_PATH, OUTPUT_DIR)