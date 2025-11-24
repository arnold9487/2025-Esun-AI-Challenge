"""
Data Shuffling and Finalization Module.

This module performs a random shuffle on the sampled unrelated dataset.
Shuffling is a standard preprocessing step to ensure that the order of data 
does not introduce any bias during subsequent model training or splitting 
processes.
"""

import pandas as pd
import os

# ==========================================
# File Path Configuration
# ==========================================
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
INPUT_PATH = os.path.join(BASE_DATA_DIR, "sampled_unrelate.csv")
OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "sampled_unrelate_shuffled.csv")

def shuffle_data():
    """
    Shuffle the sampled unrelated data and save the final dataset.

    Reads the `sampled_unrelate.csv` file, applies a random permutation
    to the rows using a fixed random seed for reproducibility, and saves
    the result to `sampled_unrelate_shuffled.csv`.

    Returns:
        None
    """
    if not os.path.exists(INPUT_PATH):
        print(f"[Error] Input file not found: {INPUT_PATH}")
        return

    print("Reading data...")
    df = pd.read_csv(INPUT_PATH)
    
    print("Shuffling...")
    # frac=1 returns all rows in random order
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Shuffled data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    shuffle_data()