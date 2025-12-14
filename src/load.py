import pandas as pd
from typing import List

def load_data(file_paths: List[str]):
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Loaded {path}. Shape: {df.shape}")
        except FileNotFoundError:
            print(f"ERROR: File not found at {path}")
    return dataframes

def merge_dataframes(
    df_campus: pd.DataFrame,
    df_site_only: pd.DataFrame,
    df_main_A: pd.DataFrame,
    df_main_B: pd.DataFrame,
):
    """
    Merges four dataframes using a hierarchical approach:
    1. Establish the main link table (CampusKey & SiteKey).
    2. Merge Campus data using CampusKey.
    3. Merge Site data using SiteKey.

    Returns: The single merged pandas DataFrame.
    """

    # --- Step 1: Establish the Main Link Table and Merge Main B ---
    # Use df_main_A as the starting point (Left) since it has both keys.
    print(f"Starting merge. Base shape: {df_main_A.shape}")

    # Merge df_main_B (using both keys for a strong link)
    merged_df = pd.merge(
        left=df_main_A,
        right=df_main_B,
        on=["CampusKey", "SiteKey"],  # Merging on BOTH keys
        how="left",
    )
    print(f"After merging Main B: {merged_df.shape}")

    # --- Step 2: Merge the Campus-only Data ---
    # Merge df_campus onto the current result using only CampusKey
    merged_df = pd.merge(
        left=merged_df,
        right=df_campus,
        on="CampusKey",
        how="left",
        suffixes=(
            "_main",
            "_campus",
        ),  # Good practice to differentiate same-named columns
    )
    print(f"After merging Campus data: {merged_df.shape}")

    # --- Step 3: Merge the Site-only Data ---
    # Merge df_site_only onto the current result using only SiteKey
    # This assumes SiteKey values are unique across all campuses in df_site_only.
    merged_df = pd.merge(
        left=merged_df,
        right=df_site_only,
        on="SiteKey",
        how="left",
        suffixes=("_base", "_site"),
    )
    print(f"After merging Site data: {merged_df.shape}")

    # Final cleanup (optional, but useful if keys were used as indices)
    merged_df = merged_df.reset_index(drop=True)

    return merged_df