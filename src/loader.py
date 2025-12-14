import pandas as pd
from typing import List


def load_data(file_paths: List[str]):
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Loaded {path} | Shape: {df.shape}")
        except FileNotFoundError:
            print(f"File not found at {path}")
    return dataframes


def merge_df(
    df_generation: pd.DataFrame, df_weather: pd.DataFrame, df_details: pd.DataFrame
):
    df_generation["Timestamp"] = pd.to_datetime(df_generation["Timestamp"])
    df_weather["Timestamp"] = pd.to_datetime(df_weather["Timestamp"])

    # 1. Merge Weather Data (Compound Key: CampusKey + Timestamp)
    print(f"Merge Starting. Base Shape: {df_generation.shape}")
    merged_df = pd.merge(
        left=df_generation, right=df_weather, on=["CampusKey", "Timestamp"], how="left"
    )
    print(f"After Merging Weather Data: {merged_df.shape}")

    # 2. Merge Site Details (Compound Key: CampusKey + SiteKey)
    merged_df = pd.merge(
        left=merged_df,
        right=df_details,
        on=["CampusKey", "SiteKey"],
        how="left",
        suffixes=("_base", "_site_details"),
    )
    print(f"After Merging Site Details: {merged_df.shape}")

    # Final Cleanup
    merged_df = merged_df.reset_index(drop=True)

    return merged_df
