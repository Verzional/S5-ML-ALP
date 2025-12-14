import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop Rows with Missing Target Variable (SolarGeneration)
    initial_rows = len(df)
    df = df.dropna(subset=["SolarGeneration"])
    print(f"Dropped {initial_rows - len(df)} rows with missing Target.")

    # 2. Handle Missing Weather Data (Linear Interpolation for small gaps)
    weather_cols = [
        "ApparentTemperature",
        "AirTemperature",
        "DewPointTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
    ]

    # Check if columns exist before processing
    existing_weather_cols = [c for c in weather_cols if c in df.columns]
    df[existing_weather_cols] = df[existing_weather_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # 3. Handle Negative Solar Generation (Sensor errors)
    # Clip negative values to 0
    df["SolarGeneration"] = df["SolarGeneration"].clip(lower=0)

    return df


def filter_valid_sites(df: pd.DataFrame) -> pd.DataFrame:
    # Remove sites with missing kWp capacity
    missing_kwp = df["kWp"].isnull().sum()
    if missing_kwp > 0:
        print(f"\nDropping {missing_kwp} rows with missing 'kWp' capacity data.")
        df = df.dropna(subset=["kWp"])

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Timestamp is in datetime Format
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Extract Standard Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["Year"] = df["Timestamp"].dt.year

    # Day / Night Indicator
    df["IsDaylight"] = (df["Hour"] >= 6) & (df["Hour"] <= 20)

    # Encode Cyclical Features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df
