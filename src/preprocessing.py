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
        print(
            f"\nDropping {missing_kwp} rows with missing 'kWp' capacity data."
        )
        df = df.dropna(subset=["kWp"])

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract Standard Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["Year"] = df["Timestamp"].dt.year

    # Day / Night Indicator
    df["IsDaylight"] = (df["Hour"] >= 6) & (df["Hour"] <= 20)

    return df
