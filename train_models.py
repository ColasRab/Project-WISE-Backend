import pandas as pd
import joblib
from prophet import Prophet
from tqdm import tqdm
import os
import numpy as np

DATA_DIR = "python\\data\\raw"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def derive_chance_of_rain_vectorized(df):
    """Vectorized version of chance_of_rain calculation (no memory overflow)."""
    et0 = df["et0_fao_evapotranspiration"].fillna(0)
    temp = df["temperature_2m"].fillna(0)
    rh = df["relative_humidity_2m"].fillna(0)

    rain_score = (rh * 0.6) + (np.maximum(0, 40 - temp) * 0.3) + (np.maximum(0, 5 - et0) * 2)
    return np.clip(rain_score, 0, 100)

def train_per_city(hourly_csv, cities_csv):
    hourly_path = os.path.join(DATA_DIR, hourly_csv)
    cities_path = os.path.join(DATA_DIR, cities_csv)

    df = pd.read_csv(hourly_path)
    cities = pd.read_csv(cities_path)

    df["city_name"] = df["city_name"].str.strip().str.lower()
    cities["city_name"] = cities["city_name"].str.strip().str.lower()

    df = df.merge(cities, on="city_name", how="left")

    # 🚀 Vectorized rain percentage (fast + memory-safe)
    print("🌧️ Deriving chance_of_rain...")
    df["chance_of_rain"] = derive_chance_of_rain_vectorized(df)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(["city_name", "datetime"], inplace=True)

    target_vars = [
        "chance_of_rain",
        "wind_speed_10m",
        "apparent_temperature",
        "relative_humidity_2m",
    ]

    city_names = df["city_name"].unique()
    print(f"📊 Found {len(city_names)} cities to train Prophet models for.\n")

    for city in tqdm(city_names, desc="Training per city"):
        city_df = df[df["city_name"] == city].copy()
        if city_df.empty:
            print(f"⚠️ Skipping {city} (no data)")
            continue

        for target in target_vars:
            model_df = city_df[["datetime", target]].rename(columns={"datetime": "ds", target: "y"})
            regressors = [v for v in target_vars if v != target]

            for reg in regressors:
                if reg in city_df.columns:
                    model_df[reg] = city_df[reg]

            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode="additive"
            )

            for reg in regressors:
                model.add_regressor(reg)

            try:
                model.fit(model_df)
                model_path = os.path.join(OUTPUT_DIR, f"{city}_{target}_prophet.pkl")
                joblib.dump(model, model_path)
                print(f"✅ Saved {city.title()} → {target} model")
            except Exception as e:
                print(f"❌ Skipping {city} {target}: {e}")

    print("\n🎯 All Prophet models trained and saved successfully.")

if __name__ == "__main__":
    train_per_city("hourly_data_combined_2020_to_2023.csv", "cities.csv")
