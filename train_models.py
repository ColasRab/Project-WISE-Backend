"""
Professional Weather Forecasting Models
Using LSTM, XGBoost, and LightGBM - the industry standard
"""
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

# Optional: Uncomment if you want to use LSTM
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

DATA_DIR = ""
OUTPUT_DIR = "models_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# FEATURE ENGINEERING
# ========================================
def create_weather_features(df):
    """Create time-based and lag features for weather prediction"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    # Cyclical encoding (important for time!)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_lag_features(df, columns, lags=[1, 3, 6, 12, 24]):
    """Create lag features (past values) - KEY for time series!"""
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    # Rolling statistics
    for col in columns:
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
        df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6, min_periods=1).std()
        df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
    
    return df

def calculate_real_rain_chance(df):
    """Calculate rain chance from actual data"""
    if 'precipitation' in df.columns:
        # Historical rain occurrence
        has_rain = (df['precipitation'] > 0.1).astype(float)
        rain_chance = has_rain.rolling(window=168, min_periods=24).mean() * 100
        
        # Combine with current conditions
        rh = df["relative_humidity_2m"].fillna(50)
        humidity_factor = (rh - 50) / 2
        
        rain_chance = rain_chance + humidity_factor
        return rain_chance.clip(0, 100)
    else:
        # Fallback
        rh = df["relative_humidity_2m"].fillna(50)
        return ((rh - 30) * 1.2).clip(0, 100)

# ========================================
# MODEL 1: XGBoost (Best for tabular data)
# ========================================
def train_xgboost_model(X_train, y_train, X_val, y_val, target_name):
    """Train XGBoost model - Industry standard for weather"""
    
    # Realistic bounds for each target
    bounds = {
        'wind_speed_10m': (0, 30),
        'apparent_temperature': (15, 42),
        'relative_humidity_2m': (20, 100),
        'chance_of_rain': (0, 100)
    }
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'  # Faster training
    }
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Add bounds to model for post-processing
    model.bounds = bounds.get(target_name, (None, None))
    
    return model

# ========================================
# MODEL 2: LightGBM (Faster, similar accuracy)
# ========================================
def train_lightgbm_model(X_train, y_train, X_val, y_val, target_name):
    """Train LightGBM model - Faster alternative to XGBoost"""
    
    bounds = {
        'wind_speed_10m': (0, 30),
        'apparent_temperature': (15, 42),
        'relative_humidity_2m': (20, 100),
        'chance_of_rain': (0, 100)
    }
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    model.bounds = bounds.get(target_name, (None, None))
    
    return model

# ========================================
# MAIN TRAINING PIPELINE
# ========================================
def train_city_models(hourly_csv, model_type='xgboost'):
    """
    Train models for each city
    model_type: 'xgboost' or 'lightgbm'
    """
    
    print(f"🚀 Starting training with {model_type.upper()}")
    print("=" * 60)
    
    # Load data
    hourly_path = os.path.join(DATA_DIR, hourly_csv)
    
    print("📖 Loading data...")
    
    # Read in chunks to handle large files
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(hourly_path, chunksize=chunk_size):
        chunk["city_name"] = chunk["city_name"].str.strip().str.lower()
        
        # Optimize dtypes
        for col in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'apparent_temperature']:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype('float32')
        
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory
    
    print(f"📊 Loaded {len(df):,} rows, {df.memory_usage().sum() / 1024**2:.1f} MB")
    
    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(["city_name", "datetime"], inplace=True)
    
    # Data quality: remove outliers
    print("🧹 Cleaning data...")
    for col in ['wind_speed_10m', 'temperature_2m', 'apparent_temperature', 'relative_humidity_2m']:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q1, q99)
    
    # Calculate rain chance
    print("🌧️ Calculating chance_of_rain...")
    df["chance_of_rain"] = calculate_real_rain_chance(df)
    
    # Target variables
    targets = [
        "chance_of_rain",
        "wind_speed_10m",
        "apparent_temperature",
        "relative_humidity_2m"
    ]
    
    # Base features (current conditions)
    base_features = [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'apparent_temperature'
    ]
    
    city_names = df["city_name"].unique()
    print(f"📊 Training models for {len(city_names)} cities")
    print(f"🎯 Targets: {', '.join(targets)}\n")
    
    # Process cities one at a time to save memory
    for city in tqdm(city_names, desc="Training cities"):
        # Load only this city's data
        city_df = df[df["city_name"] == city].copy().reset_index(drop=True)
        
        # Free up memory
        import gc
        gc.collect()
        
        if len(city_df) < 1000:
            print(f"⚠️ Skipping {city} (insufficient data: {len(city_df)} rows)")
            continue
        
        # Create features
        city_df = create_weather_features(city_df)
        city_df = create_lag_features(city_df, base_features, lags=[1, 3, 6, 12, 24])
        
        # Remove rows with NaN (from lag features)
        city_df = city_df.dropna()
        
        if len(city_df) < 500:
            print(f"⚠️ Skipping {city} after feature creation (only {len(city_df)} rows)")
            continue
        
        # Feature columns
        feature_cols = [col for col in city_df.columns if col not in 
                       ['datetime', 'city_name'] + targets]
        
        for target in targets:
            if target not in city_df.columns or city_df[target].isna().sum() > len(city_df) * 0.3:
                continue
            
            # Prepare data
            X = city_df[feature_cols].values
            y = city_df[target].values
            
            # Split: 80% train, 20% validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            try:
                # Train model
                if model_type == 'xgboost':
                    model = train_xgboost_model(X_train, y_train, X_val, y_val, target)
                elif model_type == 'lightgbm':
                    model = train_lightgbm_model(X_train, y_train, X_val, y_val, target)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Calculate validation error
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                
                # Save model and scaler
                model_name = f"{city}_{target}_{model_type}"
                model_path = os.path.join(OUTPUT_DIR, f"{model_name}.pkl")
                scaler_path = os.path.join(OUTPUT_DIR, f"{model_name}_scaler.pkl")
                features_path = os.path.join(OUTPUT_DIR, f"{model_name}_features.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                joblib.dump(feature_cols, features_path)
                
                print(f"✅ {city.title()} → {target} | Val RMSE: {val_rmse:.3f}")
                
            except Exception as e:
                print(f"❌ Failed {city} → {target}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"📁 Models saved to: {OUTPUT_DIR}")
    print("\n💡 Next steps:")
    print("1. Update weather_module.py to load these models")
    print("2. Test predictions on holdout data")
    print(f"3. Models are {model_type.upper()}-based with lag features")

if __name__ == "__main__":
    # Choose your model type
    MODEL_TYPE = 'lightgbm'  # or 'xgboost'
    
    print("🌤️  PROFESSIONAL WEATHER MODEL TRAINING")
    print("=" * 60)
    print(f"Model: {MODEL_TYPE.upper()}")
    print("=" * 60)
    
    train_city_models(
        "data\\raw\\hourly_data_combined_2020_to_2023.csv",
        model_type=MODEL_TYPE
    )