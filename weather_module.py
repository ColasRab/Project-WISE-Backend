"""
Optimized Weather Module - Fast forecasts using XGBoost/LightGBM models with lazy loading
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import joblib
import threading


class WeatherAPI:
    """API for weather forecasting using lazy-loaded XGBoost/LightGBM models"""
    
    def __init__(self, model_dir: str):
        """
        Initialize the Weather API with lazy loading support
        
        Args:
            model_dir: Directory containing the pickled models, scalers, and feature lists
        """
        self.models = {}  # Cached loaded models: {city: {target: model}}
        self.scalers = {}  # Cached scalers: {city: {target: scaler}}
        self.features = {}  # Cached feature lists: {city: {target: [feature_names]}}
        self.model_dir = model_dir
        self.available_cities = set()
        self.available_model_files = {}  # Map of city -> {target -> {model, scaler, features}}
        self._lock = threading.Lock()  # Thread-safe model loading
        
        # Scan for available models without loading them
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan directory for available models without loading them"""
        if not os.path.exists(self.model_dir):
            print(f"⚠️  Model directory not found: {self.model_dir}")
            return
        
        # Look for model files (either xgboost or lightgbm)
        all_files = os.listdir(self.model_dir)
        model_files = [f for f in all_files if f.endswith('.pkl') and 
                      ('xgboost' in f or 'lightgbm' in f) and 
                      'scaler' not in f and 'features' not in f]

        # Known targets
        known_targets = [
            'chance_of_rain',
            'wind_speed_10m',
            'apparent_temperature',
            'relative_humidity_2m'
        ]

        for model_file in model_files:
            # Extract city, target, and model_type
            # Format: {city}_{target}_{model_type}.pkl
            base = model_file.replace('.pkl', '')
            
            # Find model type (xgboost or lightgbm)
            if base.endswith('_xgboost'):
                model_type = 'xgboost'
                base = base[:-9]  # Remove '_xgboost'
            elif base.endswith('_lightgbm'):
                model_type = 'lightgbm'
                base = base[:-9]  # Remove '_lightgbm'
            else:
                continue
            
            # Find which target the filename contains (longest-first)
            matched_target = None
            for t in sorted(known_targets, key=len, reverse=True):
                if base.endswith('_' + t):
                    matched_target = t
                    city = base[:-(len(t) + 1)]  # remove '_' + target
                    break

            if not matched_target:
                print(f"⚠️  Unable to parse model filename: {model_file}")
                continue

            # Look for corresponding scaler and features files
            scaler_file = f"{city}_{matched_target}_{model_type}_scaler.pkl"
            features_file = f"{city}_{matched_target}_{model_type}_features.pkl"
            
            scaler_path = os.path.join(self.model_dir, scaler_file)
            features_path = os.path.join(self.model_dir, features_file)
            
            if not os.path.exists(scaler_path) or not os.path.exists(features_path):
                print(f"⚠️  Missing scaler or features for {city} -> {matched_target}")
                continue

            if city not in self.available_model_files:
                self.available_model_files[city] = {}

            self.available_model_files[city][matched_target] = {
                'model': os.path.join(self.model_dir, model_file),
                'scaler': scaler_path,
                'features': features_path,
                'model_type': model_type
            }
            self.available_cities.add(city)
            print(f"📋 Found model: {city} -> {matched_target} ({model_type})")
        
        if not self.available_model_files:
            raise FileNotFoundError(f"No valid models found in {self.model_dir}")
        
        print(f"\n🔍 Scanned {len(self.available_cities)} cities with models available")
        print(f"Available cities: {sorted(self.available_cities)}")
        print(f"💡 Models will be loaded on-demand (lazy loading)")
    
    def _load_city_model(self, city: str, target: str):
        """
        Lazy load a specific model, scaler, and features for a city and target
        
        Args:
            city: City name
            target: Target variable (e.g., 'wind_speed_10m')
        """
        with self._lock:  # Thread-safe loading
            # Check if already loaded
            if (city in self.models and target in self.models[city] and
                city in self.scalers and target in self.scalers[city] and
                city in self.features and target in self.features[city]):
                return
            
            # Check if model files exist
            if city not in self.available_model_files:
                raise ValueError(f"No models available for city: {city}")
            
            if target not in self.available_model_files[city]:
                raise ValueError(f"No model available for {city} -> {target}")
            
            # Load the model, scaler, and features
            files = self.available_model_files[city][target]
            try:
                print(f"⚡ Lazy loading: {city} -> {target} ({files['model_type']})")
                
                model = joblib.load(files['model'])
                scaler = joblib.load(files['scaler'])
                feature_list = joblib.load(files['features'])
                
                # Cache the loaded components
                if city not in self.models:
                    self.models[city] = {}
                    self.scalers[city] = {}
                    self.features[city] = {}
                
                self.models[city][target] = model
                self.scalers[city][target] = scaler
                self.features[city][target] = feature_list
                
                print(f"✅ Loaded and cached: {city} -> {target} ({len(feature_list)} features)")
                
            except Exception as e:
                print(f"⚠️  Failed to load model for {city} -> {target}: {e}")
                raise

    def _create_time_features(self, dt: datetime) -> Dict[str, float]:
        """Create time-based features for a single datetime"""
        hour = dt.hour
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday
        month = dt.month
        week_of_year = dt.isocalendar()[1]
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'day_of_year': day_of_year,
            'month': month,
            'week_of_year': week_of_year,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12)
        }
    
    def _create_feature_vector(self, city: str, target: str, dt: datetime, 
                              historical_values: Dict[str, List[float]]) -> np.ndarray:
        """
        Create feature vector for prediction
        
        Args:
            city: City name
            target: Target variable
            dt: Target datetime
            historical_values: Dictionary of historical values for lag features
                              Format: {'wind_speed_10m': [v1, v2, ...], ...}
        
        Returns:
            Feature vector as numpy array
        """
        feature_list = self.features[city][target]
        feature_dict = {}
        
        # Add time features
        time_features = self._create_time_features(dt)
        feature_dict.update(time_features)
        
        # Add lag features
        base_vars = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'apparent_temperature']
        lags = [1, 3, 6, 12, 24]
        
        for var in base_vars:
            if var in historical_values:
                values = historical_values[var]
                for lag in lags:
                    lag_key = f'{var}_lag_{lag}h'
                    # Use the lag-th value from history (if available)
                    if lag <= len(values):
                        feature_dict[lag_key] = values[-lag]
                    else:
                        feature_dict[lag_key] = 0  # Default for missing history
                
                # Rolling statistics
                if len(values) >= 6:
                    feature_dict[f'{var}_rolling_mean_6h'] = np.mean(values[-6:])
                    feature_dict[f'{var}_rolling_std_6h'] = np.std(values[-6:])
                else:
                    feature_dict[f'{var}_rolling_mean_6h'] = values[-1] if values else 0
                    feature_dict[f'{var}_rolling_std_6h'] = 0
                
                if len(values) >= 24:
                    feature_dict[f'{var}_rolling_mean_24h'] = np.mean(values[-24:])
                else:
                    feature_dict[f'{var}_rolling_mean_24h'] = values[-1] if values else 0
        
        # Build feature vector in correct order
        feature_vector = []
        for feat_name in feature_list:
            feature_vector.append(feature_dict.get(feat_name, 0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _ensure_city_models_loaded(self, city: str, targets: List[str]):
        """Ensure all required models for a city are loaded"""
        for target in targets:
            if (city not in self.models or target not in self.models[city] or
                city not in self.scalers or target not in self.scalers[city]):
                self._load_city_model(city, target)
    
    def _find_nearest_city(self, city_name: str) -> str:
        """Find the nearest matching city name (case-insensitive)"""
        city_lower = city_name.lower().strip()
        
        # Exact match
        if city_lower in self.available_cities:
            return city_lower
        
        # Partial match
        for available_city in self.available_cities:
            if city_lower in available_city or available_city in city_lower:
                return available_city
        
        # Default to first available city if no match
        if self.available_cities:
            default_city = sorted(self.available_cities)[0]
            print(f"⚠️  City '{city_name}' not found, using '{default_city}' instead")
            return default_city
        
        raise ValueError(f"No cities available and '{city_name}' not found")
    
    def get_forecast_for_datetime(self, target_datetime: datetime, city_name: str = None) -> Dict[str, Any]:
        """
        Generate weather forecast for a specific datetime
        
        Args:
            target_datetime: The exact datetime to forecast
            city_name: Name of the city (will find nearest match)
            
        Returns:
            Single forecast dictionary with weather data and assessment
        """
        if not self.available_cities:
            raise Exception("No models available")
        
        # Find matching city
        if city_name:
            city = self._find_nearest_city(city_name)
        else:
            city = sorted(self.available_cities)[0]
        
        if city not in self.available_model_files:
            raise ValueError(f"No models available for city: {city}")
        
        # Available targets from training
        targets = ['wind_speed_10m', 'apparent_temperature', 'relative_humidity_2m', 'chance_of_rain']
        
        # Lazy load only the models we need
        self._ensure_city_models_loaded(city, targets)
        
        # Initialize historical values with reasonable defaults
        # In production, you'd load recent actual data
        historical_values = {
            'temperature_2m': [28.0] * 24,
            'relative_humidity_2m': [70.0] * 24,
            'wind_speed_10m': [5.0] * 24,
            'apparent_temperature': [28.0] * 24
        }
        
        predictions = {}
        
        # Predict each target
        for target in targets:
            if target not in self.models[city]:
                continue
            
            try:
                # Create feature vector
                X = self._create_feature_vector(city, target, target_datetime, historical_values)
                
                # Scale features
                X_scaled = self.scalers[city][target].transform(X)
                
                # Predict
                pred = self.models[city][target].predict(X_scaled)[0]
                
                # Apply bounds if available
                if hasattr(self.models[city][target], 'bounds'):
                    lower, upper = self.models[city][target].bounds
                    if lower is not None and upper is not None:
                        pred = np.clip(pred, lower, upper)
                
                predictions[target] = float(pred)
                
                # Update historical values for next prediction
                if target == 'wind_speed_10m':
                    historical_values['wind_speed_10m'].append(pred)
                elif target == 'apparent_temperature':
                    historical_values['apparent_temperature'].append(pred)
                    historical_values['temperature_2m'].append(pred)
                elif target == 'relative_humidity_2m':
                    historical_values['relative_humidity_2m'].append(pred)
                
            except Exception as e:
                print(f"⚠️  Prediction failed for {target}: {e}")
                # Fallback defaults
                if target == 'wind_speed_10m':
                    predictions[target] = 5.0
                elif target == 'apparent_temperature':
                    predictions[target] = 28.0
                elif target == 'relative_humidity_2m':
                    predictions[target] = 70.0
                elif target == 'chance_of_rain':
                    predictions[target] = 20.0
        
        # Extract values with defaults
        wind_speed = max(0, predictions.get('wind_speed_10m', 5.0))
        temp = predictions.get('apparent_temperature', 28.0)
        humidity = max(0, min(100, predictions.get('relative_humidity_2m', 70.0)))
        chance_of_rain = max(0, min(100, predictions.get('chance_of_rain', 20.0)))
        
        # Calculate precipitation from chance of rain
        precip = (chance_of_rain / 100) * 5
        
        # Assess conditions
        assessment = self._assess_conditions(wind_speed, precip, temp, humidity)
        
        return {
            'datetime': target_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': int(target_datetime.timestamp()),
            'city': city.title(),
            'predicted_wind_speed': round(wind_speed, 2),
            'predicted_precip_mm': round(precip, 2),
            'predicted_temp_c': round(temp, 2),
            'predicted_humidity': round(humidity, 2),
            'chance_of_rain': round(chance_of_rain, 2),
            'assessment': assessment
        }
    
    def get_forecast_for_day(self, target_date: datetime, city_name: str = None, sample_every: int = 3) -> List[Dict[str, Any]]:
        """
        Generate weather forecast for a full day
        
        Args:
            target_date: The target date (time will be ignored)
            city_name: Name of the city (will find nearest match)
            sample_every: Sample interval in hours (e.g., 3 = every 3 hours)
            
        Returns:
            List of forecast dictionaries for the day
        """
        if not self.available_cities:
            raise Exception("No models available")
        
        # Find matching city
        if city_name:
            city = self._find_nearest_city(city_name)
        else:
            city = sorted(self.available_cities)[0]
        
        if city not in self.available_model_files:
            raise ValueError(f"No models available for city: {city}")
        
        # Available targets from training
        targets = ['wind_speed_10m', 'apparent_temperature', 'relative_humidity_2m', 'chance_of_rain']
        
        # Lazy load only the models we need
        self._ensure_city_models_loaded(city, targets)
        
        # Create list of datetimes for the target day
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        future_dates = []
        
        for hour in range(0, 24, sample_every):
            future_dates.append(start_of_day + timedelta(hours=hour))
        
        # Generate forecasts for each timestamp
        forecasts = []
        for dt in future_dates:
            try:
                forecast = self.get_forecast_for_datetime(dt, city_name=city)
                forecasts.append(forecast)
            except Exception as e:
                print(f"⚠️  Failed to generate forecast for {dt}: {e}")
        
        return forecasts
    
    def get_loaded_models_info(self) -> Dict[str, Any]:
        """Get information about currently loaded models"""
        loaded_count = sum(len(targets) for targets in self.models.values())
        available_count = sum(len(targets) for targets in self.available_model_files.values())
        
        return {
            'loaded_models': loaded_count,
            'available_models': available_count,
            'loaded_cities': list(self.models.keys()),
            'available_cities': sorted(self.available_cities),
            'memory_saved': f"{((available_count - loaded_count) / available_count * 100):.1f}%" if available_count > 0 else "0%"
        }
    
    def _assess_conditions(self, wind_speed: float, precip: float, 
                          temp: float, humidity: float) -> Dict[str, Any]:
        """Assess weather conditions and generate risk assessment"""
        # Wind assessment
        if wind_speed < 3:
            wind_cat = "Calm"
            wind_severity = 0.0
        elif wind_speed < 7:
            wind_cat = "Breezy"
            wind_severity = 0.3
        elif wind_speed < 12:
            wind_cat = "Windy"
            wind_severity = 0.6
        else:
            wind_cat = "Very Windy"
            wind_severity = 0.9
        
        # Precipitation assessment
        if precip < 2.5:
            precip_cat = "Dry"
            precip_severity = 0.0
        elif precip < 7.6:
            precip_cat = "Light Rain"
            precip_severity = 0.3
        elif precip < 50:
            precip_cat = "Moderate Rain"
            precip_severity = 0.6
        else:
            precip_cat = "Heavy Rain"
            precip_severity = 0.9
        
        # Temperature assessment
        if temp < 20:
            temp_cat = "Cool"
            temp_severity = 0.3
        elif temp < 28:
            temp_cat = "Comfortable"
            temp_severity = 0.0
        elif temp < 33:
            temp_cat = "Warm"
            temp_severity = 0.3
        else:
            temp_cat = "Hot"
            temp_severity = 0.6
        
        # Humidity assessment
        if humidity < 40:
            humid_cat = "Dry"
            humid_severity = 0.2
        elif humidity < 70:
            humid_cat = "Comfortable"
            humid_severity = 0.0
        else:
            humid_cat = "Humid"
            humid_severity = 0.4
        
        # Overall risk calculation
        overall_risk = (wind_severity + precip_severity + temp_severity + humid_severity) / 4
        
        # Safety assessment
        safe_for_outdoors = overall_risk < 0.5
        
        if safe_for_outdoors:
            recommendation = "Conditions are favorable for outdoor activities."
        elif overall_risk < 0.7:
            recommendation = "Proceed with caution. Some outdoor activities may be affected."
        else:
            recommendation = "Not recommended for outdoor activities. Stay indoors if possible."
        
        return {
            'wind': {
                'category': wind_cat,
                'severity': wind_severity,
                'safe': wind_severity < 0.6
            },
            'precipitation': {
                'category': precip_cat,
                'severity': precip_severity,
                'safe': precip_severity < 0.6
            },
            'temperature': {
                'category': temp_cat,
                'severity': temp_severity,
                'safe': temp_severity < 0.6
            },
            'humidity': {
                'category': humid_cat,
                'severity': humid_severity,
                'safe': humid_severity < 0.6
            },
            'overall_risk': round(overall_risk, 2),
            'safe_for_outdoors': safe_for_outdoors,
            'recommendation': recommendation
        }