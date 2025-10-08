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
    
    def _calculate_heat_index(self, temp: float, humidity: float) -> float:
        if temp < 27:  # Heat index not significant below 27°C
            return temp
        
        # Convert to Fahrenheit for calculation
        temp_f = (temp * 9/5) + 32
        
        # Rothfusz regression for heat index
        hi_f = (-42.379 + 
                2.04901523 * temp_f + 
                10.14333127 * humidity - 
                0.22475541 * temp_f * humidity - 
                0.00683783 * temp_f * temp_f - 
                0.05481717 * humidity * humidity + 
                0.00122874 * temp_f * temp_f * humidity + 
                0.00085282 * temp_f * humidity * humidity - 
                0.00000199 * temp_f * temp_f * humidity * humidity)
        
        # Convert back to Celsius
        hi_celsius = (hi_f - 32) * 5/9
        return round(hi_celsius, 2)


    def _assess_conditions(self, wind_speed: float, precip: float, 
                    temp: float, humidity: float, chance_of_rain: float = None) -> Dict[str, Any]:
        """
        Assess weather conditions and generate risk assessment.
        Calibrated for Philippine tropical climate.
            
        Args:
                wind_speed: Wind speed in m/s
                precip: Precipitation in mm
                temp: Temperature in Celsius
                humidity: Relative humidity (0-100)
                chance_of_rain: Optional probability of rain (0-100)
            """
            
            # === WIND ASSESSMENT (Beaufort Scale adapted) ===
        if wind_speed < 1.5:  # 0-5.4 km/h
            wind_cat = "Calm"
            wind_severity = 0.0
        elif wind_speed < 3.3:  # 5.4-12 km/h
            wind_cat = "Light Breeze"
            wind_severity = 0.1
        elif wind_speed < 5.5:  # 12-20 km/h
            wind_cat = "Gentle Breeze"
            wind_severity = 0.2
        elif wind_speed < 8.0:  # 20-29 km/h
            wind_cat = "Moderate Breeze"
            wind_severity = 0.3
        elif wind_speed < 10.8:  # 29-39 km/h
            wind_cat = "Fresh Breeze"
            wind_severity = 0.5
        elif wind_speed < 13.9:  # 39-50 km/h
            wind_cat = "Strong Breeze"
            wind_severity = 0.7
        elif wind_speed < 17.2:  # 50-62 km/h
            wind_cat = "Near Gale"
            wind_severity = 0.85
        else:
            wind_cat = "Gale/Storm"
            wind_severity = 1.0
        
        # === CHANCE OF RAIN ASSESSMENT (if provided) ===
        rain_chance_severity = 0.0
        rain_chance_cat = None
        
        if chance_of_rain is not None:
            if chance_of_rain < 20:
                rain_chance_cat = "Unlikely"
                rain_chance_severity = 0.0
            elif chance_of_rain < 40:
                rain_chance_cat = "Slight Chance"
                rain_chance_severity = 0.2
            elif chance_of_rain < 60:
                rain_chance_cat = "Possible"
                rain_chance_severity = 0.4
            elif chance_of_rain < 75:
                rain_chance_cat = "Likely"
                rain_chance_severity = 0.6
            elif chance_of_rain < 90:
                rain_chance_cat = "Very Likely"
                rain_chance_severity = 0.8
            else:
                rain_chance_cat = "Imminent"
                rain_chance_severity = 1.0
        
        # === TEMPERATURE ASSESSMENT (with Heat Index) ===
        heat_index = self._calculate_heat_index(temp, humidity)
        assessment_temp = heat_index  # Use heat index for assessment
        
        if assessment_temp < 20:
            temp_cat = "Cool"
            temp_severity = 0.2
            temp_description = "Cooler than usual for Philippines"
        elif assessment_temp < 24:
            temp_cat = "Comfortable"
            temp_severity = 0.0
            temp_description = "Ideal temperature"
        elif assessment_temp < 27:
            temp_cat = "Pleasant"
            temp_severity = 0.1
            temp_description = "Comfortable warm weather"
        elif assessment_temp < 32:
            temp_cat = "Warm"
            temp_severity = 0.3
            temp_description = "Stay hydrated"
        elif assessment_temp < 37:
            temp_cat = "Hot"
            temp_severity = 0.5
            temp_description = "Limit sun exposure, drink plenty of water"
        elif assessment_temp < 41:
            temp_cat = "Very Hot"
            temp_severity = 0.7
            temp_description = "Heat exhaustion possible, limit outdoor exposure"
        elif assessment_temp < 54:
            temp_cat = "Extreme Heat"
            temp_severity = 0.9
            temp_description = "Heat stroke likely, stay indoors with AC"
        else:
            temp_cat = "Dangerous"
            temp_severity = 1.0
            temp_description = "Life-threatening heat, stay indoors"
        
        # === HUMIDITY ASSESSMENT (Philippine context: 70-90% is normal) ===
        if humidity < 30:
            humid_cat = "Very Dry"
            humid_severity = 0.3
            humid_description = "Unusually dry for Philippines"
        elif humidity < 50:
            humid_cat = "Dry"
            humid_severity = 0.1
            humid_description = "Lower than typical"
        elif humidity < 65:
            humid_cat = "Comfortable"
            humid_severity = 0.0
            humid_description = "Ideal humidity level"
        elif humidity < 75:
            humid_cat = "Moderate"
            humid_severity = 0.2
            humid_description = "Typical for Philippines"
        elif humidity < 85:
            humid_cat = "Humid"
            humid_severity = 0.4
            humid_description = "Noticeably humid"
        elif humidity < 92:
            humid_cat = "Very Humid"
            humid_severity = 0.6
            humid_description = "High humidity, may feel uncomfortable"
        else:
            humid_cat = "Oppressive"
            humid_severity = 0.8
            humid_description = "Extremely humid, rain likely imminent"
        
        # === OVERALL RISK CALCULATION ===
        # Weight factors based on impact on outdoor activities
        weights = {
            'wind': 0.25,           # Wind conditions
            'rain_chance': 0.40,    # Chance of rain - HIGHEST PRIORITY for planning
            'temperature': 0.20,    # Temperature/heat index
            'humidity': 0.15        # Humidity - correlates with rain and comfort
        }
        
        # Use rain chance if available, otherwise use precipitation
        rain_severity = rain_chance_severity
        overall_risk = (
            wind_severity * weights['wind'] +
            rain_severity * weights['rain_chance'] +
            temp_severity * weights['temperature'] +
            humid_severity * weights['humidity']
        )
        
        # Adjust overall risk if humidity is extreme (>90%) - rain is very likely
        if humidity > 90 and chance_of_rain is not None and chance_of_rain < 70:
            overall_risk = min(overall_risk + 0.15, 1.0)  # Boost risk
        
        # === SAFETY ASSESSMENT ===
        safe_for_outdoors = all([
            wind_severity < 0.6,
            rain_severity < 0.6,
            temp_severity < 0.6,
            humid_severity < 0.7
        ])
        
        # === RECOMMENDATION ===
        concerns = []
        if rain_severity >= 0.6:
            concerns.append("high chance of rain")
        if wind_severity >= 0.5:
            concerns.append("strong winds")
        if temp_severity >= 0.6:
            concerns.append("extreme temperature")
        if humid_severity >= 0.7:
            concerns.append("very high humidity")
        
        if overall_risk < 0.2:
            recommendation = "☀️ Excellent conditions for outdoor activities!"
        elif overall_risk < 0.4:
            recommendation = "✅ Good conditions. Enjoy outdoor activities with normal precautions."
            if concerns:
                recommendation += f" Watch for: {', '.join(concerns)}."
        elif overall_risk < 0.6:
            recommendation = "⚠️ Moderate conditions. Be prepared and monitor weather changes."
            if concerns:
                recommendation += f" Concerns: {', '.join(concerns)}."
        elif overall_risk < 0.8:
            recommendation = "⛔ Challenging conditions. Consider postponing outdoor plans."
            if concerns:
                recommendation += f" Due to: {', '.join(concerns)}."
        else:
            recommendation = "🚨 Dangerous conditions. Stay indoors if possible."
            if concerns:
                recommendation += f" Severe: {', '.join(concerns)}."
        
        # === BUILD RESPONSE ===
        result = {
            'wind': {
                'category': wind_cat,
                'severity': round(wind_severity, 2),
                'safe': wind_severity < 0.6
            },
            
            'temperature': {
                'category': temp_cat,
                'severity': round(temp_severity, 2),
                'safe': temp_severity < 0.6,
                'actual_temp': round(temp, 2),
                'feels_like': round(heat_index, 2),
                'description': temp_description
            },
            'humidity': {
                'category': humid_cat,
                'severity': round(humid_severity, 2),
                'safe': humid_severity < 0.7,
                'description': humid_description
            },
            'overall_risk': round(overall_risk, 2),
            'safe_for_outdoors': safe_for_outdoors,
            'recommendation': recommendation
        }
        
        # Add rain chance info if provided
        if chance_of_rain is not None:
            result['rain_chance'] = {
                'category': rain_chance_cat,
                'probability': round(chance_of_rain, 1),
                'severity': round(rain_chance_severity, 2),
                'safe': rain_chance_severity < 0.6
            }
        
        return result   