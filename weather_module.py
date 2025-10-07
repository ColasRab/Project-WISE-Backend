"""
Optimized Weather Module - Fast forecasts using per-city Prophet models with lazy loading
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import joblib
import threading


class WeatherAPI:
    """API for weather forecasting using lazy-loaded per-city Prophet models"""
    
    def __init__(self, model_dir: str):
        """
        Initialize the Weather API with lazy loading support
        
        Args:
            model_dir: Directory containing the pickled Prophet models
        """
        self.models = {}  # Cached loaded models
        self.model_dir = model_dir
        self.available_cities = set()
        self.available_model_files = {}  # Map of city -> {target -> filepath}
        self._lock = threading.Lock()  # Thread-safe model loading
        
        # Scan for available models without loading them
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan directory for available models without loading them"""
        if not os.path.exists(self.model_dir):
            print(f"⚠️  Model directory not found: {self.model_dir}")
            return
        
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_prophet.pkl')]

        # Known targets used by the forecasting models. We match longest targets first
        # so multi-word targets like 'chance_of_rain' are detected correctly.
        known_targets = [
            'chance_of_rain',
            'wind_speed_10m',
            'apparent_temperature',
            'relative_humidity_2m'
        ]

        for model_file in model_files:
            base = model_file.replace('_prophet.pkl', '')

            # Find which known target the filename ends with (longest-first)
            matched_target = None
            for t in sorted(known_targets, key=len, reverse=True):
                if base.endswith('_' + t):
                    matched_target = t
                    city = base[:-(len(t) + 1)]  # remove '_' + target
                    break

            # If no known target matched, skip this file (or fallback to last underscore)
            if not matched_target:
                # fallback: split once on last underscore
                parts = base.rsplit('_', 1)
                if len(parts) == 2:
                    city, matched_target = parts
                else:
                    print(f"⚠️  Unable to parse model filename: {model_file}")
                    continue

            if city not in self.available_model_files:
                self.available_model_files[city] = {}

            filepath = os.path.join(self.model_dir, model_file)
            self.available_model_files[city][matched_target] = filepath
            self.available_cities.add(city)
            print(f"📋 Found model: {city} -> {matched_target}")
        
        if not self.available_model_files:
            raise FileNotFoundError(f"No valid Prophet models found in {self.model_dir}")
        
        print(f"\n📍 Scanned {len(self.available_cities)} cities with models available")
        print(f"Available cities: {sorted(self.available_cities)}")
        print(f"💡 Models will be loaded on-demand (lazy loading)")
    
    def _load_city_model(self, city: str, target: str) -> Any:
        """
        Lazy load a specific model for a city and target
        
        Args:
            city: City name
            target: Target variable (e.g., 'wind_speed_10m')
            
        Returns:
            Loaded Prophet model
        """
        with self._lock:  # Thread-safe loading
            # Check if already loaded
            if city in self.models and target in self.models[city]:
                return self.models[city][target]
            
            # Check if model file exists
            if city not in self.available_model_files:
                raise ValueError(f"No models available for city: {city}")
            
            if target not in self.available_model_files[city]:
                raise ValueError(f"No model available for {city} -> {target}")
            
            # Load the model
            filepath = self.available_model_files[city][target]
            try:
                print(f"⚡ Lazy loading: {city} -> {target}")
                model = joblib.load(filepath)
                
                # Cache the loaded model
                if city not in self.models:
                    self.models[city] = {}
                self.models[city][target] = model
                
                print(f"✅ Loaded and cached: {city} -> {target}")
                return model
                
            except Exception as e:
                print(f"⚠️  Failed to load {filepath}: {e}")
                raise
    
    def _ensure_city_models_loaded(self, city: str, targets: List[str]):
        """
        Ensure all required models for a city are loaded
        
        Args:
            city: City name
            targets: List of target variables needed
        """
        for target in targets:
            if city not in self.models or target not in self.models[city]:
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
        
        import pandas as pd
        
        # Available targets from training
        targets = ['chance_of_rain', 'wind_speed_10m', 'apparent_temperature', 'relative_humidity_2m']
        
        # Lazy load only the models we need
        self._ensure_city_models_loaded(city, targets)
        
        # Create DataFrame with just the target datetime
        future_df = pd.DataFrame({'ds': [target_datetime]})
        
        # Get predictions from all models for this city
        predictions = {}
        
        for target in targets:
            if city in self.models and target in self.models[city]:
                forecast = self.models[city][target].predict(future_df)
                predictions[target] = forecast['yhat'].values[0]
        
        # Extract values with defaults
        chance_of_rain = max(0, min(100, predictions.get('chance_of_rain', 0)))
        wind_speed = max(0, predictions.get('wind_speed_10m', 0))
        temp = predictions.get('apparent_temperature', 25)
        humidity = max(0, min(100, predictions.get('relative_humidity_2m', 50)))
        
        # Calculate precipitation from chance of rain (estimate)
        # Higher chance of rain = higher expected precipitation
        precip = (chance_of_rain / 100) * 5  # Scale to reasonable mm range
        
        # Assess conditions
        assessment = self._assess_conditions(wind_speed, precip, temp, humidity)
        
        return {
            'datetime': target_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': int(target_datetime.timestamp()),
            'city': city.title(),
            'predicted_wind_speed': round(float(wind_speed), 2),
            'predicted_precip_mm': round(float(precip), 2),
            'predicted_temp_c': round(float(temp), 2),
            'predicted_humidity': round(float(humidity), 2),
            'chance_of_rain': round(float(chance_of_rain), 2),
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
        
        import pandas as pd
        
        # Available targets from training
        targets = ['chance_of_rain', 'wind_speed_10m', 'apparent_temperature', 'relative_humidity_2m']
        
        # Lazy load only the models we need
        self._ensure_city_models_loaded(city, targets)
        
        # Create list of datetimes for the target day
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        future_dates = []
        
        for hour in range(0, 24, sample_every):
            future_dates.append(start_of_day + timedelta(hours=hour))
        
        # Create DataFrame for all target times at once
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Get predictions from all models in batch
        predictions = {}
        
        for target in targets:
            if city in self.models and target in self.models[city]:
                forecast = self.models[city][target].predict(future_df)
                predictions[target] = forecast['yhat'].values
        
        # Build forecast results
        forecasts = []
        
        for i, dt in enumerate(future_dates):
            chance_of_rain = max(0, min(100, predictions.get('chance_of_rain', [0] * len(future_dates))[i]))
            wind_speed = max(0, predictions.get('wind_speed_10m', [0] * len(future_dates))[i])
            temp = predictions.get('apparent_temperature', [25] * len(future_dates))[i]
            humidity = max(0, min(100, predictions.get('relative_humidity_2m', [50] * len(future_dates))[i]))
            
            # Calculate precipitation from chance of rain
            precip = (chance_of_rain / 100) * 5
            
            # Assess conditions
            assessment = self._assess_conditions(wind_speed, precip, temp, humidity)
            
            forecast_item = {
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(dt.timestamp()),
                'city': city.title(),
                'predicted_wind_speed': round(float(wind_speed), 2),
                'predicted_precip_mm': round(float(precip), 2),
                'predicted_temp_c': round(float(temp), 2),
                'predicted_humidity': round(float(humidity), 2),
                'chance_of_rain': round(float(chance_of_rain), 2),
                'assessment': assessment
            }
            
            forecasts.append(forecast_item)
        
        return forecasts
    
    def get_loaded_models_info(self) -> Dict[str, Any]:
        """
        Get information about currently loaded models
        
        Returns:
            Dictionary with loaded models statistics
        """
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
        """
        Assess weather conditions and generate risk assessment
        
        Args:
            wind_speed: Wind speed in m/s
            precip: Precipitation in mm
            temp: Temperature in Celsius
            humidity: Humidity percentage
            
        Returns:
            Assessment dictionary with categories and safety info
        """
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
        
        # Temperature assessment (adjusted for Philippine climate)
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