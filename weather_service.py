"""
Weather Service API - Using XGBoost/LightGBM models with lazy loading
"""
import os
import sys
import threading
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
import time
from datetime import datetime

print("=" * 60)
print("STARTING ML WEATHER SERVICE (XGBoost/LightGBM)")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PORT environment variable: {os.environ.get('PORT', 'NOT SET')}")
print("=" * 60)

app = FastAPI(title="Weather API - ML Edition", version="3.0-ml")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
api = None
model_scan_error = None
models_ready = False


def scan_models():
    """Scan available models without loading them (instant startup)"""
    global api, model_scan_error, models_ready
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try both 'models' and 'models_v2' directories
        model_dirs = [
            os.path.join(script_dir, "models_v2"),
            os.path.join(script_dir, "models")
        ]
        
        model_dir = None
        for d in model_dirs:
            if os.path.exists(d):
                model_dir = d
                print(f"✅ Found model directory: {model_dir}")
                break
        
        if not model_dir:
            model_dir = model_dirs[0]
            os.makedirs(model_dir, exist_ok=True)
            print(f"📁 Created model directory: {model_dir}")

        from weather_module import WeatherAPI
        api = WeatherAPI(model_dir=model_dir)
        models_ready = True
        print("✅ Model scan complete! Ready for lazy loading.")

    except Exception as e:
        model_scan_error = str(e)
        print(f"❌ Failed to scan models: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", "8000")
    print("=" * 60)
    print("🚀 STARTUP EVENT TRIGGERED")
    print(f"🚀 Server binding to 0.0.0.0:{port}")
    print("=" * 60)

    # Scan models in background (very fast - just reads filenames)
    threading.Thread(target=scan_models, daemon=True).start()


@app.get("/")
def root():
    return {
        "status": "Weather API is running",
        "version": "3.0-ml",
        "model_type": "XGBoost/LightGBM with lag features",
        "models_ready": models_ready,
        "model_scan_error": model_scan_error,
        "optimization": "Lazy loading - models loaded on-demand for ultra-fast startup",
        "endpoints": {
            "health": "/health",
            "weather": "/api/weather?city={city}&target_date={YYYY-MM-DD}&target_hour={0-23 or 'all'}",
            "models_info": "/api/models/info"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if models_ready else "loading",
        "models_ready": models_ready,
        "model_scan_error": model_scan_error,
        "model_type": "XGBoost/LightGBM",
        "port": os.environ.get("PORT", "8000")
    }


@app.get("/api/models/info")
def models_info():
    """Get information about loaded vs available models"""
    if not models_ready or api is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "message": "Models are still being scanned"
            }
        )
    
    try:
        info = api.get_loaded_models_info()
        return {
            "status": "success",
            "model_type": "XGBoost/LightGBM",
            "lazy_loading": True,
            **info
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/api/weather")
async def get_forecast(
    city: str = Query(..., description="City name (e.g. 'Manila')"),
    target_date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    target_hour: str = Query("all", description="Target hour (0-23) or 'all' for full day")
):
    """Get weather forecast for a specific date and hour using ML models."""
    start_time = time.time()

    if not models_ready or api is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "message": f"Models are still being scanned: {model_scan_error or 'please retry shortly'}",
                "location": {"city": city},
                "forecast": []
            }
        )

    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        now = datetime.now()
        
        # Calculate days ahead for logging
        days_ahead = (target_dt - now).days
        print(f"📅 Forecast request: {days_ahead} days ahead ({target_date}) for {city}")

        if target_hour == "all":
            # Full day forecast with lazy loading
            print(f"⚡ Generating full-day ML forecast")
            target_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if target is in the past
            if target_start < now.replace(hour=0, minute=0, second=0, microsecond=0):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Cannot forecast for past dates"}
                )
            
            # Generate forecast for the day
            forecasts = api.get_forecast_for_day(target_dt, sample_every=3, city_name=city)
            
        else:
            # Single hour forecast with lazy loading
            print(f"⚡ Generating single-hour ML forecast")
            target_hour_int = int(target_hour)
            if target_hour_int < 0 or target_hour_int > 23:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Hour must be between 0 and 23"}
                )

            target_datetime = target_dt.replace(hour=target_hour_int, minute=0, second=0, microsecond=0)
            
            # Check if target is in the past
            if target_datetime < now:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Cannot forecast for past times"}
                )
            
            # Generate forecast for single datetime
            forecast = api.get_forecast_for_datetime(target_datetime, city_name=city)
            forecasts = [forecast]

        elapsed = time.time() - start_time
        
        # Get current loading stats
        models_info = api.get_loaded_models_info()
        print(f"✅ Generated {len(forecasts)} ML forecast(s) in {elapsed:.2f}s")
        print(f"📊 Models loaded: {models_info['loaded_models']}/{models_info['available_models']} ({models_info['memory_saved']} memory saved)")

        if not forecasts:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_data",
                    "message": "No forecast data available for the requested time",
                    "location": {"city": city},
                    "forecast": []
                }
            )

        return {
            "status": "success",
            "location": {"city": city, "name": city},
            "forecast": forecasts,
            "meta": {
                "generation_time_seconds": round(elapsed, 2),
                "forecast_count": len(forecasts),
                "target_date": target_date,
                "target_hour": target_hour,
                "days_ahead": days_ahead,
                "model_type": "XGBoost/LightGBM",
                "lazy_loading": True,
                "models_loaded": models_info['loaded_models'],
                "models_available": models_info['available_models']
            }
        }

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid date format. Use YYYY-MM-DD: {str(e)}"}
        )
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error generating forecast: {str(e)}",
                "error_type": type(e).__name__,
                "location": {"city": city},
                "forecast": []
            }
        )


print("=" * 60)
print("✅ ML WEATHER SERVICE INITIALIZED")
print("=" * 60)