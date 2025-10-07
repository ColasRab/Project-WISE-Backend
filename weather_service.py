# weather_service.py (optimized for fast long-range forecasts)
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
print("STARTING OPTIMIZED WEATHER SERVICE")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PORT environment variable: {os.environ.get('PORT', 'NOT SET')}")
print("=" * 60)

app = FastAPI(title="Weather API", version="2.0-optimized")

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
model_load_error = None
models_ready = False


def train_and_load_models():
    global api, model_load_error, models_ready
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        from weather_module import WeatherAPI
        api = WeatherAPI(model_dir=model_dir)
        models_ready = True
        print("✅ Pretrained models loaded successfully!")

    except Exception as e:
        model_load_error = str(e)
        print(f"❌ Failed to load models: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", "8000")
    print("=" * 60)
    print("🚀 STARTUP EVENT TRIGGERED")
    print(f"🚀 Server binding to 0.0.0.0:{port}")
    print("=" * 60)

    # Spawn background thread so port binds immediately
    threading.Thread(target=train_and_load_models, daemon=True).start()


@app.get("/")
def root():
    return {
        "status": "Weather API is running (optimized)",
        "version": "2.0-optimized",
        "models_loaded": models_ready,
        "model_load_error": model_load_error,
        "optimization": "Fast long-range forecasts up to 5+ months",
        "endpoints": {
            "health": "/health",
            "weather": "/api/weather?lat={lat}&lon={lon}&target_date={YYYY-MM-DD}&target_hour={0-23 or 'all'}"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if models_ready else "loading",
        "models_loaded": models_ready,
        "model_load_error": model_load_error,
        "port": os.environ.get("PORT", "8000")
    }


@app.get("/api/weather")
async def get_forecast(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    target_date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    target_hour: str = Query("all", description="Target hour (0-23) or 'all' for full day")
):
    """Get weather forecast for a specific date and hour (optimized)."""
    start_time = time.time()

    if not models_ready or api is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "message": f"Models are still loading: {model_load_error or 'please retry shortly'}",
                "location": {"latitude": lat, "longitude": lon},
                "forecast": []
            }
        )

    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        now = datetime.now()
        
        # Calculate days ahead for logging
        days_ahead = (target_dt - now).days
        print(f"📅 Forecast request: {days_ahead} days ahead ({target_date})")

        if target_hour == "all":
            # OPTIMIZED: Get full day forecast directly without iterating through all intermediate dates
            print(f"⚡ Using optimized full-day forecast method")
            target_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if target is in the past
            if target_start < now.replace(hour=0, minute=0, second=0, microsecond=0):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Cannot forecast for past dates"}
                )
            
            forecasts = api.get_forecast_for_day(target_dt, sample_every=3)
            
        else:
            # OPTIMIZED: Get single hour forecast directly
            print(f"⚡ Using optimized single-hour forecast method")
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
            
            forecast = api.get_forecast_for_datetime(target_datetime)
            forecasts = [forecast]

        elapsed = time.time() - start_time
        print(f"✅ Generated {len(forecasts)} forecast(s) in {elapsed:.2f}s (optimized)")

        if not forecasts:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_data",
                    "message": "No forecast data available for the requested time",
                    "location": {"latitude": lat, "longitude": lon},
                    "forecast": []
                }
            )

        return {
            "status": "success",
            "location": {"latitude": lat, "longitude": lon, "name": f"{lat}, {lon}"},
            "forecast": forecasts,
            "meta": {
                "generation_time_seconds": round(elapsed, 2),
                "forecast_count": len(forecasts),
                "target_date": target_date,
                "target_hour": target_hour,
                "days_ahead": days_ahead,
                "optimized": True
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
                "location": {"latitude": lat, "longitude": lon},
                "forecast": []
            }
        )


print("=" * 60)
print("✅ OPTIMIZED APP CREATED SUCCESSFULLY")
print("=" * 60)