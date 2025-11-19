import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import requests

from database import create_document, get_documents
from schemas import Scan

app = FastAPI(title="AQI Vision API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Utility functions ---------

def categorize_aqi(aqi: Optional[int]) -> str:
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def recommendations_for_category(category: str) -> List[str]:
    c = (category or "").lower()
    if c in ("hazardous",):
        return [
            "Stay indoors and keep windows/doors closed",
            "Use a HEPA air purifier if available",
            "Avoid all outdoor physical activity",
            "Wear a properly fitted N95/FFP2 mask if you must go outside",
            "Set HVAC to recirculate and seal gaps to reduce infiltration",
            "Consider relocating temporarily if conditions persist",
        ]
    if c in ("very unhealthy",):
        return [
            "Avoid outdoor activities; move activities indoors",
            "Use masks (N95/FFP2) if outdoor exposure is unavoidable",
            "Run air purifier on high in rooms you occupy",
            "Sensitive groups should remain indoors",
        ]
    if c in ("unhealthy",):
        return [
            "Reduce prolonged or heavy exertion outdoors",
            "Children, older adults, and those with heart/lung disease should stay indoors",
            "Keep indoor air clean: close windows, run purifier",
        ]
    if c in ("unhealthy for sensitive",):
        return [
            "Sensitive individuals should limit time outdoors",
            "Consider wearing a mask during outdoor activity",
            "Monitor symptoms such as coughing or shortness of breath",
        ]
    return [
        "Enjoy outdoor activities as usual",
        "Keep an eye on changes throughout the day",
    ]


def estimate_aqi_from_image(image: Image.Image) -> Dict[str, Any]:
    """
    Heuristic image-based estimation.
    We DO NOT claim accurate AQI from camera — this is a fun approximation.

    Steps:
    - Downscale and analyze haze by contrast & color cast
    - Compute a "haze index" from saturation/contrast
    - Map haze index to an AQI-like scale (0-200)
    """
    # Normalize size
    img = image.convert("RGB").resize((256, 256))
    arr = np.asarray(img) / 255.0

    # Saturation and brightness
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc

    # Approx saturation
    sat = np.where(maxc == 0, 0, delta / (maxc + 1e-6))
    mean_sat = float(np.mean(sat))

    # Contrast via standard deviation of luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    contrast = float(np.std(luminance))

    # Haze index: low saturation and low contrast -> more haze
    haze_index = (0.5 - mean_sat) + (0.1 - contrast)
    haze_index = max(0.0, haze_index)

    # Map to AQI-ish 0-200
    aqi_est = int(min(200, haze_index * 400))
    category = categorize_aqi(aqi_est)

    return {
        "aqi": aqi_est,
        "category": category,
        "metrics": {
            "mean_saturation": round(mean_sat, 4),
            "contrast": round(contrast, 4),
            "haze_index": round(float(haze_index), 4),
        },
        "note": "Heuristic estimate from image characteristics; not a regulatory measurement.",
    }


class GeoPoint(BaseModel):
    lat: float
    lon: float


@app.get("/")
def read_root():
    return {"message": "AQI Vision API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
    }
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Connected"
    except Exception as e:
        response["database"] = f"❌ {str(e)[:60]}"
    return response


@app.post("/api/estimate/camera")
async def estimate_from_camera(file: UploadFile = File(...)):
    """
    Accepts an image from the camera and returns a heuristic AQI estimate.
    Persists the scan to the database if available.
    """
    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = estimate_aqi_from_image(image)

    # Save to DB (best-effort)
    try:
        doc = Scan(
            source="camera",
            method="heuristic-v1",
            aqi=result["aqi"],
            category=result["category"],
            metrics=result.get("metrics"),
            note=result.get("note"),
        )
        create_document("scan", doc)
    except Exception:
        pass

    return result


@app.post("/api/estimate/geo")
async def estimate_from_geo(point: GeoPoint):
    """
    Uses an external AQI service (OpenAQ) as a proxy by fetching PM2.5 and mapping to AQI.
    If API fails, returns a graceful fallback.
    """
    try:
        # Query OpenAQ latest measurements near the coordinates
        url = (
            "https://api.openaq.org/v2/measurements?parameter=pm25&radius=20000&limit=1&"
            f"coordinates={point.lat},{point.lon}"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        if r.ok and data.get("results"):
            pm25 = data["results"][0]["value"]
            # Convert PM2.5 ug/m3 to AQI (EPA simplified approximation)
            def pm25_to_aqi(c):
                breakpoints = [
                    (0.0, 12.0, 0, 50),
                    (12.1, 35.4, 51, 100),
                    (35.5, 55.4, 101, 150),
                    (55.5, 150.4, 151, 200),
                    (150.5, 250.4, 201, 300),
                    (250.5, 350.4, 301, 400),
                    (350.5, 500.4, 401, 500),
                ]
                for Cl, Ch, Il, Ih in breakpoints:
                    if Cl <= c <= Ch:
                        return int((Ih - Il) / (Ch - Cl) * (c - Cl) + Il)
                return 500

            aqi = pm25_to_aqi(pm25)
            category = categorize_aqi(aqi)
            payload = {
                "aqi": aqi,
                "category": category,
                "metrics": {"pm25": pm25},
                "note": "Based on nearby PM2.5 from OpenAQ",
            }
        else:
            payload = {
                "aqi": None,
                "category": "Unknown",
                "note": "No nearby PM2.5 data found",
            }
    except Exception:
        payload = {"aqi": None, "category": "Unknown", "note": "Unable to fetch external data"}

    # Save to DB (best-effort)
    try:
        doc = Scan(
            source="geolocation",
            method="openaq-pm25",
            aqi=payload.get("aqi"),
            category=payload.get("category"),
            lat=point.lat,
            lon=point.lon,
            metrics=payload.get("metrics"),
            note=payload.get("note"),
        )
        create_document("scan", doc)
    except Exception:
        pass

    return payload


@app.get("/api/tips")
def tips():
    return {
        "tips": [
            "Wear a mask on Very Unhealthy days",
            "Use an air purifier indoors",
            "Avoid strenuous outdoor activity when AQI > 150",
            "Keep windows closed during high pollution hours",
        ]
    }


@app.get("/api/recommendations")
def get_recommendations(aqi: Optional[int] = Query(None), category: Optional[str] = Query(None)):
    cat = category or categorize_aqi(aqi) if (category or aqi is not None) else "Good"
    recs = recommendations_for_category(cat)
    return {"category": cat, "recommendations": recs}


@app.get("/api/history")
def history(limit: int = 10):
    try:
        docs = get_documents("scan", {}, limit=limit)
        # Convert ObjectId and datetimes to strings
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
            for k in ("created_at", "updated_at"):
                if k in d and hasattr(d[k], "isoformat"):
                    d[k] = d[k].isoformat()
        return {"items": docs}
    except Exception:
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
