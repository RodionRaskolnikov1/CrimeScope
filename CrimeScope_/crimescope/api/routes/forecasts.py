from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json

router = APIRouter()

FORECAST_DIR = Path("artifacts/forecasts")


@router.get("/zones")
async def list_forecast_zones():
    """List all zones that have forecast data available."""
    if not FORECAST_DIR.exists():
        raise HTTPException(status_code=404, detail="Forecast directory not found.")

    zone_files = sorted(FORECAST_DIR.glob("zone_*_forecast.png"))
    zone_ids = [
        f.stem.replace("_forecast", "").replace("zone_", "")
        for f in zone_files
    ]
    return {"zones": zone_ids, "count": len(zone_ids)}


@router.get("/citywide/image")
async def get_citywide_forecast_image():
    """Serve the citywide forecast PNG."""
    path = FORECAST_DIR / "citywide_forecast.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Citywide forecast image not found.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/zone/{zone_id}/image")
async def get_zone_forecast_image(zone_id: str):
    """Serve a zone's forecast PNG."""
    path = FORECAST_DIR / f"zone_{zone_id}_forecast.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast for zone {zone_id} not found.")
    return FileResponse(str(path), media_type="image/png")