from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()

SCORES_PATH = Path("artifacts/vision/zone_risk_scores.json")


@router.get("/zones")
async def get_zone_risk_scores():
    """Return all zone risk scores for the heatmap."""
    if not SCORES_PATH.exists():
        raise HTTPException(status_code=404, detail="Risk scores not found. Run the vision pipeline first.")

    with open(SCORES_PATH) as f:
        data = json.load(f)

    # Convert to list with coordinates
    from crimescope.vision.street_fetcher import zone_id_to_coords

    zones = []
    for zone_id, info in data.items():
        lat, lon = zone_id_to_coords(int(zone_id))
        zones.append({
            "zone_id": int(zone_id),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "risk_score": info["risk_score"],
            "darkness_score": info.get("darkness_score", 0),
            "edge_density": info.get("edge_density", 0),
            "green_ratio": info.get("green_ratio", 0),
            "gray_ratio": info.get("gray_ratio", 0),
            "risk_level": (
                "high" if info["risk_score"] > 65
                else "moderate" if info["risk_score"] > 50
                else "low"
            ),
        })

    # Summary stats
    scores = [z["risk_score"] for z in zones]
    summary = {
        "total_zones": len(zones),
        "high_risk": sum(1 for z in zones if z["risk_level"] == "high"),
        "moderate_risk": sum(1 for z in zones if z["risk_level"] == "moderate"),
        "low_risk": sum(1 for z in zones if z["risk_level"] == "low"),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
    }

    return {"zones": zones, "summary": summary}