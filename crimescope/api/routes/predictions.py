from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PredictionRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    season: int
    is_weekend: bool
    zone_id: int
    temp_max: float = 20.0
    precipitation: float = 0.0
    windspeed: float = 10.0


@router.post("/predict")
async def predict_crime(req: PredictionRequest):
    """Predict crime severity (violent / property / other) for given conditions."""
    try:
        from crimescope.models.classifier import predict
        result = predict(req.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run main.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_prediction(req: PredictionRequest):
    """Return SHAP explanation for a severity prediction."""
    try:
        from crimescope.models.explainability import explain_single
        result = explain_single(req.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))