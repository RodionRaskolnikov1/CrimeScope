import polars as pl
from crimescope.config import settings
from crimescope.data.ingestion import run_ingestion
from crimescope.data.preprocessing import run_preprocessing
from crimescope.data.validation import validate
from crimescope.models.classifier import train
from crimescope.models.explainability import explain_global
from crimescope.models.forecaster import run_forecasting
from crimescope.vision.risk_scorer import run_vision_pipeline
from crimescope.utils.logger import logger


def main():
    logger.info("🔍 CrimeScope Pipeline Starting...")

    # ── Week 1 — Data ──────────────────────────────
    data = run_ingestion()
    processed_df = run_preprocessing(data["crime"], data["weather"])

    # ── Week 2 — ML ────────────────────────────────
    validated_df = validate(processed_df)
    model, le, metrics = train(validated_df)
    explain_global(validated_df)

    # ── Week 3 — Forecasting ───────────────────────
    forecast_results = run_forecasting(validated_df, horizon=30)

    # ── Week 4 — Computer Vision ───────────────────
    top_zones = (
        validated_df
        .group_by("zone_id")
        .agg(pl.len().alias("total"))
        .sort("total", descending=True)
        .head(10)
        ["zone_id"]
        .to_list()
    )
    vision_df = run_vision_pipeline(top_zones)
    logger.info(f"Vision scores:\n{vision_df}")

    logger.success("✅ Pipeline complete!")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"F1 Score: {metrics['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()