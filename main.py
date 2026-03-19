import polars as pl
from crimescope.config import settings
from crimescope.data.ingestion import run_ingestion
from crimescope.data.preprocessing import run_preprocessing
from crimescope.data.validation import validate
from crimescope.models.classifier import train
from crimescope.models.explainability import explain_global
from crimescope.utils.logger import logger


def main():
    logger.info("🔍 CrimeScope Pipeline Starting...")

    # Week 1 — Data
    data = run_ingestion()
    processed_df = run_preprocessing(data["crime"], data["weather"])

     # ── Week 2 — ML ────────────────────────────────
    validated_df = validate(processed_df)
    model, le, metrics = train(validated_df)
    explain_global(validated_df)

    logger.success("✅ Pipeline complete!")
    logger.info(f"Final Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Final F1: {metrics['f1_weighted']:.3f}")
    logger.success(f"✅ Pipeline complete! {processed_df.shape[0]:,} records ready.")


if __name__ == "__main__":
    main()
