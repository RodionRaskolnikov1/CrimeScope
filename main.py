from crimescope.data.ingestion import run_ingestion
from crimescope.data.preprocessing import run_preprocessing
from crimescope.utils.logger import logger


def main():
    logger.info("🔍 CrimeScope Pipeline Starting...")

    # Week 1 — Data
    data = run_ingestion()
    processed_df = run_preprocessing(data["crime"], data["weather"])

    logger.success(f"✅ Pipeline complete! {processed_df.shape[0]:,} records ready.")


if __name__ == "__main__":
    main()
