import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
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

    def _main_():
        main()

    _main_()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(
        f"""

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    print("hello")
    return


@app.cell
def _():
    a = 1
    print(a)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
