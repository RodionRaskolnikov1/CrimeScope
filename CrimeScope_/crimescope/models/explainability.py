import shap
import numpy as np
import polars as pl
import joblib
import matplotlib.pyplot as plt
from crimescope.models.classifier import load, prepare_features, FEATURE_COLS
from crimescope.config import settings
from crimescope.utils.logger import logger


def get_explainer(model, X_sample: np.ndarray) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for our XGBoost model.
    TreeExplainer is specifically optimized for tree-based models.
    Much faster than generic KernelExplainer.
    """
    logger.info("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_global(df: pl.DataFrame) -> None:
    """
    Global explainability — which features matter MOST overall?
    Generates a SHAP summary bar plot and saves it.
    Shows feature importance across ALL predictions.
    """

    model, le = load()
    X, y, _ = prepare_features(df)

    # Use a sample for speed — SHAP on 600k rows is slow
    sample_idx = np.random.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X[sample_idx]

    explainer = get_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot — shows impact of each feature on predictions
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=FEATURE_COLS,
        class_names=le.classes_,
        plot_type="bar",
        show=False,
    )
    out = settings.artifacts_dir / "shap_global_importance.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    logger.success(f"Global SHAP plot saved → {out}")


def explain_single(features: dict) -> dict:
    """
    Local explainability — WHY did the model predict THIS
    specific crime type for THIS specific input?
    Returns SHAP values per feature for one prediction.
    """

    model, le = load()

    X = np.array([[features[col] for col in FEATURE_COLS]])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Get predicted class index
    pred_class = model.predict(X)[0]

    # SHAP values for the predicted class only
    sv = shap_values[pred_class][0]

    explanation = {
        "predicted_crime": le.classes_[pred_class],
        "feature_contributions": [
            {
                "feature": FEATURE_COLS[i],
                "value": float(features[FEATURE_COLS[i]]),
                "shap_value": round(float(sv[i]), 4),
                # Positive = pushed toward this prediction
                # Negative = pushed away from this prediction
                "impact": "increases risk" if sv[i] > 0 else "decreases risk",
            }
            for i in range(len(FEATURE_COLS))
        ],
    }

    # Sort by absolute SHAP value — biggest impact first
    explanation["feature_contributions"].sort(
        key=lambda x: abs(x["shap_value"]), reverse=True
    )

    return explanation