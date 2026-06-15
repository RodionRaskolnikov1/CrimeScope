import math
import polars as pl
import numpy as np
import joblib
from functools import lru_cache
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from crimescope.config import settings
from crimescope.data.preprocessing import ZONE_RATE_PATH, ZONE_HOUR_RATE_PATH
from crimescope.utils.logger import logger


FEATURE_COLS = [
    "hour", "day_of_week", "month", "season",
    "is_weekend", "zone_id", "temp_max",
    "precipitation", "windspeed",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "zone_event_rate", "zone_hour_rate",
]

# Raw API inputs the caller supplies; the rest of FEATURE_COLS is derived.
RAW_INPUT_COLS = [
    "hour", "day_of_week", "month", "season",
    "is_weekend", "zone_id", "temp_max",
    "precipitation", "windspeed",
]

# Fallbacks when an inference input omits weather.
WEATHER_DEFAULTS = {"temp_max": 20.0, "precipitation": 0.0, "windspeed": 10.0}


TARGET_COL = "severity"

MODEL_PATH = settings.models_dir / "crime_classifier.pkl"
ENCODER_PATH = settings.models_dir / "label_encoder.pkl"



def prepare_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:

    logger.info("Preparing features for training...")

    df = df.with_columns([
        pl.col("temp_max").fill_null(pl.col("temp_max").median()),
        pl.col("precipitation").fill_null(0.0),
        pl.col("windspeed").fill_null(pl.col("windspeed").median()),
        pl.col("is_weekend").cast(pl.Int8),  # bool → 0/1 for XGBoost
    ])

    X = df.select(FEATURE_COLS).to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].to_numpy())

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Classes: {list(le.classes_)}")

    return X, y, le


# ── Inference-time feature enrichment ──────────────────────────────

@lru_cache(maxsize=1)
def _load_zone_stats() -> dict:
    """Load zone-rate lookups once; cache for the process lifetime."""
    if not ZONE_RATE_PATH.exists() or not ZONE_HOUR_RATE_PATH.exists():
        raise FileNotFoundError(
            "Zone-rate lookups missing. Run the preprocessing pipeline first."
        )

    zone_rate = pl.read_parquet(ZONE_RATE_PATH)
    zone_hour_rate = pl.read_parquet(ZONE_HOUR_RATE_PATH)

    return {
        "zone_event": {
            z: r for z, r in zip(zone_rate["zone_id"], zone_rate["zone_event_rate"])
        },
        "zone_hour": {
            (z, h): r for z, h, r in zip(
                zone_hour_rate["zone_id"],
                zone_hour_rate["hour"],
                zone_hour_rate["zone_hour_rate"],
            )
        },
        # Median rates as fallbacks for unseen zones / zone-hours.
        "default_event": float(zone_rate["zone_event_rate"].median()),
        "default_hour": float(zone_hour_rate["zone_hour_rate"].median()),
    }


def enrich_features(raw: dict) -> dict:
    """
    Expand the raw API inputs into the full FEATURE_COLS vector:
    cyclical encodings + zone-activity rates from the saved lookups.
    Keeps the prediction API contract identical to before.
    """
    stats = _load_zone_stats()

    hour = raw["hour"]
    dow = raw["day_of_week"]
    month = raw["month"]
    zone_id = raw["zone_id"]

    out = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "season": raw["season"],
        "is_weekend": int(raw["is_weekend"]),
        "zone_id": zone_id,
        "temp_max": raw.get("temp_max") if raw.get("temp_max") is not None else WEATHER_DEFAULTS["temp_max"],
        "precipitation": raw.get("precipitation") if raw.get("precipitation") is not None else WEATHER_DEFAULTS["precipitation"],
        "windspeed": raw.get("windspeed") if raw.get("windspeed") is not None else WEATHER_DEFAULTS["windspeed"],
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "zone_event_rate": stats["zone_event"].get(zone_id, stats["default_event"]),
        "zone_hour_rate": stats["zone_hour"].get((zone_id, hour), stats["default_hour"]),
    }
    return out



# Fallback config used when tuning is disabled.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
)


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    sample_size: int,
) -> dict:
    """
    Search XGBoost hyperparameters with Optuna, optimizing macro-F1
    (3-fold CV). Macro-F1 weights all severity classes equally, so the
    search is pushed to do well on the minority classes rather than just
    the majority — the imbalance-aware objective.

    Runs on a random subsample for speed; the winning params are then
    refit on the full training set in train().
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if sample_size and len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        Xs, ys = X[idx], y[idx]
        logger.info(f"Tuning on a {sample_size:,}-row subsample ({n_trials} trials)...")
    else:
        Xs, ys = X, y
        logger.info(f"Tuning on all {len(X):,} rows ({n_trials} trials)...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        candidate = XGBClassifier(**params)
        scores = cross_val_score(candidate, Xs, ys, cv=cv, scoring="f1_macro", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.success(f"Best macro-F1 (tuning CV): {study.best_value:.3f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


def train(
    df: pl.DataFrame,
    tune: bool | None = None,
    n_trials: int | None = None,
) -> tuple[XGBClassifier, LabelEncoder, dict]:

    X, y, le = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Hyperparameter selection ──────────────────────────────────
    tune = settings.tune_hyperparams if tune is None else tune
    n_trials = settings.tune_trials if n_trials is None else n_trials

    if tune:
        best = tune_hyperparameters(
            X_train, y_train,
            n_trials=n_trials,
            sample_size=settings.tune_sample_size,
        )
    else:
        logger.info("Tuning disabled — using default XGBoost params.")
        best = dict(DEFAULT_XGB_PARAMS)

    xgb_params = {
        **best,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    # ── Class imbalance — weight samples inversely to class frequency ─
    sample_weight = compute_sample_weight("balanced", y_train)

    # ── Main model WITH early stopping ────────────────────────────
    model = XGBClassifier(
        **xgb_params,
        early_stopping_rounds=20,  # only works with eval_set
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── CV stability check WITHOUT early stopping ─────────────────
    # cross_val_score manages its own splits internally;
    # early_stopping needs an eval_set which CV can't provide.
    cv_model = XGBClassifier(**xgb_params)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cv_model, X, y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    # ── Evaluation ────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    metrics = {
        "accuracy": report["accuracy"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "cv_macro_mean": cv_scores.mean(),
        "cv_macro_std": cv_scores.std(),
        "best_params": best,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.success(f"Accuracy:        {metrics['accuracy']:.3f}")
    logger.success(f"F1 (weighted):   {metrics['f1_weighted']:.3f}")
    logger.success(f"F1 (macro):      {metrics['f1_macro']:.3f}")
    logger.success(f"CV macro-F1:     {metrics['cv_macro_mean']:.3f} ± {metrics['cv_macro_std']:.3f}")

    save(model, le)
    return model, le, metrics


def save(model: XGBClassifier, le: LabelEncoder) -> None:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    logger.success(f"Model saved → {MODEL_PATH}")
    logger.success(f"Encoder saved → {ENCODER_PATH}")
    
    
def load() -> tuple[XGBClassifier, LabelEncoder]:
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model found. Run train() first.")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    logger.info("Model and encoder loaded from disk")
    return model, le


def predict(features: dict) -> dict:

    model, le = load()

    # Expand raw inputs → full feature vector, then order by FEATURE_COLS
    enriched = enrich_features(features)
    X = np.array([[enriched[col] for col in FEATURE_COLS]])
    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    # All severity classes ranked by probability
    ranked = [
        {"severity": le.classes_[i], "probability": round(float(pred_proba[i]), 3)}
        for i in np.argsort(pred_proba)[::-1]
    ]

    return {
        "predicted_severity": le.classes_[pred_encoded],
        "confidence": round(float(pred_proba[pred_encoded]), 3),
        "probabilities": ranked,
    }
