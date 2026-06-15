"""
Tests for crimescope/models/classifier.py
Run with: uv run pytest tests/test_classifier.py -v

NOTE: Tests that require a trained model are marked with @pytest.mark.requires_model
      Run all tests: uv run pytest tests/test_classifier.py -v
      Skip model tests: uv run pytest tests/test_classifier.py -v -m "not requires_model"
"""
import numpy as np
import polars as pl
import pytest
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import LabelEncoder

from crimescope.models.classifier import (
    FEATURE_COLS,
    TARGET_COL,
    prepare_features,
    enrich_features,
    predict,
)

SEVERITY_CLASSES = {"violent", "property", "other"}


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_validated_df():
    """
    Minimal validated dataframe matching what comes out of validation.py.
    All 17 feature columns (raw + cyclical + zone-rate) + severity target.
    """
    n = 100
    rng = np.random.default_rng(42)

    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    month = rng.integers(1, 13, n)

    return pl.DataFrame({
        "hour":          hour.tolist(),
        "day_of_week":   dow.tolist(),
        "month":         month.tolist(),
        "season":        rng.integers(0, 4, n).tolist(),
        "is_weekend":    rng.choice([True, False], n).tolist(),
        "zone_id":       rng.integers(0, 2500, n).tolist(),
        "temp_max":      rng.uniform(-10, 40, n).tolist(),
        "precipitation": rng.uniform(0, 30, n).tolist(),
        "windspeed":     rng.uniform(0, 50, n).tolist(),
        "hour_sin":      np.sin(2 * np.pi * hour / 24).tolist(),
        "hour_cos":      np.cos(2 * np.pi * hour / 24).tolist(),
        "dow_sin":       np.sin(2 * np.pi * dow / 7).tolist(),
        "dow_cos":       np.cos(2 * np.pi * dow / 7).tolist(),
        "month_sin":     np.sin(2 * np.pi * month / 12).tolist(),
        "month_cos":     np.cos(2 * np.pi * month / 12).tolist(),
        "zone_event_rate": rng.integers(1, 5000, n).tolist(),
        "zone_hour_rate":  rng.integers(1, 500, n).tolist(),
        "severity":      rng.choice(["violent", "property", "other"], n).tolist(),
    })


@pytest.fixture
def sample_features_dict():
    """Valid raw feature dict for a single prediction (API input shape)."""
    return {
        "hour": 22,
        "day_of_week": 5,
        "month": 7,
        "season": 2,
        "is_weekend": 1,
        "zone_id": 1434,
        "temp_max": 31.0,
        "precipitation": 0.0,
        "windspeed": 12.0,
    }


@pytest.fixture
def fake_zone_stats():
    """Stand-in zone-rate lookups so enrich_features works without a pipeline run."""
    return {
        "zone_event": {1434: 4200},
        "zone_hour": {(1434, 22): 310},
        "default_event": 1000.0,
        "default_hour": 80.0,
    }


# ── FEATURE_COLS constant tests ───────────────────────────────────

class TestFeatureCols:

    def test_feature_cols_count(self):
        assert len(FEATURE_COLS) == 17

    def test_all_expected_features_present(self):
        expected = {
            "hour", "day_of_week", "month", "season",
            "is_weekend", "zone_id", "temp_max",
            "precipitation", "windspeed",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "zone_event_rate", "zone_hour_rate",
        }
        assert set(FEATURE_COLS) == expected

    def test_target_col_not_in_features(self):
        assert TARGET_COL not in FEATURE_COLS

    def test_target_col_is_severity(self):
        assert TARGET_COL == "severity"

    def test_feature_cols_is_list(self):
        assert isinstance(FEATURE_COLS, list)


# ── prepare_features tests ────────────────────────────────────────

class TestPrepareFeatures:

    def test_returns_tuple_of_three(self, sample_validated_df):
        result = prepare_features(sample_validated_df)
        assert len(result) == 3

    def test_X_shape_correct(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert X.shape == (100, 17)

    def test_y_length_matches_rows(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert len(y) == 100

    def test_label_encoder_fitted(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert isinstance(le, LabelEncoder)
        assert hasattr(le, "classes_")

    def test_three_severity_classes_encoded(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert len(le.classes_) == 3

    def test_correct_class_names(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert set(le.classes_) == SEVERITY_CLASSES

    def test_X_is_numpy_array(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert isinstance(X, np.ndarray)

    def test_y_is_numpy_array(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert isinstance(y, np.ndarray)

    def test_null_weather_filled(self, sample_validated_df):
        # Inject some nulls into weather columns
        df_with_nulls = sample_validated_df.with_columns([
            pl.when(pl.col("temp_max") > 30)
              .then(None)
              .otherwise(pl.col("temp_max"))
              .alias("temp_max")
        ])
        # Should not raise
        X, y, le = prepare_features(df_with_nulls)
        assert not np.isnan(X[:, FEATURE_COLS.index("temp_max")]).any()

    def test_is_weekend_cast_to_int(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        weekend_col = FEATURE_COLS.index("is_weekend")
        unique_values = np.unique(X[:, weekend_col])
        # Should only be 0 or 1 after bool -> int8 cast
        assert all(v in [0, 1] for v in unique_values)

    def test_feature_count_matches_cols(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert X.shape[1] == len(FEATURE_COLS)


# ── enrich_features tests ─────────────────────────────────────────

class TestEnrichFeatures:

    def test_produces_full_feature_vector(self, sample_features_dict, fake_zone_stats):
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(sample_features_dict)
        # Every model feature must be present after enrichment
        for col in FEATURE_COLS:
            assert col in enriched

    def test_cyclical_values_in_range(self, sample_features_dict, fake_zone_stats):
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(sample_features_dict)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert -1.0 <= enriched[col] <= 1.0

    def test_is_weekend_coerced_to_int(self, sample_features_dict, fake_zone_stats):
        sample_features_dict["is_weekend"] = True
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(sample_features_dict)
        assert enriched["is_weekend"] == 1

    def test_known_zone_rates_looked_up(self, sample_features_dict, fake_zone_stats):
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(sample_features_dict)
        assert enriched["zone_event_rate"] == 4200
        assert enriched["zone_hour_rate"] == 310

    def test_unknown_zone_falls_back_to_default(self, fake_zone_stats):
        unknown = {
            "hour": 3, "day_of_week": 2, "month": 1, "season": 0,
            "is_weekend": 0, "zone_id": 999999,
            "temp_max": 10.0, "precipitation": 0.0, "windspeed": 5.0,
        }
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(unknown)
        assert enriched["zone_event_rate"] == fake_zone_stats["default_event"]
        assert enriched["zone_hour_rate"] == fake_zone_stats["default_hour"]

    def test_missing_weather_uses_defaults(self, fake_zone_stats):
        no_weather = {
            "hour": 22, "day_of_week": 5, "month": 7, "season": 2,
            "is_weekend": 1, "zone_id": 1434,
            "temp_max": None, "precipitation": None, "windspeed": None,
        }
        with patch("crimescope.models.classifier._load_zone_stats", return_value=fake_zone_stats):
            enriched = enrich_features(no_weather)
        assert enriched["temp_max"] == 20.0
        assert enriched["precipitation"] == 0.0
        assert enriched["windspeed"] == 10.0


# ── predict function tests ────────────────────────────────────────

class TestPredict:

    def test_raises_file_not_found_without_model(self, sample_features_dict, tmp_path):
        """When no trained model exists, predict should raise FileNotFoundError."""
        with patch("crimescope.models.classifier.MODEL_PATH", tmp_path / "nonexistent.pkl"):
            with pytest.raises(FileNotFoundError):
                predict(sample_features_dict)

    @pytest.mark.requires_model
    def test_predict_returns_dict(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert isinstance(result, dict)

    @pytest.mark.requires_model
    def test_predict_has_required_keys(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert "predicted_severity" in result
        assert "confidence" in result
        assert "probabilities" in result

    @pytest.mark.requires_model
    def test_predicted_severity_is_valid_class(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert result["predicted_severity"] in SEVERITY_CLASSES

    @pytest.mark.requires_model
    def test_confidence_between_zero_and_one(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.requires_model
    def test_probabilities_cover_all_classes(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert len(result["probabilities"]) == 3

    @pytest.mark.requires_model
    def test_probabilities_sum_to_one(self, sample_features_dict):
        result = predict(sample_features_dict)
        total = sum(item["probability"] for item in result["probabilities"])
        assert abs(total - 1.0) <= 0.01  # small float tolerance

    @pytest.mark.requires_model
    def test_probabilities_sorted_descending(self, sample_features_dict):
        result = predict(sample_features_dict)
        probs = [item["probability"] for item in result["probabilities"]]
        assert probs == sorted(probs, reverse=True)

    @pytest.mark.requires_model
    def test_predict_deterministic(self, sample_features_dict):
        """Same input should always produce same output."""
        result1 = predict(sample_features_dict)
        result2 = predict(sample_features_dict)
        assert result1["predicted_severity"] == result2["predicted_severity"]
        assert result1["confidence"] == result2["confidence"]
