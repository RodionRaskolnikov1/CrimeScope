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
    predict,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_validated_df():
    """
    Minimal validated dataframe matching what comes out of validation.py.
    All 9 feature columns + target column present.
    """
    n = 100
    rng = np.random.default_rng(42)

    return pl.DataFrame({
        "hour":          rng.integers(0, 24, n).tolist(),
        "day_of_week":   rng.integers(0, 7, n).tolist(),
        "month":         rng.integers(1, 13, n).tolist(),
        "season":        rng.integers(0, 4, n).tolist(),
        "is_weekend":    rng.choice([True, False], n).tolist(),
        "zone_id":       rng.integers(0, 2500, n).tolist(),
        "temp_max":      rng.uniform(-10, 40, n).tolist(),
        "precipitation": rng.uniform(0, 30, n).tolist(),
        "windspeed":     rng.uniform(0, 50, n).tolist(),
        "primary_type":  rng.choice([
            "THEFT", "BATTERY", "ASSAULT", "NARCOTICS", "ROBBERY",
            "BURGLARY", "CRIMINAL DAMAGE", "OTHER OFFENSE",
            "DECEPTIVE PRACTICE", "MOTOR VEHICLE THEFT"
        ], n).tolist(),
    })


@pytest.fixture
def sample_features_dict():
    """Valid feature dict for single prediction."""
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
def sample_features_with_nulls():
    """Feature dict with null weather values (common in real data)."""
    return {
        "hour": 10,
        "day_of_week": 1,
        "month": 3,
        "season": 1,
        "is_weekend": 0,
        "zone_id": 800,
        "temp_max": None,
        "precipitation": None,
        "windspeed": None,
    }


# ── FEATURE_COLS constant tests ───────────────────────────────────

class TestFeatureCols:

    def test_feature_cols_count(self):
        assert len(FEATURE_COLS) == 9

    def test_all_expected_features_present(self):
        expected = {
            "hour", "day_of_week", "month", "season",
            "is_weekend", "zone_id", "temp_max",
            "precipitation", "windspeed",
        }
        assert set(FEATURE_COLS) == expected

    def test_target_col_not_in_features(self):
        assert TARGET_COL not in FEATURE_COLS

    def test_feature_cols_is_list(self):
        assert isinstance(FEATURE_COLS, list)


# ── prepare_features tests ────────────────────────────────────────

class TestPrepareFeatures:

    def test_returns_tuple_of_three(self, sample_validated_df):
        result = prepare_features(sample_validated_df)
        assert len(result) == 3

    def test_X_shape_correct(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert X.shape == (100, 9)

    def test_y_length_matches_rows(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert len(y) == 100

    def test_label_encoder_fitted(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert isinstance(le, LabelEncoder)
        assert hasattr(le, "classes_")

    def test_all_ten_classes_encoded(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        assert len(le.classes_) == 10

    def test_correct_class_names(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        expected_classes = {
            "THEFT", "BATTERY", "ASSAULT", "NARCOTICS", "ROBBERY",
            "BURGLARY", "CRIMINAL DAMAGE", "OTHER OFFENSE",
            "DECEPTIVE PRACTICE", "MOTOR VEHICLE THEFT"
        }
        assert set(le.classes_) == expected_classes

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

    def test_features_in_correct_order(self, sample_validated_df):
        X, y, le = prepare_features(sample_validated_df)
        # hour should be first column
        assert X.shape[1] == len(FEATURE_COLS)


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
        assert "predicted_crime" in result
        assert "confidence" in result
        assert "top_3" in result

    @pytest.mark.requires_model
    def test_predicted_crime_is_valid_class(self, sample_features_dict):
        valid_classes = {
            "THEFT", "BATTERY", "ASSAULT", "NARCOTICS", "ROBBERY",
            "BURGLARY", "CRIMINAL DAMAGE", "OTHER OFFENSE",
            "DECEPTIVE PRACTICE", "MOTOR VEHICLE THEFT"
        }
        result = predict(sample_features_dict)
        assert result["predicted_crime"] in valid_classes

    @pytest.mark.requires_model
    def test_confidence_between_zero_and_one(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.requires_model
    def test_top_3_has_three_items(self, sample_features_dict):
        result = predict(sample_features_dict)
        assert len(result["top_3"]) == 3

    @pytest.mark.requires_model
    def test_top_3_probabilities_sum_less_than_one(self, sample_features_dict):
        result = predict(sample_features_dict)
        total = sum(item["probability"] for item in result["top_3"])
        assert total <= 1.01  # small float tolerance

    @pytest.mark.requires_model
    def test_top_3_sorted_by_probability_descending(self, sample_features_dict):
        result = predict(sample_features_dict)
        probs = [item["probability"] for item in result["top_3"]]
        assert probs == sorted(probs, reverse=True)

    @pytest.mark.requires_model
    def test_predict_deterministic(self, sample_features_dict):
        """Same input should always produce same output."""
        result1 = predict(sample_features_dict)
        result2 = predict(sample_features_dict)
        assert result1["predicted_crime"] == result2["predicted_crime"]
        assert result1["confidence"] == result2["confidence"]