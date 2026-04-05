"""
Tests for crimescope/models/forecaster.py
Run with: uv run pytest tests/test_forecaster.py -v
"""
import polars as pl
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from crimescope.models.forecaster import (
    prepare_time_series,
    train_prophet,
    TOP_N_ZONES,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_crime_df():
    """
    Synthetic validated crime dataframe spanning 2 years
    across 5 different zones. Matches what comes out of
    validation.py.
    """
    rng = np.random.default_rng(42)
    n = 500

    dates = pd.date_range("2021-01-01", periods=n, freq="4h")
    crime_dates = [d.date() for d in dates]

    return pl.DataFrame({
        "zone_id":      rng.choice([1434, 1484, 1483, 1433, 1384], n).tolist(),
        "crime_date":   crime_dates,
        "hour":         rng.integers(0, 24, n).tolist(),
        "primary_type": rng.choice(["THEFT", "BATTERY", "ASSAULT"], n).tolist(),
        "is_weekend":   rng.choice([True, False], n).tolist(),
    })


@pytest.fixture
def high_volume_crime_df():
    """
    Crime df where one zone dominates — tests top-N selection logic.
    Zone 9999 has 3x more crimes than others.
    """
    rng = np.random.default_rng(0)
    n_high = 300
    n_low = 50

    dates_high = pd.date_range("2021-01-01", periods=n_high, freq="6h")
    dates_low  = pd.date_range("2021-01-01", periods=n_low, freq="D")

    high_zone = pl.DataFrame({
        "zone_id":      [9999] * n_high,
        "crime_date":   [d.date() for d in dates_high],
        "hour":         rng.integers(0, 24, n_high).tolist(),
        "primary_type": ["THEFT"] * n_high,
        "is_weekend":   [False] * n_high,
    })

    low_zones = pl.DataFrame({
        "zone_id":      rng.choice([1, 2, 3, 4], n_low).tolist(),
        "crime_date":   [d.date() for d in dates_low],
        "hour":         rng.integers(0, 24, n_low).tolist(),
        "primary_type": ["BATTERY"] * n_low,
        "is_weekend":   [False] * n_low,
    })

    return pl.concat([high_zone, low_zones])


@pytest.fixture
def single_zone_series():
    """
    Pandas DataFrame with daily crime counts for one zone.
    Exactly what train_prophet receives.
    """
    dates = pd.date_range("2021-01-01", periods=365, freq="D")
    rng = np.random.default_rng(7)
    counts = rng.integers(3, 20, 365).tolist()

    return pd.DataFrame({
        "ds": dates,
        "y":  counts,
        "unique_id": ["1434"] * 365,
    })


# ── prepare_time_series tests ─────────────────────────────────────

class TestPrepareTimeSeries:

    def test_returns_polars_dataframe(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        assert isinstance(result, pl.DataFrame)

    def test_output_has_required_columns(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_y_values_are_positive(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        assert result["y"].min() > 0

    def test_selects_correct_number_of_zones(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        n_zones = result["unique_id"].n_unique()
        # Can't exceed TOP_N_ZONES or available zones
        assert n_zones <= TOP_N_ZONES
        assert n_zones <= sample_crime_df["zone_id"].n_unique()

    def test_aggregates_to_daily_counts(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        # Each row should be unique zone + date combination
        combo_count = result.select(["unique_id", "ds"]).unique().shape[0]
        assert combo_count == result.shape[0]

    def test_unique_id_is_string(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        assert result["unique_id"].dtype == pl.Utf8

    def test_ds_is_string(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        assert result["ds"].dtype == pl.Utf8

    def test_top_zone_by_volume_is_selected(self, high_volume_crime_df):
        result = prepare_time_series(high_volume_crime_df)
        selected_zones = result["unique_id"].unique().to_list()
        assert "9999" in selected_zones

    def test_sorted_by_zone_and_date(self, sample_crime_df):
        result = prepare_time_series(sample_crime_df)
        # After sorting, each zone's dates should be monotonically increasing
        for zone in result["unique_id"].unique().to_list():
            zone_dates = result.filter(pl.col("unique_id") == zone)["ds"].to_list()
            assert zone_dates == sorted(zone_dates)


# ── train_prophet tests ───────────────────────────────────────────

class TestTrainProphet:

    def test_returns_tuple_of_two(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        assert isinstance(model, object)
        assert isinstance(forecast, pd.DataFrame)

    def test_forecast_has_required_columns(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        for col in required_cols:
            assert col in forecast.columns, f"Missing forecast column: {col}"

    def test_forecast_horizon_respected(self, single_zone_series):
        horizon = 14
        model, forecast = train_prophet(single_zone_series, horizon=horizon)
        # forecast should have historical + future rows
        assert len(forecast) >= horizon

    def test_future_rows_count(self, single_zone_series):
        horizon = 30
        model, forecast = train_prophet(single_zone_series, horizon=horizon)
        future_rows = forecast[forecast["ds"] > single_zone_series["ds"].max()]
        assert len(future_rows) == horizon

    def test_yhat_is_numeric(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        assert pd.api.types.is_numeric_dtype(forecast["yhat"])

    def test_confidence_intervals_ordered(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        # yhat_lower should always be <= yhat <= yhat_upper
        assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
        assert (forecast["yhat"] <= forecast["yhat_upper"]).all()

    def test_different_horizons_produce_different_lengths(self, single_zone_series):
        _, forecast_7  = train_prophet(single_zone_series, horizon=7)
        _, forecast_30 = train_prophet(single_zone_series, horizon=30)
        future_7  = forecast_7[forecast_7["ds"] > single_zone_series["ds"].max()]
        future_30 = forecast_30[forecast_30["ds"] > single_zone_series["ds"].max()]
        assert len(future_30) > len(future_7)

    def test_no_nan_in_yhat(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        assert not forecast["yhat"].isna().any()

    def test_forecast_dates_are_daily(self, single_zone_series):
        model, forecast = train_prophet(single_zone_series, horizon=7)
        diffs = forecast["ds"].diff().dropna()
        # All date differences should be exactly 1 day
        assert (diffs == pd.Timedelta("1 day")).all()


# ── TOP_N_ZONES constant ──────────────────────────────────────────

class TestConstants:

    def test_top_n_zones_is_positive(self):
        assert TOP_N_ZONES > 0

    def test_top_n_zones_is_int(self):
        assert isinstance(TOP_N_ZONES, int)

    def test_top_n_zones_reasonable_value(self):
        # Should be between 1 and 100 for a reasonable analysis
        assert 1 <= TOP_N_ZONES <= 100