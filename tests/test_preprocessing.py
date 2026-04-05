"""
Tests for crimescope/data/preprocessing.py
Run with: uv run pytest tests/test_preprocessing.py -v
"""
import polars as pl
import pytest
from crimescope.data.preprocessing import (
    clean_crime_data,
    engineer_features,
    assign_grid_zones,
    merge_weather,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def raw_crime_df():
    """Minimal raw crime dataframe matching Chicago API schema."""
    return pl.DataFrame({
        "ID":                   [1, 2, 3, 4, 5],
        "Date":                 [
            "01/15/2022 10:30:00 PM",
            "06/20/2022 02:15:00 AM",
            "12/01/2021 08:00:00 AM",
            "07/04/2022 11:59:00 PM",
            "03/10/2022 06:45:00 PM",
        ],
        "Primary Type":         ["THEFT", "BATTERY", "ASSAULT", "NARCOTICS", "ROBBERY"],
        "Description":          ["FROM AUTO", "SIMPLE", "SIMPLE", "POSS", "ARMED"],
        "Location Description": ["STREET", "SIDEWALK", "PARK", "ALLEY", "STREET"],
        "Arrest":               [False, True, False, True, False],
        "Domestic":             [False, False, True, False, False],
        "Latitude":             [41.85, 41.90, 41.75, 42.00, 41.88],
        "Longitude":            [-87.65, -87.70, -87.60, -87.80, -87.62],
        "Year":                 [2022, 2022, 2021, 2022, 2022],
        "Community Area":       [1, 2, 3, 4, 5],
    })


@pytest.fixture
def raw_crime_with_nulls():
    """Dataframe with some null lat/lon rows that should be dropped."""
    return pl.DataFrame({
        "ID":                   [1, 2, 3],
        "Date":                 ["01/15/2022 10:30:00 PM"] * 3,
        "Primary Type":         ["THEFT", "BATTERY", "ASSAULT"],
        "Description":          ["FROM AUTO", "SIMPLE", "SIMPLE"],
        "Location Description": ["STREET"] * 3,
        "Arrest":               [False, True, False],
        "Domestic":             [False, False, True],
        "Latitude":             [41.85, None, 41.90],
        "Longitude":            [-87.65, -87.70, None],
        "Year":                 [2022, 2022, 2022],
        "Community Area":       [1, 2, 3],
    })


@pytest.fixture
def raw_crime_out_of_bounds():
    """Dataframe with GPS coordinates outside Chicago bounds."""
    return pl.DataFrame({
        "ID":                   [1, 2, 3],
        "Date":                 ["01/15/2022 10:30:00 PM"] * 3,
        "Primary Type":         ["THEFT"] * 3,
        "Description":          ["FROM AUTO"] * 3,
        "Location Description": ["STREET"] * 3,
        "Arrest":               [False] * 3,
        "Domestic":             [False] * 3,
        "Latitude":             [41.85, 40.00, 43.00],  # 40 and 43 are outside bounds
        "Longitude":            [-87.65, -87.65, -87.65],
        "Year":                 [2022] * 3,
        "Community Area":       [1, 2, 3],
    })


@pytest.fixture
def weather_df():
    """Minimal weather dataframe."""
    return pl.DataFrame({
        "date": pl.Series(["2022-01-15", "2022-06-20", "2021-12-01",
                           "2022-07-04", "2022-03-10"]).cast(pl.Date),
        "temp_max":      [2.0, 31.0, -5.0, 35.0, 15.0],
        "temp_min":      [-3.0, 22.0, -12.0, 27.0, 8.0],
        "precipitation": [0.0, 0.0, 5.0, 2.0, 0.0],
        "windspeed":     [15.0, 8.0, 25.0, 12.0, 10.0],
    })


# ── clean_crime_data tests ─────────────────────────────────────────

class TestCleanCrimeData:

    def test_columns_normalized_to_lowercase(self, raw_crime_df):
        result = clean_crime_data(raw_crime_df)
        for col in result.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_columns_have_no_spaces(self, raw_crime_df):
        result = clean_crime_data(raw_crime_df)
        for col in result.columns:
            assert " " not in col, f"Column '{col}' has spaces"

    def test_null_lat_lon_rows_dropped(self, raw_crime_with_nulls):
        result = clean_crime_data(raw_crime_with_nulls)
        assert result.shape[0] == 1
        assert result["latitude"][0] == 41.85

    def test_out_of_bounds_rows_dropped(self, raw_crime_out_of_bounds):
        result = clean_crime_data(raw_crime_out_of_bounds)
        assert result.shape[0] == 1
        assert result["latitude"][0] == 41.85

    def test_required_columns_present(self, raw_crime_df):
        result = clean_crime_data(raw_crime_df)
        required = ["latitude", "longitude", "date", "primary_type"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_polars_dataframe(self, raw_crime_df):
        result = clean_crime_data(raw_crime_df)
        assert isinstance(result, pl.DataFrame)

    def test_all_valid_rows_kept(self, raw_crime_df):
        result = clean_crime_data(raw_crime_df)
        assert result.shape[0] == 5


# ── engineer_features tests ───────────────────────────────────────

class TestEngineerFeatures:

    @pytest.fixture
    def cleaned_df(self, raw_crime_df):
        return clean_crime_data(raw_crime_df)

    def test_hour_column_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "hour" in result.columns

    def test_hour_range_valid(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23

    def test_day_of_week_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_month_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "month" in result.columns
        assert result["month"].min() >= 1
        assert result["month"].max() <= 12

    def test_season_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "season" in result.columns
        assert result["season"].min() >= 0
        assert result["season"].max() <= 3

    def test_is_weekend_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "is_weekend" in result.columns
        assert result["is_weekend"].dtype == pl.Boolean

    def test_time_of_day_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "time_of_day" in result.columns
        valid_values = {"morning", "afternoon", "evening", "night"}
        assert set(result["time_of_day"].unique().to_list()).issubset(valid_values)

    def test_zone_id_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "zone_id" in result.columns
        assert result["zone_id"].min() >= 0

    def test_grid_row_col_added(self, cleaned_df):
        result = engineer_features(cleaned_df)
        assert "grid_row" in result.columns
        assert "grid_col" in result.columns

    def test_no_rows_lost(self, cleaned_df):
        original_count = cleaned_df.shape[0]
        result = engineer_features(cleaned_df)
        assert result.shape[0] == original_count


# ── assign_grid_zones tests ───────────────────────────────────────

class TestAssignGridZones:

    def test_zone_id_within_bounds(self, raw_crime_df):
        cleaned = clean_crime_data(raw_crime_df)
        result = assign_grid_zones(cleaned)
        grid_size = 50
        assert result["zone_id"].min() >= 0
        assert result["zone_id"].max() < grid_size * grid_size

    def test_grid_row_col_within_bounds(self, raw_crime_df):
        cleaned = clean_crime_data(raw_crime_df)
        result = assign_grid_zones(cleaned)
        assert result["grid_row"].min() >= 0
        assert result["grid_col"].min() >= 0
        assert result["grid_row"].max() < 50
        assert result["grid_col"].max() < 50

    def test_zone_id_deterministic(self, raw_crime_df):
        cleaned = clean_crime_data(raw_crime_df)
        result1 = assign_grid_zones(cleaned)
        result2 = assign_grid_zones(cleaned)
        assert result1["zone_id"].to_list() == result2["zone_id"].to_list()

    def test_custom_grid_size(self, raw_crime_df):
        cleaned = clean_crime_data(raw_crime_df)
        result = assign_grid_zones(cleaned, grid_size=10)
        assert result["zone_id"].max() < 10 * 10


# ── merge_weather tests ───────────────────────────────────────────

class TestMergeWeather:

    @pytest.fixture
    def featured_df(self, raw_crime_df):
        cleaned = clean_crime_data(raw_crime_df)
        return engineer_features(cleaned)

    def test_weather_columns_added(self, featured_df, weather_df):
        result = merge_weather(featured_df, weather_df)
        assert "temp_max" in result.columns
        assert "precipitation" in result.columns
        assert "windspeed" in result.columns

    def test_row_count_preserved_after_left_join(self, featured_df, weather_df):
        original_count = featured_df.shape[0]
        result = merge_weather(featured_df, weather_df)
        assert result.shape[0] == original_count

    def test_returns_polars_dataframe(self, featured_df, weather_df):
        result = merge_weather(featured_df, weather_df)
        assert isinstance(result, pl.DataFrame)