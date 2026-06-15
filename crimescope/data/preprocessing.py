import math
import polars as pl
from crimescope.config import settings
from crimescope.utils.logger import logger


# ── Severity grouping ──────────────────────────────────────────────
# Collapse the 10 Chicago primary_type categories into 3 severity
# classes. Predicting severity (violent / property / other) is both
# more learnable and more actionable than predicting the exact crime
# type, which is near-random given only location + time + weather.
SEVERITY_MAP = {
    "BATTERY": "violent",
    "ASSAULT": "violent",
    "ROBBERY": "violent",
    "THEFT": "property",
    "BURGLARY": "property",
    "MOTOR VEHICLE THEFT": "property",
    "CRIMINAL DAMAGE": "property",
    "NARCOTICS": "other",
    "DECEPTIVE PRACTICE": "other",
    "OTHER OFFENSE": "other",
}

SEVERITY_CLASSES = ["violent", "property", "other"]

# Lookup tables (one value per zone / per zone-hour) persisted at
# preprocessing time so the stateless prediction API can recover the
# same zone-activity features it was trained on.
ZONE_RATE_PATH = settings.models_dir / "zone_event_rate.parquet"
ZONE_HOUR_RATE_PATH = settings.models_dir / "zone_hour_rate.parquet"


def clean_crime_data(df: pl.DataFrame) -> pl.DataFrame:
    
    logger.info("Cleaning crime data...")
    
    df = df.rename({col: col.lower().replace(" ", "_") for col in df.columns})
    
    keep_cols = [
        "id", "date", "primary_type", "description",
        "location_description", "arrest", "domestic",
        "latitude", "longitude", "year", "community_area"
    ]
    
    
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(keep_cols)
    
    df = df.drop_nulls(subset=["latitude", "longitude", "date"])
    
    df = df.filter(
        (pl.col("latitude").is_between(41.6, 42.1)) &
        (pl.col("longitude").is_between(-87.9, -87.5))
    )
    
    
    logger.success(f"After cleaning: {df.shape[0]:,} rows remaining")
    return df

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    
    logger.info("Engineering features...")
    
    if df["date"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p")
        )
        

    df = df.with_columns([
        pl.col("date").dt.hour().alias("hour"),
        pl.col("date").dt.weekday().alias("day_of_week"),   
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.date().alias("crime_date"),
        (pl.col("date").dt.weekday() >= 5).alias("is_weekend"),
        
        (
            pl.when(pl.col("date").dt.month().is_in([12, 1, 2])).then(0)
            .when(pl.col("date").dt.month().is_in([3, 4, 5])).then(1)
            .when(pl.col("date").dt.month().is_in([6, 7, 8])).then(2)
            .otherwise(3)
        ).alias("season"),
        
        
        (
            pl.when(pl.col("date").dt.hour().is_between(6, 11)).then(pl.lit("morning"))
            .when(pl.col("date").dt.hour().is_between(12, 17)).then(pl.lit("afternoon"))
            .when(pl.col("date").dt.hour().is_between(18, 21)).then(pl.lit("evening"))
            .otherwise(pl.lit("night"))
        ).alias("time_of_day"),

    ])

    # Cyclical encoding — hour 23 and hour 0 are adjacent, not 23 apart.
    # sin/cos pairs give the model that wrap-around structure.
    df = df.with_columns([
        (pl.col("hour") * (2 * math.pi / 24)).sin().alias("hour_sin"),
        (pl.col("hour") * (2 * math.pi / 24)).cos().alias("hour_cos"),
        (pl.col("day_of_week") * (2 * math.pi / 7)).sin().alias("dow_sin"),
        (pl.col("day_of_week") * (2 * math.pi / 7)).cos().alias("dow_cos"),
        (pl.col("month") * (2 * math.pi / 12)).sin().alias("month_sin"),
        (pl.col("month") * (2 * math.pi / 12)).cos().alias("month_cos"),
    ])

    # Severity target derived from primary_type. Rows whose type isn't in
    # the canonical 10 fall through to null and are dropped at validation.
    df = df.with_columns(
        pl.col("primary_type").replace_strict(
            SEVERITY_MAP, default=None, return_dtype=pl.Utf8
        ).alias("severity")
    )

    df = assign_grid_zones(df)

    logger.success(f"Features engineered. Columns now: {df.columns}")
    return df


def add_zone_rate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add historical zone-activity features and persist them as lookup
    tables for inference.

    - zone_event_rate: total crime volume in the zone (how busy the zone is)
    - zone_hour_rate:  crime volume for that zone at that hour of day

    These are the deployable form of "lag" features: true per-event
    rolling counts (last-7-days, etc.) can't be served by the stateless
    prediction API, so we use the historical zone / zone-hour rates that
    the API can recover from a saved lookup keyed by (zone_id, hour).
    """

    logger.info("Computing zone-rate features...")

    zone_rate = df.group_by("zone_id").agg(pl.len().alias("zone_event_rate"))
    zone_hour_rate = df.group_by(["zone_id", "hour"]).agg(
        pl.len().alias("zone_hour_rate")
    )

    df = (
        df.join(zone_rate, on="zone_id", how="left")
          .join(zone_hour_rate, on=["zone_id", "hour"], how="left")
    )

    zone_rate.write_parquet(ZONE_RATE_PATH)
    zone_hour_rate.write_parquet(ZONE_HOUR_RATE_PATH)
    logger.success(
        f"Zone-rate lookups saved → {ZONE_RATE_PATH.name}, {ZONE_HOUR_RATE_PATH.name}"
    )

    return df

def assign_grid_zones(df: pl.DataFrame, grid_size: int = None) -> pl.DataFrame:

    grid_size = grid_size or settings.grid_size
    
    lat_min, lat_max = 41.6, 42.1
    lon_min, lon_max = -87.9, -87.5
    
    df = df.with_columns([
        (
            ((pl.col("latitude") - lat_min) / (lat_max - lat_min) * grid_size)
            .cast(pl.Int32)
            .clip(0, grid_size - 1)
        ).alias("grid_row"),

        (
            ((pl.col("longitude") - lon_min) / (lon_max - lon_min) * grid_size)
            .cast(pl.Int32)
            .clip(0, grid_size - 1)
        ).alias("grid_col"),
    ]).with_columns(
        (pl.col("grid_row") * grid_size + pl.col("grid_col")).alias("zone_id")
    )

    return df

def merge_weather(crime_df: pl.DataFrame, weather_df: pl.DataFrame) -> pl.DataFrame:
    
    logger.info("Merging weather data...")
    
    
    merged = crime_df.join(
        weather_df,
        left_on="crime_date",
        right_on="date",
        how="left"
    )
    
    logger.success(f"Merged. Shape: {merged.shape}")
    return merged

def save_processed(df: pl.DataFrame, filename: str = "crime_processed.parquet") -> None:
    
    out = settings.processed_data_dir / filename
    df.write_parquet(out)
    logger.success(f"Processed data saved → {out}")
    logger.info(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

def run_preprocessing(crime_df: pl.DataFrame, weather_df: pl.DataFrame) -> pl.DataFrame:
    
    logger.info("=" * 50)
    logger.info("Starting Preprocessing Pipeline")
    logger.info("=" * 50)

    df = clean_crime_data(crime_df)
    df = engineer_features(df)
    df = add_zone_rate_features(df)
    df = merge_weather(df, weather_df)
    save_processed(df)

    return df

if __name__ == "__main__":
    from crimescope.data.ingestion import load_chicago_crime
    from pathlib import Path

    crime_df = load_chicago_crime()
    weather_df = pl.read_parquet(
        settings.external_data_dir / "chicago_weather.parquet"
    )
    final_df = run_preprocessing(crime_df, weather_df)
    print(final_df.head(5))
    
    