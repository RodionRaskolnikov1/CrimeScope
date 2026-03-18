import polars as pl
from crimescope.config import settings
from crimescope.utils.logger import logger


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
    
    df = assign_grid_zones(df)
    
    logger.success(f"Features engineered. Columns now: {df.columns}")
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
    
    