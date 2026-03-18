import httpx
import polars as pl
from pathlib import Path

from crimescope.config import settings
from crimescope.utils.logger import logger


CHICAGO_CRIME_URL = (
    "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
    "?$where=year>=2021"
    "&$limit=700000"
    "&$order=date DESC"
)

def download_chicago_crime(force: bool = False) -> Path:
    
    output_path = settings.raw_data_dir / "chicago_crime_raw.csv"
    
    if output_path.exists() and not force:
        logger.info(f"Crime data already exists at {output_path}, skipping download.")
        return output_path
    
    
    logger.info("Starting Chicago crime data download... (this may take a few minutes)")
    
    with httpx.stream("GET", CHICAGO_CRIME_URL, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = 0
        with open(output_path, "wb") as f:        
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    total += len(chunk)
                    logger.debug(f"Downloaded {total / 1024 / 1024:.1f} MB...")
                    
    
    logger.success(f"Download complete → {output_path}")
    return output_path



def load_chicago_crime(path: Path | None = None) -> pl.DataFrame:
    
    path = path or settings.raw_data_dir / "chicago_crime_raw.csv"
    logger.info(f"Loading crime data from {path}")
                    
    
    df = pl.read_csv(
        path,
        try_parse_dates=True,
        ignore_errors=True,
        infer_schema_length=10000,
    )                
    
    logger.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df




def fetch_weather_data(
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> pl.DataFrame:
    

    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    
    
    params = {
        "latitude": 41.85,
        "longitude": -87.65,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
        ],
        "timezone": "America/Chicago",
    }
    
    
    response = httpx.get(
        settings.open_meteo_base_url,
        params=params,
        timeout=60,
    )
    
    response.raise_for_status()
    data = response.json()
    
    
    df = pl.DataFrame({
        "date": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"],
        "windspeed": data["daily"]["windspeed_10m_max"],
    }).with_columns(
        pl.col("date").str.to_date()
    )
    
    out = settings.external_data_dir / "chicago_weather.parquet"
    df.write_parquet(out)
    logger.success(f"Weather data saved → {out} ({df.shape[0]} rows)")
    return df




def run_ingestion() -> dict:
    
    logger.info("=" * 50)
    logger.info("Starting CrimeScope Ingestion Pipeline")
    logger.info("=" * 50)

    crime_path = download_chicago_crime()
    crime_df = load_chicago_crime(crime_path)
    weather_df = fetch_weather_data()

    return {
        "crime": crime_df,
        "weather": weather_df,
    }
    
    

if __name__ == "__main__":
    result = run_ingestion()
    print(result["crime"].head(5))
    print(result["weather"].head(5))    