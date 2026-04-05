import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame, Series
from crimescope.utils.logger import logger


class CrimeSchema(pa.DataFrameModel):
    
    hour:           Series[int]  = pa.Field(ge=0, le=23)
    day_of_week:    Series[int]  = pa.Field(ge=0, le=6)
    month:          Series[int]  = pa.Field(ge=1, le=12)
    season:         Series[int]  = pa.Field(ge=0, le=3)
    is_weekend:     Series[bool]
    
    zone_id:        Series[int]  = pa.Field(ge=0)
    latitude:       Series[float] = pa.Field(ge=41.6, le=42.1)
    longitude:      Series[float] = pa.Field(ge=-87.9, le=-87.5)
    
    temp_max:       Series[float] = pa.Field(nullable=True)
    precipitation:  Series[float] = pa.Field(ge=0, nullable=True)
    windspeed:      Series[float] = pa.Field(ge=0, nullable=True)

    primary_type:   Series[str]  = pa.Field(isin=[
        "THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT",
        "DECEPTIVE PRACTICE", "OTHER OFFENSE", "NARCOTICS",
        "BURGLARY", "MOTOR VEHICLE THEFT", "ROBBERY",
    ])
    
    
    class Config:
        coerce = True       
        drop_invalid_rows = True 
        
    
def validate(df: pl.DataFrame) -> pl.DataFrame:
    
    original_count = df.shape[0]
    logger.info(f"Validating {original_count:,} rows...")
    
    try:
        validated_df = CrimeSchema.validate(df, lazy=True)
        dropped = original_count - validated_df.shape[0]

        if dropped > 0:
            logger.warning(f"Dropped {dropped:,} invalid rows during validation")
        else:
            logger.success("All rows passed validation ✓")

        return validated_df

    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation failed: {e}")
        raise