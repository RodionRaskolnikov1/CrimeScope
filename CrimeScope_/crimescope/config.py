from pydantic_settings import BaseSettings
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    
    project_name: str = "CrimeScope"
    debug: bool = False
    
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    external_data_dir: Path = ROOT_DIR / "data" / "external"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    models_dir: Path = ROOT_DIR / "artifacts" / "models"
    chroma_dir: Path = ROOT_DIR / "artifacts" / "chroma_db"
    
    google_gemini_api_key: str = ""
    google_maps_api_key: str = ""
    open_meteo_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    
    crime_classifier_model: str = "xgboost"
    forecast_horizon_days: int = 30
    grid_size: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        
settings = Settings()


for path in [
    settings.raw_data_dir,
    settings.processed_data_dir,
    settings.external_data_dir,
    settings.models_dir,
    settings.chroma_dir,
]:
    path.mkdir(parents=True, exist_ok=True)