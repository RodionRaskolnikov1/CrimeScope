import chromadb
import polars as pl
import json
from pathlib import Path
from chromadb.utils import embedding_functions
from crimescope.config import settings
from crimescope.utils.logger import logger


# ── Constants ─────────────────────────────────────────────────────

COLLECTION_NAME = "crimescope_zones"


# ── ChromaDB Client ───────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Get persistent ChromaDB client saved to artifacts/chroma_db/"""
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection(client: chromadb.PersistentClient):
    """Get or create the crimescope collection."""
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Document Builder ──────────────────────────────────────────────

def build_zone_documents(
    validated_df: pl.DataFrame,
    forecast_results: dict,
    vision_df: pl.DataFrame | None = None,
) -> list[dict]:
    """
    Convert all zone data into text documents for ChromaDB.
    Each document = one zone's complete profile as natural language.
    This is what the RAG system searches through.

    Example document:
        "Zone 1434 is a high-crime area in Chicago.
         Total crimes: 8,234. Most common crime: THEFT (34%).
         Peak hours: 6PM-10PM. Highest crime day: Friday.
         Weather correlation: crime spikes when temp > 25°C.
         30-day forecast: 45 crimes/day expected.
         Visual risk score: 70.9/100 (dense urban, high edge density)."
    """

    logger.info("Building zone documents for ChromaDB...")
    documents = []

    # Get top zones from forecast
    top_zones = list(forecast_results["prophet"].keys())

    for zone_id in top_zones:
        zone_str = str(zone_id)
        zone_int = int(zone_id)

        # Filter crime data for this zone
        zone_df = validated_df.filter(pl.col("zone_id") == zone_int)

        if zone_df.is_empty():
            continue

        total_crimes = zone_df.shape[0]

        # Most common crime type
        top_crime = (
            zone_df.group_by("primary_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["primary_type"][0]
        )

        top_crime_pct = round(
            zone_df.filter(pl.col("primary_type") == top_crime).shape[0]
            / total_crimes * 100, 1
        )

        # Peak hour
        peak_hour = (
            zone_df.group_by("hour")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["hour"][0]
        )

        # Peak day
        day_names = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        peak_day_num = (
            zone_df.group_by("day_of_week")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["day_of_week"][0]
        )
        peak_day = day_names[peak_day_num]

        # Weekend vs weekday
        weekend_crimes = zone_df.filter(pl.col("is_weekend") == True).shape[0]
        weekend_pct = round(weekend_crimes / total_crimes * 100, 1)

        # Season with most crime
        season_names = ["Winter", "Spring", "Summer", "Fall"]
        peak_season_num = (
            zone_df.group_by("season")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["season"][0]
        )
        peak_season = season_names[peak_season_num]

        # Prophet forecast — avg predicted daily crimes
        forecast_df = forecast_results["prophet"][zone_str]["forecast"]
        future_only = forecast_df[
            forecast_df["ds"] > forecast_results["prophet"][zone_str]["actual"]["ds"].max()
        ]
        avg_forecast = round(future_only["yhat"].mean(), 1)
        max_forecast = round(future_only["yhat"].max(), 1)

        # Vision risk score
        risk_score = 50.0
        if vision_df is not None:
            vision_row = vision_df.filter(pl.col("zone_id") == zone_int)
            if not vision_row.is_empty():
                risk_score = vision_row["risk_score"][0]

        # Crime type breakdown
        crime_breakdown = (
            zone_df.group_by("primary_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(5)
        )
        breakdown_str = ", ".join([
            f"{row['primary_type']} ({round(row['count']/total_crimes*100, 1)}%)"
            for row in crime_breakdown.iter_rows(named=True)
        ])

        # Build natural language document
        doc_text = f"""
Zone {zone_id} Crime Analysis Report:

Location: Chicago urban zone {zone_id}, grid area covering approximately
{0.02:.2f} square degrees of the city.

Crime Volume: This zone recorded {total_crimes:,} crimes between 2021-2023,
making it one of the top 10 highest-crime zones in Chicago.

Crime Types: The most common crime is {top_crime} at {top_crime_pct}% of all incidents.
Full breakdown: {breakdown_str}.

Temporal Patterns: Crime peaks at {peak_hour}:00 hours ({"PM" if peak_hour >= 12 else "AM"}).
The highest crime day is {peak_day}. Weekend crimes account for {weekend_pct}% of total.
{peak_season} is the most dangerous season for this zone.

Forecast: Prophet model predicts an average of {avg_forecast} crimes per day
over the next 30 days, with a peak of up to {max_forecast} crimes on high-risk days.

Visual Risk Assessment: Urban risk score of {risk_score}/100 based on
map tile analysis using EfficientNet. Higher scores indicate denser urban
environments with less green space and more infrastructure complexity.

Safety Assessment: {"HIGH RISK ZONE" if risk_score > 65 else "MODERATE RISK ZONE" if risk_score > 45 else "LOWER RISK ZONE"} -
{"Avoid late night hours, especially weekends." if risk_score > 65 else "Exercise normal urban caution." if risk_score > 45 else "Generally safer area relative to other zones."}
        """.strip()

        documents.append({
            "id": f"zone_{zone_id}",
            "text": doc_text,
            "metadata": {
                "zone_id": zone_int,
                "total_crimes": total_crimes,
                "top_crime": top_crime,
                "peak_hour": int(peak_hour),
                "peak_day": peak_day,
                "risk_score": float(risk_score),
                "avg_forecast": float(avg_forecast),
            }
        })

    logger.success(f"Built {len(documents)} zone documents")
    return documents


# ── Indexing ──────────────────────────────────────────────────────

def index_documents(documents: list[dict]) -> None:
    """
    Add all zone documents to ChromaDB.
    ChromaDB automatically converts text → embeddings and stores them.
    """

    logger.info(f"Indexing {len(documents)} documents into ChromaDB...")

    client = get_chroma_client()
    collection = get_collection(client)

    # Clear existing documents — fresh index every run
    existing = collection.count()
    if existing > 0:
        collection.delete(ids=[d["id"] for d in documents
                               if collection.get(ids=[d["id"]])["ids"]])
        logger.debug(f"Cleared {existing} existing documents")

    collection.upsert(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )

    logger.success(f"Indexed {collection.count()} documents into ChromaDB ✓")


def run_embeddings(
    validated_df: pl.DataFrame,
    forecast_results: dict,
    vision_df: pl.DataFrame | None = None,
) -> None:
    """Build documents and index them into ChromaDB."""

    docs = build_zone_documents(validated_df, forecast_results, vision_df)
    index_documents(docs)