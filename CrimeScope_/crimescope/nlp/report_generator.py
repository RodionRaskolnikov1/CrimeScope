import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from crimescope.config import settings
from crimescope.utils.logger import logger
import polars as pl
import json
from pathlib import Path


REPORTS_DIR = Path("artifacts/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_SYSTEM_PROMPT = """You are an urban safety analyst writing professional
neighborhood safety reports for Chicago. Write in a clear, factual tone.
Reports should be 3-4 paragraphs and include actionable safety recommendations."""


def generate_zone_report(zone_stats: dict) -> str:
    """
    Generate a professional safety report for one zone using Gemini.
    Input is a dict of zone statistics, output is a natural language report.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=settings.google_gemini_api_key,
        temperature=0.4,
        max_output_tokens=512,
    )

    prompt = f"""
Generate a professional urban safety report for Chicago Zone {zone_stats['zone_id']}.

Data:
- Total crimes (2021-2023): {zone_stats['total_crimes']:,}
- Most common crime: {zone_stats['top_crime']}
- Peak crime hour: {zone_stats['peak_hour']}:00
- Peak crime day: {zone_stats['peak_day']}
- Weekend crime share: {zone_stats.get('weekend_pct', 'N/A')}%
- Predicted daily crimes (next 30 days): {zone_stats['avg_forecast']}
- Visual urban risk score: {zone_stats['risk_score']}/100

Write a 3-paragraph safety report covering:
1. Current crime situation and patterns
2. Forecast and trend analysis
3. Safety recommendations for residents and visitors
    """.strip()

    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    content = response.content
    
    if isinstance(content, list):
        return content[0]["text"] if content else ""
    
    return content


def generate_all_reports(
    validated_df: pl.DataFrame,
    forecast_results: dict,
    vision_df: pl.DataFrame | None = None,
) -> dict[str, str]:
    """Generate safety reports for all top zones and save to files."""

    logger.info("Generating zone safety reports...")
    reports = {}
    top_zones = list(forecast_results["prophet"].keys())

    for zone_id in top_zones:
        zone_int = int(zone_id)
        zone_df = validated_df.filter(pl.col("zone_id") == zone_int)

        if zone_df.is_empty():
            continue

        total_crimes = zone_df.shape[0]
        top_crime = (
            zone_df.group_by("primary_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["primary_type"][0]
        )
        peak_hour = (
            zone_df.group_by("hour")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["hour"][0]
        )
        day_names = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        peak_day = day_names[
            zone_df.group_by("day_of_week")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(1)["day_of_week"][0]
        ]
        weekend_pct = round(
            zone_df.filter(pl.col("is_weekend") == True).shape[0]
            / total_crimes * 100, 1
        )
        forecast_df = forecast_results["prophet"][zone_id]["forecast"]
        actual_max = forecast_results["prophet"][zone_id]["actual"]["ds"].max()
        avg_forecast = round(
            forecast_df[forecast_df["ds"] > actual_max]["yhat"].mean(), 1
        )
        risk_score = 50.0
        if vision_df is not None:
            vrow = vision_df.filter(pl.col("zone_id") == zone_int)
            if not vrow.is_empty():
                risk_score = vrow["risk_score"][0]

        zone_stats = {
            "zone_id": zone_id,
            "total_crimes": total_crimes,
            "top_crime": top_crime,
            "peak_hour": int(peak_hour),
            "peak_day": peak_day,
            "weekend_pct": weekend_pct,
            "avg_forecast": avg_forecast,
            "risk_score": risk_score,
        }

        try:
            report = generate_zone_report(zone_stats)
            reports[zone_id] = report

            # Save individual report
            out = REPORTS_DIR / f"zone_{zone_id}_report.txt"
            out.write_text(report)
            logger.success(f"Zone {zone_id} report saved → {out}")
        

        except Exception as e:
            logger.warning(f"Zone {zone_id} report failed: {e}")

        time.sleep(4)
        
    # Save all reports as JSON
    json_out = REPORTS_DIR / "all_reports.json"
    json_out.write_text(json.dumps(reports, indent=2))
    logger.success(f"All reports saved → {json_out}")

    return reports


def run_report_generation(
    validated_df: pl.DataFrame,
    forecast_results: dict,
    vision_df: pl.DataFrame | None = None,
) -> dict:
    """Full report generation pipeline."""

    logger.info("=" * 50)
    logger.info("Starting Report Generation Pipeline")
    logger.info("=" * 50)

    return generate_all_reports(validated_df, forecast_results, vision_df)