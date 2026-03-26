import polars as pl
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from prophet import Prophet
from crimescope.config import settings
from crimescope.utils.logger import logger


# ── Constants ─────────────────────────────────────────────────────

FORECAST_DIR = settings.artifacts_dir / "forecasts"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

# Top zones by crime volume — we forecast these
TOP_N_ZONES = 10


# ── Data Prep ─────────────────────────────────────────────────────

def prepare_time_series(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate crime records into daily counts per zone.
    Prophet needs: ds (date), y (value), unique_id (series id)
    NeuralForecast needs the same format.

    Output shape:
        zone_id | ds         | y
        1200    | 2021-01-01 | 14
        1200    | 2021-01-02 | 9
        ...
    """

    logger.info("Preparing time series data...")

    # Get top N zones by total crime volume
    top_zones = (
        df.group_by("zone_id")
        .agg(pl.len().alias("total"))
        .sort("total", descending=True)
        .head(TOP_N_ZONES)
        ["zone_id"]
        .to_list()
    )

    logger.info(f"Top {TOP_N_ZONES} zones selected: {top_zones}")

    # Filter to top zones only
    df_filtered = df.filter(pl.col("zone_id").is_in(top_zones))

    # Aggregate: count crimes per zone per day
    daily = (
        df_filtered
        .group_by(["zone_id", "crime_date"])
        .agg(pl.len().alias("y"))
        .sort(["zone_id", "crime_date"])
        .rename({"crime_date": "ds", "zone_id": "unique_id"})
        .with_columns([
            pl.col("ds").cast(pl.Utf8),      # Prophet needs string dates
            pl.col("unique_id").cast(pl.Utf8) # needs string IDs
        ])
    )

    logger.success(f"Time series prepared: {daily.shape[0]} rows across {TOP_N_ZONES} zones")
    return daily


# ── Prophet ───────────────────────────────────────────────────────

def train_prophet(
    series: pd.DataFrame,
    horizon: int = 30,
) -> tuple[Prophet, pd.DataFrame]:
    """
    Train Prophet on a single zone's time series.
    Returns fitted model and forecast DataFrame.

    Prophet automatically handles:
    - Weekly seasonality (more crimes on weekends)
    - Yearly seasonality (more crimes in summer)
    - Holiday effects
    - Trend changes
    """

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,    # daily is too noisy for crime data
        seasonality_mode="multiplicative",  # crimes scale with baseline
        changepoint_prior_scale=0.05,       # controls trend flexibility
        interval_width=0.95,               # 95% confidence interval
    )

    model.fit(series[["ds", "y"]])

    # Create future dates dataframe
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)

    return model, forecast


def run_prophet_all_zones(
    daily_df: pl.DataFrame,
    horizon: int = 30,
) -> dict:
    """
    Train Prophet for each of the top N zones.
    Returns dict of {zone_id: forecast_df}
    """

    logger.info(f"Training Prophet for {TOP_N_ZONES} zones (horizon={horizon} days)...")

    zones = daily_df["unique_id"].unique().to_list()
    results = {}

    for zone in zones:
        # Filter to this zone, convert to pandas (Prophet needs pandas)
        zone_df = (
            daily_df
            .filter(pl.col("unique_id") == zone)
            .to_pandas()
        )
        zone_df["ds"] = pd.to_datetime(zone_df["ds"])
        zone_df = zone_df.sort_values("ds")

        try:
            model, forecast = train_prophet(zone_df, horizon=horizon)
            results[zone] = {
                "model": model,
                "forecast": forecast,
                "actual": zone_df,
            }
            logger.debug(f"Zone {zone} → Prophet trained ✓")

        except Exception as e:
            logger.warning(f"Zone {zone} Prophet failed: {e}")

    logger.success(f"Prophet complete for {len(results)} zones")
    return results


# ── Visualization ─────────────────────────────────────────────────

def plot_zone_forecast(
    zone_id: str,
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> Path:
    """
    Create an interactive Plotly chart for one zone.
    Shows: actual crime counts + Prophet forecast + confidence interval.
    Saves as PNG to artifacts/forecasts/
    """

    fig = go.Figure()

    # Actual crime counts (last 90 days for clarity)
    recent_actual = actual_df.tail(90)
    fig.add_trace(go.Scatter(
        x=recent_actual["ds"],
        y=recent_actual["y"],
        mode="lines+markers",
        name="Actual crimes",
        line=dict(color="#3B8BD4", width=2),
        marker=dict(size=4),
    ))

    # Prophet forecast (future only)
    future_forecast = forecast_df[forecast_df["ds"] > actual_df["ds"].max()]

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast["ds"], future_forecast["ds"].iloc[::-1]]),
        y=pd.concat([future_forecast["yhat_upper"], future_forecast["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(239,159,39,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% confidence",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat"].clip(lower=0),  # no negative crimes
        mode="lines",
        name="Forecast",
        line=dict(color="#EF9F27", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=f"Crime Forecast — Zone {zone_id} (next 30 days)",
        xaxis_title="Date",
        yaxis_title="Daily crime count",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    out = FORECAST_DIR / f"zone_{zone_id}_forecast.png"
    fig.write_image(str(out), width=900, height=400)
    logger.success(f"Chart saved → {out}")
    return out


def plot_all_zones(prophet_results: dict) -> None:
    """Generate forecast charts for all zones."""

    logger.info("Generating forecast charts...")
    for zone_id, data in prophet_results.items():
        plot_zone_forecast(
            zone_id=zone_id,
            actual_df=data["actual"],
            forecast_df=data["forecast"],
        )
    logger.success(f"All charts saved to {FORECAST_DIR}")


# ── Combined summary forecast ──────────────────────────────────────

def citywide_forecast(daily_df: pl.DataFrame, horizon: int = 30) -> Path:
    """
    Aggregate all zones into a single citywide forecast.
    Shows the big picture — total daily crimes across Chicago.
    """

    logger.info("Building citywide forecast...")

    # Sum all zones per day
    citywide = (
        daily_df
        .with_columns(pl.col("ds").str.to_date())
        .group_by("ds")
        .agg(pl.col("y").sum())
        .sort("ds")
        .to_pandas()
    )
    citywide.columns = ["ds", "y"]
    citywide["ds"] = pd.to_datetime(citywide["ds"])

    model, forecast = train_prophet(citywide, horizon=horizon)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=citywide["ds"],
        y=citywide["y"],
        mode="lines",
        name="Actual (all zones)",
        line=dict(color="#3B8BD4", width=2),
    ))

    future = forecast[forecast["ds"] > citywide["ds"].max()]

    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(239,159,39,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% confidence",
    ))

    fig.add_trace(go.Scatter(
        x=future["ds"],
        y=future["yhat"].clip(lower=0),
        mode="lines",
        name="30-day forecast",
        line=dict(color="#EF9F27", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title="CrimeScope — Citywide Crime Forecast (Top 10 Zones)",
        xaxis_title="Date",
        yaxis_title="Daily crime count",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    out = FORECAST_DIR / "citywide_forecast.png"
    fig.write_image(str(out), width=1000, height=450)
    logger.success(f"Citywide forecast saved → {out}")
    return out


# ── Main runner ───────────────────────────────────────────────────

def run_forecasting(df: pl.DataFrame, horizon: int = 30) -> dict:
    """Full forecasting pipeline — Prophet only."""

    logger.info("=" * 50)
    logger.info("Starting Forecasting Pipeline")
    logger.info("=" * 50)

    # Prepare aggregated daily data
    daily_df = prepare_time_series(df)

    # Prophet per zone
    prophet_results = run_prophet_all_zones(daily_df, horizon=horizon)

    # Charts per zone
    plot_all_zones(prophet_results)

    # Citywide summary chart
    citywide_forecast(daily_df, horizon=horizon)

    # NOTE: NeuralForecast LSTM skipped on Windows
    # Will be added on Linux/cloud deployment

    return {
        "prophet": prophet_results,
        "daily_df": daily_df,
    }