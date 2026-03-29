# CrimeScope — Chicago Urban Safety Intelligence System

> AI-powered crime analysis platform combining machine learning, time-series forecasting, computer vision, and RAG-powered natural language queries over 546,160 real Chicago crime records.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Prophet](https://img.shields.io/badge/Prophet-1.1-red?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-Flash-4285F4?style=flat-square&logo=google)
![uv](https://img.shields.io/badge/uv-package%20manager-purple?style=flat-square)

---

## Screenshots

| Heatmap | Forecast |
|---|---|
| ![Heatmap](docs/screenshots/heatmap.png) | ![Forecast](docs/screenshots/forecast.png) |

| Risk Scores | Ask AI |
|---|---|
| ![Risk](docs/screenshots/risk_scores.png) | ![Chat](docs/screenshots/chat.png) |

---

## What It Does

CrimeScope ingests and analyzes real Chicago crime data across six AI/ML modules:

| Module | What it does |
|---|---|
| **Data Pipeline** | Downloads 700k+ crime records, merges with weather data, engineers 15+ features using Polars |
| **ML Classifier** | XGBoost model predicts crime type (10 categories) from location, time, and weather features |
| **Explainability** | SHAP values explain why the model made each prediction — which features drove the outcome |
| **Forecasting** | Facebook Prophet forecasts crime volume 30 days ahead per zone and citywide |
| **Computer Vision** | EfficientNet-B0 (TIMM) scores urban risk from map tile images using darkness, edge density, and vegetation signals |
| **RAG Chat** | LangChain + ChromaDB + Gemini Flash answers natural language questions over indexed zone intelligence reports |

Everything is served through a **FastAPI REST backend** with a **vanilla HTML/CSS/JS frontend**.

---

## Architecture

```
Chicago Crime Data (700k rows)
         +  Open-Meteo Weather API
         +  OpenStreetMap tile images
                    │
                    ▼
        ┌─────────────────────┐
        │   Data Pipeline      │  Polars · Parquet · Pandera
        └──────────┬──────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
      ┌───────┐ ┌──────┐ ┌────────┐
      │  ML   │ │ Time │ │Vision  │
      │Classi-│ │Series│ │(TIMM + │
      │ fier  │ │(Pro- │ │OpenCV) │
      │+SHAP  │ │phet) │ │        │
      └───┬───┘ └──┬───┘ └───┬────┘
          │        │         │
          └────────┼─────────┘
                   │
          ┌────────┴────────┐
          │   LLM / RAG     │  LangChain · ChromaDB · Gemini
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          │   FastAPI REST  │  4 route groups · auto docs
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          │HTML/CSS/JS SPA  │  Leaflet · Chart.js · DM fonts
          └─────────────────┘
```

---

## Tech Stack

### Data & ML
| Tool | Purpose |
|---|---|
| **Polars** | Dataframe processing — 10–100x faster than pandas |
| **Pandera** | Schema validation before model training |
| **XGBoost** | Crime type classifier (10 categories, 546k training rows) |
| **SHAP** | Model explainability — TreeExplainer for XGBoost |
| **Prophet** | Time-series forecasting with seasonality decomposition |
| **TIMM + EfficientNet-B0** | Pretrained vision model for urban risk scoring |
| **OpenCV** | Visual feature extraction (darkness, edges, vegetation) |

### LLM / RAG
| Tool | Purpose |
|---|---|
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Vector store for zone intelligence documents |
| **FastEmbed** | Local embedding generation |
| **Gemini Flash** | LLM for Q&A and report generation |

### Backend & Frontend
| Tool | Purpose |
|---|---|
| **FastAPI** | REST API with auto-generated OpenAPI docs |
| **Uvicorn** | ASGI server |
| **Leaflet.js** | Interactive crime zone map |
| **Chart.js** | Forecast visualizations |
| **Vanilla JS** | SPA routing and API calls |

### Dev Tooling
| Tool | Purpose |
|---|---|
| **uv** | Package manager (10–100x faster than pip) |
| **Loguru** | Structured logging |
| **Pydantic-Settings** | Type-safe config from `.env` |
| **Ruff** | Linter + formatter |

---

## Project Structure

```
crimescope/
├── crimescope/                  # Main Python package
│   ├── config.py                # Pydantic settings
│   ├── data/
│   │   ├── ingestion.py         # Data download + loading
│   │   ├── preprocessing.py     # Cleaning + feature engineering
│   │   └── validation.py        # Pandera schema validation
│   ├── models/
│   │   ├── classifier.py        # XGBoost training + inference
│   │   ├── explainability.py    # SHAP explanations
│   │   └── forecaster.py        # Prophet time-series
│   ├── vision/
│   │   ├── street_fetcher.py    # OSM tile image downloader
│   │   └── risk_scorer.py       # EfficientNet-B0 risk scoring
│   ├── nlp/
│   │   ├── embeddings.py        # ChromaDB document indexing
│   │   ├── qa_chain.py          # LangChain RAG chain
│   │   └── report_generator.py  # AI safety report generation
│   └── api/
│       ├── main.py              # FastAPI app + static file serving
│       └── routes/
│           ├── heatmap.py       # Zone risk score endpoints
│           ├── forecasts.py     # Forecast image endpoints
│           ├── predictions.py   # ML prediction + SHAP endpoints
│           └── chat.py          # RAG Q&A endpoint
├── frontend/
│   ├── index.html               # Single-page app (1300+ lines)
│   └── static/                  # Static assets
├── data/
│   ├── raw/                     # Downloaded CSVs (gitignored)
│   ├── processed/               # Parquet files (gitignored)
│   └── external/                # Weather data (gitignored)
├── artifacts/
│   ├── models/                  # Trained model files (gitignored)
│   ├── forecasts/               # Prophet chart PNGs (gitignored)
│   ├── vision/                  # Zone images + scores (gitignored)
│   ├── chroma_db/               # Vector store (gitignored)
│   └── reports/                 # AI-generated reports (gitignored)
├── main.py                      # Full pipeline runner
├── pyproject.toml               # Dependencies (uv)
├── .env.example                 # Environment variable template
└── .python-version              # Python 3.11 pin
```

---

## API Endpoints

Once running, interactive docs available at `http://localhost:8000/docs`

### Heatmap
```
GET  /api/heatmap/zones              → All zone risk scores + coordinates
```

### Forecasts
```
GET  /api/forecasts/zones            → List zones with forecast data
GET  /api/forecasts/citywide/image   → Citywide Prophet forecast PNG
GET  /api/forecasts/zone/{id}/image  → Per-zone forecast PNG
```

### Predictions
```
POST /api/predictions/predict        → Predict crime type from features
POST /api/predictions/explain        → SHAP explanation for a prediction
```

**Prediction request body:**
```json
{
  "hour": 22,
  "day_of_week": 5,
  "month": 7,
  "season": 2,
  "is_weekend": true,
  "zone_id": 1434,
  "temp_max": 31.0,
  "precipitation": 0.0,
  "windspeed": 12.0
}
```

### Chat
```
POST /api/chat/ask                   → RAG-powered Q&A over crime data
```

```json
{ "query": "Which zone is most dangerous on Friday nights?" }
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone and install

```bash
git clone https://github.com/RodionRaskolnikov1/CrimeScope
cd crimescope

# Install all dependencies with uv (takes ~10 seconds)
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

```env
GOOGLE_GEMINI_API_KEY="your_key_here"
```

### 3. Run the full ML pipeline

This downloads data, trains models, and generates all artifacts. Run once — takes 15–20 minutes on first run.

```bash
uv run python main.py
```

Pipeline stages:
```
✅ Data ingestion    — downloads 700k Chicago crime records
✅ Preprocessing     — cleans, feature engineers, saves parquet
✅ Validation        — Pandera schema check (drops invalid rows)
✅ ML training       — XGBoost classifier + CV scoring
✅ SHAP analysis     — global feature importance chart
✅ Forecasting       — Prophet models for 10 zones + citywide
✅ Vision pipeline   — OSM tiles + EfficientNet risk scoring
✅ RAG indexing      — builds ChromaDB vector store
✅ Report generation — AI writes safety reports per zone
```

### 4. Start the API server

```bash
uv run uvicorn crimescope.api.main:app --reload --port 8000
```

Open `http://localhost:8000` — the full app loads instantly.

---

## Data Sources

| Source | Data | Size |
|---|---|---|
| [Chicago Open Data Portal](https://data.cityofchicago.org) | Crime records 2021–2023 | ~700k rows |
| [Open-Meteo](https://open-meteo.com) | Historical weather for Chicago | 1,461 days |
| [OpenStreetMap](https://openstreetmap.org) | Map tile images per zone | 10 tiles |

All data is free and publicly available. No API keys required for data collection (Gemini key only needed for LLM features).

---

## Model Performance

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 29.2% | 10-class classification (3× better than random) |
| F1 Weighted | 0.210 | Cross-validated across 5 folds |
| CV Stability | ±0.001 | Very consistent across folds |

The relatively low accuracy reflects the genuine difficulty of predicting which specific crime type will occur — location and time are strong signals but crime type has high inherent randomness. The SHAP analysis shows `zone_id` and `hour` are by far the most predictive features, which aligns with real criminology research.

---

## Key Design Decisions

**Why Polars instead of pandas?**
10–100x faster on large dataframes. 700k rows loads in milliseconds.

**Why Prophet instead of ARIMA?**
Prophet handles Chicago-specific seasonality (summer crime spikes, holiday effects, weekly patterns) automatically without manual parameter tuning.

**Why OSM tiles instead of Google Street View?**
Street View requires billing setup. OSM is completely free with no API key. The architecture is identical — swapping to Street View is a one-line URL change.

**Why vanilla JS instead of React?**
Zero build tooling, instant loading, no node_modules. The entire frontend is one HTML file served directly by FastAPI.

**Why uv instead of pip?**
Package resolution and installation is 10–100x faster. `uv sync` installs all dependencies in under 10 seconds vs 3+ minutes with pip.

---

## Roadmap

- [ ] Add NeuralForecast LSTM upgrade (Linux/cloud deployment)
- [ ] Google Street View integration when billing is set up
- [ ] Real-time data refresh via Chicago Open Data API webhooks
- [ ] Add community area demographics from US Census API
- [ ] Docker deployment configuration
- [ ] GitHub Actions CI pipeline with Ruff + Pytest

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Chicago Data Portal](https://data.cityofchicago.org) for the open crime dataset
- [Meta Prophet](https://facebook.github.io/prophet/) for the forecasting library
- [TIMM](https://github.com/huggingface/pytorch-image-models) for pretrained vision models
- [Open-Meteo](https://open-meteo.com) for free historical weather data