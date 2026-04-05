from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from crimescope.api.routes import chat, forecasts, predictions, heatmap

app = FastAPI(title="CrimeScope API", version="1.0.0")

app.include_router(heatmap.router,     prefix="/api/heatmap",     tags=["heatmap"])
app.include_router(forecasts.router,   prefix="/api/forecasts",   tags=["forecasts"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(chat.router,        prefix="/api/chat",        tags=["chat"])

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
STATIC_DIR   = FRONTEND_DIR / "static"

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"error": "index.html not found in frontend/ directory"}
    return FileResponse(str(index))

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Don't intercept API routes (shouldn't happen but safety net)
    if full_path.startswith("api/"):
        return {"error": "not found"}
    file = FRONTEND_DIR / full_path
    if file.exists() and file.is_file():
        return FileResponse(str(file))
    # Fallback to index.html for SPA routing
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "Frontend not found. Place index.html in frontend/ directory."}