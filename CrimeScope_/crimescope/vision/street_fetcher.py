import httpx
import time
from pathlib import Path
from crimescope.config import settings
from crimescope.utils.logger import logger

# ── Constants ─────────────────────────────────────────────────────

IMAGE_DIR = settings.artifacts_dir / "vision" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# OpenStreetMap tile server — free, no key needed
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ZOOM = 17  # street level zoom

    
# ── Coordinate → Tile Math ────────────────────────────────────────

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """
    Convert lat/lon to OSM tile x/y coordinates.
    This is standard OSM tile math used everywhere.
    """
    import math
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) +
             1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def zone_id_to_coords(zone_id: int, grid_size: int = 50) -> tuple[float, float]:
    """
    Convert zone_id back to approximate lat/lon center.
    Reverse of the grid assignment in preprocessing.py
    """
    lat_min, lat_max = 41.6, 42.1
    lon_min, lon_max = -87.9, -87.5

    row = zone_id // grid_size
    col = zone_id % grid_size

    # Center of the grid cell
    lat = lat_min + (row + 0.5) * (lat_max - lat_min) / grid_size
    lon = lon_min + (col + 0.5) * (lon_max - lon_min) / grid_size

    return lat, lon


# ── Image Fetcher ─────────────────────────────────────────────────

def fetch_zone_image(
    zone_id: int,
    force: bool = False,
) -> Path | None:
    """
    Fetch OSM map tile for a zone.
    Returns path to saved image or None if failed.
    """

    out_path = IMAGE_DIR / f"zone_{zone_id}.png"

    if out_path.exists() and not force:
        logger.debug(f"Zone {zone_id} image already exists, skipping.")
        return out_path

    lat, lon = zone_id_to_coords(zone_id)
    x, y = lat_lon_to_tile(lat, lon, ZOOM)
    url = OSM_TILE_URL.format(z=ZOOM, x=x, y=y)

    try:
        headers = {"User-Agent": "CrimeScope/1.0 (research project)"}
        response = httpx.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(response.content)

        # OSM rate limit — be polite
        time.sleep(0.5)

        logger.debug(f"Zone {zone_id} → image saved ({lat:.4f}, {lon:.4f})")
        return out_path

    except Exception as e:
        logger.warning(f"Zone {zone_id} image fetch failed: {e}")
        return None


def fetch_all_zones(zone_ids: list[int]) -> dict[int, Path]:
    """
    Fetch OSM images for all zones.
    Returns dict of {zone_id: image_path}
    """

    logger.info(f"Fetching map images for {len(zone_ids)} zones...")
    results = {}

    for i, zone_id in enumerate(zone_ids):
        path = fetch_zone_image(zone_id)
        if path:
            results[zone_id] = path
        logger.info(f"Progress: {i+1}/{len(zone_ids)}")

    logger.success(f"Fetched {len(results)} zone images → {IMAGE_DIR}")
    return results