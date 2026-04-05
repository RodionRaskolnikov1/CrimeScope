"""
Geospatial utility functions for CrimeScope.
Handles coordinate conversions between lat/lon and grid zones.
"""
import math


# ── Chicago bounding box ──────────────────────────────────────────
LAT_MIN, LAT_MAX = 41.6, 42.1
LON_MIN, LON_MAX = -87.9, -87.5
GRID_SIZE = 50


def lat_lon_to_zone_id(
    lat: float,
    lon: float,
    grid_size: int = GRID_SIZE,
) -> int:
    """
    Convert lat/lon coordinates to a grid zone ID.

    Divides Chicago into a grid_size x grid_size grid.
    Each cell gets a unique integer zone_id.

    Args:
        lat: Latitude (must be within Chicago bounds)
        lon: Longitude (must be within Chicago bounds)
        grid_size: Number of cells per axis (default 50 → 2500 zones)

    Returns:
        Integer zone_id in range [0, grid_size^2 - 1]
    """
    row = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * grid_size)
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * grid_size)

    row = max(0, min(grid_size - 1, row))
    col = max(0, min(grid_size - 1, col))

    return row * grid_size + col


def zone_id_to_lat_lon(
    zone_id: int,
    grid_size: int = GRID_SIZE,
) -> tuple[float, float]:
    """
    Convert a grid zone ID back to the center lat/lon of that cell.

    This is the reverse of lat_lon_to_zone_id — used when we need
    to plot zone markers on the map from stored zone IDs.

    Args:
        zone_id: Integer zone ID in range [0, grid_size^2 - 1]
        grid_size: Number of cells per axis (must match what was used to create zone_id)

    Returns:
        Tuple of (latitude, longitude) at the center of the zone cell
    """
    row = zone_id // grid_size
    col = zone_id % grid_size

    lat = LAT_MIN + (row + 0.5) * (LAT_MAX - LAT_MIN) / grid_size
    lon = LON_MIN + (col + 0.5) * (LON_MAX - LON_MIN) / grid_size

    return round(lat, 5), round(lon, 5)


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate great-circle distance between two lat/lon points in kilometers.

    Uses the Haversine formula — accurate for short distances
    (error < 0.3% for distances under 500km).

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_within_chicago(lat: float, lon: float) -> bool:
    """
    Check if coordinates are within Chicago's bounding box.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        True if within Chicago bounds, False otherwise
    """
    return (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)


def get_zone_bounds(
    zone_id: int,
    grid_size: int = GRID_SIZE,
) -> dict:
    """
    Get the bounding box of a zone cell.

    Args:
        zone_id: Integer zone ID
        grid_size: Grid size used when creating zones

    Returns:
        Dict with lat_min, lat_max, lon_min, lon_max
    """
    row = zone_id // grid_size
    col = zone_id % grid_size

    lat_step = (LAT_MAX - LAT_MIN) / grid_size
    lon_step = (LON_MAX - LON_MIN) / grid_size

    return {
        "lat_min": round(LAT_MIN + row * lat_step, 5),
        "lat_max": round(LAT_MIN + (row + 1) * lat_step, 5),
        "lon_min": round(LON_MIN + col * lon_step, 5),
        "lon_max": round(LON_MIN + (col + 1) * lon_step, 5),
    }