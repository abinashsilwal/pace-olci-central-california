"""Distance from a fixed, documented coastline (Natural Earth 1:10m land polygons), independent of sensor land flags.

Distances are Euclidean in an equidistant azimuthal projection centred on the region (error well under 1 % over a few degrees).
The coastline is densified to 200 m vertex spacing and distances are taken to the nearest vertex (error < 0.1 km).
Points inside a land polygon get distance 0.
"""
from pathlib import Path

import numpy as np
import shapely
from cartopy.io import shapereader
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import unary_union

_STATE = {}
CACHE = Path("data/cache"); CACHE.mkdir(parents=True, exist_ok=True)


def _prepare(lat, lon, pad=1.5):
    """Land polygons and coastline vertices for the region covering the given points (cached by rounded bounds)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    b = (np.floor(np.nanmin(lon) - pad), np.floor(np.nanmin(lat) - pad), np.ceil(np.nanmax(lon) + pad), np.ceil(np.nanmax(lat) + pad))
    if b in _STATE:
        return _STATE[b]
    shp = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    clip = box(*b)
    parts = [g.intersection(clip) for g in shapereader.Reader(shp).geometries() if g.intersects(clip)]
    land = unary_union(parts) if parts else None
    lon0, lat0 = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    tr = Transformer.from_crs("EPSG:4326", CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"), always_xy=True)
    if land is None:
        _STATE[b] = dict(tr=tr, land=None, tree=None)
        return _STATE[b]
    land_p = shapely.transform(land, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    coast = shapely.segmentize(shapely.boundary(land_p), 200.0)
    xy = shapely.get_coordinates(coast)
    clip_p = shapely.transform(clip, lambda a: np.column_stack(tr.transform(a[:, 0], a[:, 1])))
    keep = shapely.distance(shapely.points(xy[:, 0], xy[:, 1]), shapely.boundary(clip_p)) > 500.0
    shapely.prepare(land_p)
    _STATE[b] = dict(tr=tr, land=land_p, tree=cKDTree(xy[keep]))
    return _STATE[b]


def distance_km(lat, lon):
    """Distance (km) from each (lat, lon) to the coastline; 0 for points on land. Arrays of any shape."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    shape = np.broadcast(lat, lon).shape
    la, lo = np.broadcast_to(lat, shape).ravel(), np.broadcast_to(lon, shape).ravel()
    st = _prepare(la, lo)
    if st["tree"] is None:
        return np.full(shape, 999.0)
    x, y = st["tr"].transform(lo, la)
    d, _ = st["tree"].query(np.column_stack([x, y]))
    d = np.where(shapely.contains_xy(st["land"], x, y), 0.0, d) / 1000.0
    return d.reshape(shape)


def grid_distance_km(lat1d, lon1d, name="grid"):
    """Cached distance field on a regular lat/lon grid (ny, nx)."""
    key = CACHE / f"coast_dist_{name}_{lat1d.size}x{lon1d.size}_{lat1d[0]:.4f}_{lon1d[0]:.4f}.npy"
    if key.exists():
        return np.load(key)
    la, lo = np.meshgrid(lat1d, lon1d, indexing="ij")
    d = distance_km(la, lo)
    np.save(key, d)
    return d
