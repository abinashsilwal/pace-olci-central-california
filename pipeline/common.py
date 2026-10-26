"""Shared helpers: config, grid, L2 reading, flag masks, binning."""
import re
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Fallback l2gen flag bit positions (0-indexed), used only if the file lacks flag_masks.
_OC_BAD = ["ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "STRAYLIGHT", "CLDICE",
           "COCCOLITH", "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "MAXAERITER",
           "ATMWARN", "NAVFAIL", "FILTER"]
_FALLBACK_BITS = {"ATMFAIL": 0, "LAND": 1, "HIGLINT": 3, "HILT": 4, "HISATZEN": 5,
                  "STRAYLIGHT": 8, "CLDICE": 9, "COCCOLITH": 10, "HISOLZEN": 12,
                  "LOWLW": 14, "CHLFAIL": 15, "NAVWARN": 16, "MAXAERITER": 19,
                  "ATMWARN": 22, "NAVFAIL": 25, "FILTER": 26}


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


class Grid:
    def __init__(self, cfg):
        a, self.res = cfg["study_area"], cfg["grid"]["res_deg"]
        self.lon0, self.lat0 = a["lon_min"], a["lat_min"]
        self.nx = int(round((a["lon_max"] - a["lon_min"]) / self.res))
        self.ny = int(round((a["lat_max"] - a["lat_min"]) / self.res))
        self.lon = self.lon0 + (np.arange(self.nx) + 0.5) * self.res
        self.lat = self.lat0 + (np.arange(self.ny) + 0.5) * self.res

    def index(self, lat, lon):
        ix = np.floor((lon - self.lon0) / self.res).astype(int)
        iy = np.floor((lat - self.lat0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy * self.nx + ix, ok

    def bin(self, lat, lon, val):
        """Return (sum, count) arrays shaped (ny, nx); NaNs in val are skipped."""
        flat, ok = self.index(lat, lon)
        ok &= np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
        n = self.ny * self.nx
        s = np.bincount(flat[ok], weights=val[ok], minlength=n).reshape(self.ny, self.nx)
        c = np.bincount(flat[ok], minlength=n).reshape(self.ny, self.nx)
        return s.astype(float), c.astype(float)


def nearest_pixel(grid, lat, lon, dmax_km):
    """For every grid-cell centre: index (flattened over `lat`) of the nearest geolocated pixel of
    the granule and whether it lies within dmax_km. PACE pixels grow to ~5 km at the swath edge, so
    cells can fall between pixels; the nearest pixel is used to fill those cells (see fill_empty)."""
    from scipy.spatial import cKDTree
    a = 35.5
    kx, ky = 111.32 * np.cos(np.deg2rad(a)), 110.57
    ok = np.isfinite(lat) & np.isfinite(lon)
    pad = 0.3
    ok &= (lat > grid.lat[0] - pad) & (lat < grid.lat[-1] + pad) & (lon > grid.lon[0] - pad) & (lon < grid.lon[-1] + pad)
    idx_all = np.flatnonzero(ok.ravel())
    pts = np.c_[lon.ravel()[idx_all] * kx, lat.ravel()[idx_all] * ky]
    LON, LAT = np.meshgrid(grid.lon, grid.lat)
    d, j = cKDTree(pts).query(np.c_[LON.ravel() * kx, LAT.ravel() * ky])
    return idx_all[j].reshape(grid.ny, grid.nx), (d <= dmax_km).reshape(grid.ny, grid.nx)


def fill_empty(s, c, nn, val):
    """Cells with no valid pixel take the nearest pixel's value if that pixel is itself valid and
    within reach; cells whose nearest pixel is flagged stay empty (real gaps are preserved)."""
    idx, near = nn
    v = val.ravel()[idx]
    fill = (c == 0) & near & np.isfinite(v)
    s = s.copy(); c = c.copy()
    s[fill] = v[fill]; c[fill] = 1
    return s, c, int(fill.sum())


def granule_date(path):
    m = re.search(r"\.(\d{8})T", Path(path).name)
    return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None


def open_l2(path):
    nav = xr.open_dataset(path, group="navigation_data")
    geo = xr.open_dataset(path, group="geophysical_data")
    return nav, geo


def flag_mask(geo, names=_OC_BAD):
    """Boolean array True where pixel carries any of the listed l2 flags."""
    fl = geo["l2_flags"]
    flags = fl.values.astype("int64")
    masks = fl.attrs.get("flag_masks")
    meanings = fl.attrs.get("flag_meanings")
    bits = 0
    if masks is not None and meanings is not None:
        for m, name in zip(np.atleast_1d(masks), str(meanings).split()):
            if name in names:
                bits |= int(m)
    else:
        for name in names:
            bits |= 1 << _FALLBACK_BITS[name]
    return (flags & bits) != 0


def land_mask(geo):
    return flag_mask(geo, ["LAND"])


def olci_var(geo, wl):
    """Find the OLCI Rrs variable whose name wavelength is nearest `wl` (e.g. Rrs_443)."""
    cands = {int(re.search(r"(\d+)", v).group(1)): v for v in geo.data_vars if re.fullmatch(r"Rrs_\d+", v)}
    if not cands:
        raise KeyError("no Rrs_### variables in OLCI file")
    best = min(cands, key=lambda k: abs(k - wl))
    return cands[best]


def pace_rrs(path, wl, mode, boxcar_nm):
    """Return (lat, lon, rrs, badmask) for PACE AOP granule at wavelength wl."""
    nav, geo = open_l2(path)
    wls = xr.open_dataset(path, group="sensor_band_parameters")["wavelength_3d"].values  # 172 Rrs bands ("wavelength" has 286 -> misaligned)
    if mode == "boxcar":
        sel = np.where(np.abs(wls - wl) <= boxcar_nm / 2)[0]
        if sel.size == 0:
            sel = [int(np.argmin(np.abs(wls - wl)))]
    else:
        sel = [int(np.argmin(np.abs(wls - wl)))]
    assert geo["Rrs"].shape[-1] == len(wls), "Rrs band count != wavelength_3d length"
    rrs = geo["Rrs"].isel(wavelength=list(sel)).mean("wavelength").values
    return nav["latitude"].values, nav["longitude"].values, rrs, flag_mask(geo)
