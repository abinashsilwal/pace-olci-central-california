"""Step 3: quality control, spectral matching, gridding and co-location. One NetCDF per date on the common grid.

Method summary (this is what the paper describes):
  * l2gen flags and value-range limits are applied per pixel for both sensors.
  * PACE Rrs is convolved with the OLCI band-mean spectral response (band-weighted mean of the 172 PACE bands).
  * Both sensors are box-averaged onto a common regular grid; empty PACE cells (wide swath-edge pixels)
    take the nearest PACE pixel if that pixel is valid and within `pace_fill_km`.
  * PACE view zenith angle and solar zenith angle (both sensors) are gridded for diagnostics.
"""
import numpy as np
import xarray as xr

from .common import (_OC_BAD, Grid, fill_empty, flag_mask, granule_date, land_mask, nearest_pixel, olci_var,
                     open_l2)
from .geometry import BAND_OF, granule_time, load_srf, pace_vza, solar_zenith


def _files(cfg, key, date):
    d = cfg["paths"]["raw"] / key
    return sorted(p for p in d.glob("*.nc") if granule_date(p) == date)


def _acc(grid, store, name, lat, lon, val, nn=None):
    s, c = grid.bin(lat, lon, val)
    if nn is not None:
        s, c, _ = fill_empty(s, c, nn, val)
    if name in store:
        store[name][0] += s
        store[name][1] += c
    else:
        store[name] = [s, c]


def pace_rrs_bands(path, wls_nominal, srf, flag_names=None):
    """Return lat, lon, {nominal_wl: Rrs array (lines x pixels)}, flag mask, for one PACE AOP granule."""
    nav = xr.open_dataset(path, group="navigation_data")
    geo = xr.open_dataset(path, group="geophysical_data")
    w = xr.open_dataset(path, group="sensor_band_parameters")["wavelength_3d"].values
    assert geo["Rrs"].shape[-1] == len(w), "Rrs band count != wavelength_3d length"
    dw = np.gradient(w)
    weights, need = {}, set()
    for wl in wls_nominal:
        sw, ss = srf[BAND_OF[wl]]
        s = np.interp(w, sw, ss, left=0, right=0)
        keep = np.where(s > 1e-3 * s.max())[0]
        weights[wl] = (keep, s[keep] * dw[keep])
        need.update(keep.tolist())
    need = sorted(need)
    pos = {k: i for i, k in enumerate(need)}
    R = geo["Rrs"].isel(wavelength=need).values.astype("float32")
    out = {}
    for wl, (keep, wt) in weights.items():
        sub = R[:, :, [pos[k] for k in keep]]
        ok = np.isfinite(sub)
        wsum = (ok * wt).sum(-1)
        val = np.where(ok, sub, 0) @ wt
        with np.errstate(invalid="ignore", divide="ignore"):
            v = val / wsum
        v[wsum < 0.7 * wt.sum()] = np.nan                    # need >= 70% of the band weight observed
        out[wl] = v
    return nav["latitude"].values, nav["longitude"].values, out, (flag_mask(geo) if flag_names is None else flag_mask(geo, flag_names))


def process_date(cfg, date):
    grid = Grid(cfg)
    wls = cfg["wavelengths"]
    qc = cfg["qc"]
    lo_c, hi_c = qc["chl_range"]
    srf = load_srf()
    store, land = {}, [np.zeros((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx))]
    dmax = qc.get("pace_fill_km", 3.5)
    drop = qc.get("rrs_flags_drop", [])
    rnames = [f for f in _OC_BAD if f not in drop] if drop else None   # reflectance mask without the listed flags (chlorophyll keeps all)

    for p in _files(cfg, "pace_aop", date):
        lat, lon, bands, bad = pace_rrs_bands(p, wls, srf, rnames)
        nn = nearest_pixel(grid, lat, lon, dmax)
        for wl, rrs in bands.items():
            rrs = np.where(bad | (rrs <= 0) | (rrs > qc["rrs_max"]), np.nan, rrs)
            _acc(grid, store, f"pace_Rrs_{wl:g}", lat, lon, rrs, nn)
        _acc(grid, store, "pace_vza", lat, lon, pace_vza(lat, lon), nn)
        _acc(grid, store, "pace_sza", lat, lon, solar_zenith(lat, lon, granule_time(p, 2.5)), nn)
    for p in _files(cfg, "pace_bgc", date):
        nav, geo = open_l2(p)
        cv = geo["chlor_a"].values
        chl = np.where(flag_mask(geo) | (cv < lo_c) | (cv > hi_c), np.nan, cv)
        la, lo = nav["latitude"].values, nav["longitude"].values
        _acc(grid, store, "pace_chlor_a", la, lo, chl, nearest_pixel(grid, la, lo, dmax))
        s, c = grid.bin(la, lo, land_mask(geo).astype(float))
        land[0] += s; land[1] += c
    for key in ("olci_3a", "olci_3b"):
        for p in _files(cfg, key, date):
            nav, geo = open_l2(p)
            bad = flag_mask(geo)
            bad_r = flag_mask(geo, rnames) if rnames else bad
            la, lo = nav["latitude"].values, nav["longitude"].values
            for wl in wls:
                v = geo[olci_var(geo, wl)].values
                _acc(grid, store, f"olci_Rrs_{wl:g}", la, lo, np.where(bad_r | (v <= 0) | (v > qc["rrs_max"]), np.nan, v))
            cv = geo["chlor_a"].values
            _acc(grid, store, "olci_chlor_a", la, lo, np.where(bad | (cv < lo_c) | (cv > hi_c), np.nan, cv))
            sub = (slice(None, None, 8), slice(None, None, 8))
            _acc(grid, store, "olci_sza", la[sub], lo[sub], solar_zenith(la[sub], lo[sub], granule_time(p, 1.5)))
            s, c = grid.bin(la, lo, land_mask(geo).astype(float))
            land[0] += s; land[1] += c

    if not store:
        print(date, "no data found, skipped")
        return None
    min_n = qc["min_olci_pixels_per_cell"]
    ds = xr.Dataset(coords={"lat": grid.lat, "lon": grid.lon})
    for name, (s, c) in store.items():
        thr = min_n if (name.startswith("olci") and "sza" not in name) else 1
        with np.errstate(invalid="ignore", divide="ignore"):
            ds[name] = (("lat", "lon"), np.where(c >= thr, s / c, np.nan).astype("float32"))
        ds[name + "_n"] = (("lat", "lon"), c.astype("int32"))
    with np.errstate(invalid="ignore", divide="ignore"):
        ds["land"] = (("lat", "lon"), np.where(land[1] > 0, land[0] / land[1], np.nan) > 0.5)
    out = cfg["paths"]["processed"] / f"grid_{date.replace('-', '')}.nc"
    ds.to_netcdf(out)
    print(date, "->", out.name, flush=True)
    return out


def process_all(cfg):
    for d in cfg["dates"]:
        process_date(cfg, str(d))
