"""Sensitivity tests and replication (Methods, 'Sensitivity and replication')."""
import copy
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from .common import Grid, fill_empty, granule_date, nearest_pixel, pace_rrs
from .final import PRIMARY, rb
from .stats import coast_distance_km, load_grids

VARS = [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"]


def _pooled(m, label, vars_=VARS):
    rows = []
    for v in vars_:
        g = m[m["var"] == v]
        if len(g) < 200:
            continue
        q = rb(g.pace, g.olci, v)
        rows.append(dict(test=label, var=v, n=q["n"], r=q["r"], eps=q["eps"], beta=q["beta"]))
    return rows


def spectral_matching(cfg, long_tbl, dist_thr=2.0):
    """SRF (as gridded) vs nearest PACE band vs 10 nm boxcar, for the four main bands on all days."""
    grid = Grid(cfg)
    grids = load_grids(cfg)
    dist, _ = coast_distance_km(grids)
    far = dist > dist_thr
    res = {k: [] for k in ("nearest", "boxcar10")}
    for date, g in grids.items():
        files = sorted((cfg["paths"]["raw"] / "pace_aop").glob("*.nc"))
        files = [f for f in files if granule_date(f) == f"{date[:4]}-{date[4:6]}-{date[6:]}"]
        if not files:
            continue
        nns = []
        for f in files:
            nav = xr.open_dataset(f, group="navigation_data")
            nns.append(nearest_pixel(grid, nav["latitude"].values, nav["longitude"].values, cfg["qc"].get("pace_fill_km", 3.5)))
        for wl in PRIMARY:
            ol = g[f"olci_Rrs_{wl:g}"].values
            for mode, width in (("nearest", 0), ("boxcar10", 10)):
                S = np.zeros((grid.ny, grid.nx)); C = np.zeros((grid.ny, grid.nx)); C0 = np.zeros((grid.ny, grid.nx))
                for f, nn in zip(files, nns):
                    lat, lon, rrs, bad = pace_rrs(f, wl, "nearest" if mode == "nearest" else "boxcar", width)
                    rrs = np.where(bad | (rrs <= 0) | (rrs > cfg["qc"]["rrs_max"]), np.nan, rrs)
                    s, c = grid.bin(lat, lon, rrs)
                    C0 += c
                    s, c, _ = fill_empty(s, c, nn, rrs)
                    S += s; C += c
                with np.errstate(invalid="ignore", divide="ignore"):
                    pm = np.where(C > 0, S / C, np.nan)
                k = np.isfinite(pm) & np.isfinite(ol) & far & (pm > 0) & (ol > 0) & (C0 >= 2)
                res[mode].append(pd.DataFrame({"var": f"Rrs_{wl:g}", "pace": pm[k], "olci": ol[k]}))
    rows = []
    for mode, lst in res.items():
        m = pd.concat(lst)
        rows += _pooled(m, f"PACE band: {mode}", [f"Rrs_{w:g}" for w in PRIMARY])
    return rows


def _block_mean(a, k=3):
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    return b


def coarse_grid_rows(cfg, label="Coarser grid (0.06 deg blocks)", k=3, min_cells=5):
    """Average the 0.02 deg cells (both sensors) into k x k blocks and repeat the pooled comparison."""
    from .stats import load_grids
    grids = load_grids(cfg)
    dist, _ = coast_distance_km(grids)
    parts = []
    for date, g in grids.items():
        for v in VARS:
            p = g[f"pace_{v}"].values.astype(float); o = g[f"olci_{v}"].values.astype(float)
            ok = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0) & (g[f"pace_{v}_n"].values >= 2) & (dist > 2)
            p = np.where(ok, p, np.nan); o = np.where(ok, o, np.nan)
            bp, bo = _block_mean(p, k), _block_mean(o, k)
            n = np.isfinite(bp).sum(axis=(1, 3))
            with np.errstate(invalid="ignore"):
                mp = np.nanmean(bp, axis=(1, 3)); mo = np.nanmean(bo, axis=(1, 3))
            keep = n >= min_cells
            parts.append(pd.DataFrame({"var": v, "pace": mp[keep], "olci": mo[keep]}))
    return _pooled(pd.concat(parts), label)


def run_sensitivity(cfg, long_tbl, outdir):
    rows = _pooled(long_tbl, "Baseline (response-function average; 2 km; cells with >= 2 PACE pixels)")
    # coastal exclusion 5 km
    rows += _pooled(long_tbl[long_tbl.dist_km > 5], "Coastal exclusion 5 km")
    # gap fill: allow single-pixel and nearest-pixel-filled PACE cells
    from .final import long_table
    rows += _pooled(long_table(cfg, min_pace_pixels=1), "Single-pixel and gap-filled PACE cells included")
    # footprint: coarser common grid and the low-view-angle part of the swath
    rows += coarse_grid_rows(cfg)
    rows += _pooled(long_tbl[long_tbl.pace_vza < 25], "PACE view zenith below 25 deg")
    # spectral matching
    rows += spectral_matching(cfg, long_tbl)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sensitivity.csv", index=False)
    return df


def value_limit_variants(cfg, days, outdir):
    """Reprocess a subset of days with relaxed and tightened value limits; compare pooled results."""
    from .process import process_date
    base = load_grids(cfg)
    dist, _ = coast_distance_km(base)
    variants = {"as used": None,
                "no value limits": dict(chl_range=[1e-9, 1e9], rrs_max=1e9),
                "tight limits": dict(chl_range=[0.03, 30], rrs_max=0.015),
                "QC: Rrs mask without CHLFAIL and COCCOLITH": dict(rrs_flags_drop=["CHLFAIL", "COCCOLITH"])}
    rows = []
    for name, ov in variants.items():
        c = copy.deepcopy(cfg)
        c["wavelengths"] = PRIMARY
        if ov:
            c["qc"].update(ov)
            c["paths"]["processed"] = cfg["paths"]["processed"].parent / ("variant_" + __import__("re").sub(r"[^A-Za-z0-9]+", "_", name))
            c["paths"]["processed"].mkdir(exist_ok=True)
            for d in days:
                out = c["paths"]["processed"] / f"grid_{d.replace('-', '')}.nc"
                if not out.exists():
                    process_date(c, d)
            grids = {d.replace("-", ""): xr.open_dataset(c["paths"]["processed"] / f"grid_{d.replace('-', '')}.nc").load() for d in days}
        else:
            grids = {d.replace("-", ""): base[d.replace("-", "")] for d in days}
        parts = []
        for date, g in grids.items():
            for v in VARS:
                p, o = g[f"pace_{v}"].values, g[f"olci_{v}"].values
                k = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0) & (dist > 2) & (g[f"pace_{v}_n"].values >= 2)
                parts.append(pd.DataFrame({"var": v, "pace": p[k], "olci": o[k]}))
        rows += _pooled(pd.concat(parts), (f"{name} (10 days)" if name.startswith("QC") else f"Value limits: {name} (10 days)"))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sensitivity_value_limits.csv", index=False)
    return df


def replication(cfg, outdir):
    """Agreement on the days processed for the in situ comparison (not chosen by the clear-sky rule)."""
    main = load_grids(cfg)
    dist, _ = coast_distance_km(main)
    rows = []
    for f in sorted(glob.glob("data/processed_insitu/grid_*.nc")):
        g = xr.open_dataset(f).load(); d = os.path.basename(f)[5:13]
        for v in VARS:
            if f"pace_{v}" not in g or f"olci_{v}" not in g:
                continue
            p, o = g[f"pace_{v}"].values, g[f"olci_{v}"].values
            k = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0) & (dist > 2) & (g[f"pace_{v}_n"].values >= 2)
            if k.sum() >= 500:
                q = rb(p[k], o[k], v)
                rows.append(dict(date=d, var=v, **q))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "replication_days.csv", index=False)
    rng = np.random.default_rng(4)
    summ = []
    for v, g in df.groupby("var"):
        b = g.beta.values
        bs = [np.median(rng.choice(b, len(b))) for _ in range(2000)]
        summ.append(dict(var=v, days=len(b), median_beta=np.median(b), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         frac_olci_higher=float((b > 0).mean()), median_eps=g.eps.median(), median_r=g.r.median()))
    s = pd.DataFrame(summ)
    s.to_csv(outdir / "replication_summary.csv", index=False)
    return df, s
