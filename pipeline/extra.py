"""Extra analyses for the paper: robust metrics, coverage vs distance to coast,
coastal-cutoff variants, PACE band-width sensitivity."""
import copy

import numpy as np
import pandas as pd
from scipy import stats as sst

from .common import Grid, fill_empty, nearest_pixel, pace_rrs
from .stats import build_matchups, coast_distance_km, load_grids, variables


def robust(x, y):
    """x = PACE (reference), y = OLCI. Log-ratio metrics after Seegers et al. (2018):
    eps = median symmetric accuracy (%), beta = signed symmetric percent bias (%),
    MAPD = mean absolute percentage difference (%)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return dict(n=int(x.size))
    lr = np.log10(y / x)
    eps = 100 * (10 ** np.median(np.abs(lr)) - 1)
    beta = 100 * np.sign(np.median(lr)) * (10 ** np.abs(np.median(lr)) - 1)
    r = sst.pearsonr(x, y)[0]
    rl = sst.pearsonr(np.log10(x), np.log10(y))[0]
    return dict(n=int(x.size), r=r, r_log=rl, slope=sst.linregress(x, y).slope,
                RMSE=float(np.sqrt(np.mean((y - x) ** 2))), bias=float(np.mean(y - x)),
                MAPD_pct=float(np.mean(np.abs(y - x) / x) * 100), eps_pct=eps, beta_pct=beta,
                RMSE_log10=float(np.sqrt(np.mean(lr ** 2))))


def coverage_by_distance(cfg):
    G = load_grids(cfg)
    dist, _ = coast_distance_km(G)
    edges = [0, 2, 5, 10, 25, 50, 100, 1e9]
    rows = []
    for var in ("chlor_a", "Rrs_490"):
        tot = np.zeros(len(edges) - 1); miss = np.zeros(len(edges) - 1)
        for g in G.values():
            o = np.isfinite(g[f"olci_{var}"].values); p = np.isfinite(g[f"pace_{var}"].values)
            for i in range(len(edges) - 1):
                m = o & (dist > edges[i]) & (dist <= edges[i + 1])
                tot[i] += m.sum(); miss[i] += (m & ~p).sum()
        for i in range(len(edges) - 1):
            rows.append(dict(var=var, d_from_km=edges[i], d_to_km=edges[i + 1], n_olci_cells=int(tot[i]),
                             pct_pace_missing=100 * miss[i] / max(tot[i], 1)))
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["out"] / "coverage_by_distance.csv", index=False)
    return df


def variants(cfg):
    """Pooled robust metrics for coastal-exclusion variants (2 km, 5 km) and per-date summary."""
    out, per_date = [], []
    for excl in (2, 5):
        c = copy.deepcopy(cfg)
        c["qc"]["exclude_coast_km"] = excl
        m = build_matchups(c)
        for v in variables(cfg):
            g = m[m["var"] == v]
            out.append(dict(coast_excl_km=excl, var=v, n_dates=g.date.nunique(), **robust(g.pace, g.olci)))
            if excl == 5:
                rr = [dict(date=d, **robust(h.pace, h.olci)) for d, h in g.groupby("date")]
                dd = pd.DataFrame(rr)
                per_date.append(dict(var=v, n_dates=len(dd), r_median=dd.r.median(), r_min=dd.r.min(), r_max=dd.r.max(),
                                     beta_median=dd.beta_pct.median(), beta_iqr_lo=dd.beta_pct.quantile(.25),
                                     beta_iqr_hi=dd.beta_pct.quantile(.75), eps_median=dd.eps_pct.median(),
                                     frac_dates_olci_higher=float((dd.beta_pct > 0).mean())))
        if excl == 5:
            for z, h in m.groupby("zone", observed=True):
                pass
    pooled = pd.DataFrame(out)
    pooled.to_csv(cfg["paths"]["out"] / "stats_variants.csv", index=False)
    pd.DataFrame(per_date).to_csv(cfg["paths"]["out"] / "stats_per_date_summary.csv", index=False)
    return pooled, pd.DataFrame(per_date)


def band_width_sensitivity(cfg, min_coast_km=5):
    """How does averaging PACE bands (nearest, +/-2.5, +/-5, +/-10 nm) change agreement with OLCI Rrs?"""
    grid = Grid(cfg)
    G = load_grids(cfg)
    dist, _ = coast_distance_km(G)
    far = dist >= min_coast_km
    modes = [("nearest", 0), ("boxcar", 5), ("boxcar", 10), ("boxcar", 20)]
    rows = []
    for date, g in G.items():
        d8 = date
        files = sorted(cfg["paths"]["raw"].glob(f"pace_aop/*{d8}T*.nc"))
        if not files:
            continue
        nns = []
        import xarray as xr
        for f in files:
            nav = xr.open_dataset(f, group="navigation_data")
            nns.append(nearest_pixel(grid, nav["latitude"].values, nav["longitude"].values, cfg["qc"].get("pace_fill_km", 3.5)))
        for wl in cfg["wavelengths"]:
            ol = g[f"olci_Rrs_{wl:g}"].values
            for mode, width in modes:
                S = np.zeros((grid.ny, grid.nx)); C = np.zeros((grid.ny, grid.nx))
                for f, nn in zip(files, nns):
                    lat, lon, rrs, bad = pace_rrs(f, wl, mode, width)
                    rrs = np.where(bad | (rrs <= 0) | (rrs > cfg["qc"]["rrs_max"]), np.nan, rrs)
                    s, c = grid.bin(lat, lon, rrs)
                    s, c, _ = fill_empty(s, c, nn, rrs)
                    S += s; C += c
                with np.errstate(invalid="ignore", divide="ignore"):
                    pm = np.where(C > 0, S / C, np.nan)
                m = np.isfinite(pm) & np.isfinite(ol) & far
                rows.append(dict(date=date, wl=wl, mode=f"{mode}{width}", n=int(m.sum()),
                                 pace=pm[m], olci=ol[m]))
    df = pd.DataFrame(rows)
    res = []
    for (wl, mode), h in df.groupby(["wl", "mode"]):
        x = np.concatenate(h.pace.values); y = np.concatenate(h.olci.values)
        res.append(dict(wl=wl, mode=mode, **robust(x, y)))
    res = pd.DataFrame(res)
    res.to_csv(cfg["paths"]["out"] / "sensitivity_band_width.csv", index=False)
    return res


def run_extra(cfg):
    pd.set_option("display.width", 220)
    print(coverage_by_distance(cfg).round(2).to_string(index=False))
    p, d = variants(cfg)
    print(p.round(4).to_string(index=False))
    print(d.round(3).to_string(index=False))
    print(band_width_sensitivity(cfg).round(4).to_string(index=False))
