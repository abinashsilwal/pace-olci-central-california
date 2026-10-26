"""Step 4: matchup table and statistics (pooled, per date, per zone, date-bootstrap CI)."""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats as sst


def load_grids(cfg):
    files = sorted(cfg["paths"]["processed"].glob("grid_*.nc"))
    if not files:
        raise SystemExit("no processed grids: run `process` first")
    return {f.stem[5:]: xr.open_dataset(f).load() for f in files}


def coast_distance_km(grids):
    """Distance (km) of every grid cell from the Natural Earth coastline; also returns the accumulated sensor land flag."""
    from .coast import grid_distance_km
    ds0 = next(iter(grids.values()))
    land = np.zeros(ds0.land.shape, bool)
    for g in grids.values():
        land |= g.land.values.astype(bool)
    return grid_distance_km(ds0.lat.values, ds0.lon.values), land


def variables(cfg):
    return [f"Rrs_{w:g}" for w in cfg["wavelengths"]] + ["chlor_a"]


def build_matchups(cfg, grids=None):
    grids = grids or load_grids(cfg)
    dist, land = coast_distance_km(grids)
    z1, z2 = cfg["qc"]["zones_km"]
    rows = []
    for date, g in grids.items():
        la, lo = np.meshgrid(g.lat.values, g.lon.values, indexing="ij")
        base = {"date": date, "lat": la.ravel(), "lon": lo.ravel(), "dist_km": dist.ravel()}
        for v in variables(cfg):
            df = pd.DataFrame({**base, "var": v, "pace": g[f"pace_{v}"].values.ravel(),
                               "olci": g[f"olci_{v}"].values.ravel()})
            rows.append(df)
    m = pd.concat(rows, ignore_index=True).dropna(subset=["pace", "olci"])
    m = m[(m.dist_km >= cfg["qc"]["exclude_coast_km"]) & ~np.isnan(m.dist_km)]
    m["zone"] = pd.cut(m.dist_km, [0, z1, z2, np.inf], labels=["nearshore", "shelf", "offshore"])
    # keep only dates with enough co-valid cells (use chlor_a as the reference variable)
    cnt = m[m["var"] == "chlor_a"].groupby("date").size()
    good = cnt[cnt >= cfg["qc"]["min_common_cells"]].index
    return m[m.date.isin(good)].reset_index(drop=True)


def metrics(x, y, log=False):
    """x = PACE, y = OLCI. bias = OLCI - PACE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if log:
        x, y = np.log10(x), np.log10(y)
    if x.size < 3:
        return dict(n=int(x.size))
    r = sst.pearsonr(x, y)[0]
    sl = sst.linregress(x, y)
    return dict(n=int(x.size), r=r, R2=r ** 2, RMSE=float(np.sqrt(np.mean((y - x) ** 2))),
                bias=float(np.mean(y - x)), MAE=float(np.mean(np.abs(y - x))),
                slope=sl.slope, intercept=sl.intercept,
                MAPE_pct=float(np.median(np.abs(y - x) / np.abs(x)) * 100) if not log else np.nan)


def _table(m, cfg, by):
    rows = []
    for keys, g in m.groupby(by, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        v = dict(zip(by, keys))
        log = cfg["stats"]["chl_log10"] and v["var"] == "chlor_a"
        rows.append({**v, **metrics(g.pace, g.olci, log)})
    return pd.DataFrame(rows)


def bootstrap_dates(m, cfg, var, n=None):
    """CI on pooled r and RMSE by resampling dates (pixels within a date are not independent)."""
    n = n or cfg["stats"]["n_bootstrap"]
    sub = m[m["var"] == var]
    log = cfg["stats"]["chl_log10"] and var == "chlor_a"
    groups = {d: g for d, g in sub.groupby("date")}
    dates = list(groups)
    if len(dates) < 3:
        return {}
    rng = np.random.default_rng(42)
    out = []
    for _ in range(n):
        pick = rng.choice(dates, len(dates))
        g = pd.concat([groups[d] for d in pick])
        mm = metrics(g.pace, g.olci, log)
        out.append((mm["r"], mm["RMSE"], mm["bias"]))
    a = np.array(out)
    lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=0)
    return {"r_ci": (lo[0], hi[0]), "RMSE_ci": (lo[1], hi[1]), "bias_ci": (lo[2], hi[2])}


def run_stats(cfg):
    m = build_matchups(cfg)
    outdir = cfg["paths"]["out"]
    m.to_csv(outdir / "matchups.csv.gz", index=False)
    pooled = _table(m, cfg, ["var"])
    ci = {v: bootstrap_dates(m, cfg, v) for v in variables(cfg)}
    for k in ("r_ci", "RMSE_ci", "bias_ci"):
        pooled[k] = pooled["var"].map(lambda v: ci.get(v, {}).get(k))
    pooled.to_csv(outdir / "stats_pooled.csv", index=False)
    _table(m, cfg, ["var", "date"]).to_csv(outdir / "stats_per_date.csv", index=False)
    _table(m, cfg, ["var", "zone"]).to_csv(outdir / "stats_per_zone.csv", index=False)
    print(pooled.round(4).to_string(index=False))
    print(f"\n{m.date.nunique()} dates used, {len(m)} matched cell-variable pairs -> {outdir}")
    return m, pooled
