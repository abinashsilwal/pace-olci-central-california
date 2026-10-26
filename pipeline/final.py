"""Final analysis: every table, figure and number used in the paper is produced here (see docs/analysis-protocol.md).

Convention: x = PACE, y = OLCI; beta > 0 means OLCI is higher. Bootstrap = resampling of days.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst

from .stats import coast_distance_km, load_grids

PRIMARY = [442.5, 490, 560, 665]
ALL_BANDS = [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75]
LABEL = {"chlor_a": "Chl-a"}
for _w in ALL_BANDS:
    LABEL[f"Rrs_{_w:g}"] = f"$R_{{\mathrm{{rs}}}}$({_w:g})"
BOOT, SUB = 1000, 4000
C_PACE, C_OLCI = "#1f77b4", "#d95f02"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.constrained_layout.w_pad": 0.12, "figure.constrained_layout.h_pad": 0.12, "figure.constrained_layout.wspace": 0.1, "figure.constrained_layout.hspace": 0.1, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
from .figstyle import apply as _apply_figstyle  # noqa: E402
_apply_figstyle()   # Source Sans 3 in every figure


# ----------------------------------------------------------------------------- data
def long_table(cfg, bands=ALL_BANDS, min_pace_pixels=2):
    """One row per (day, cell, variable) where both sensors are valid, the cell is > 2 km from the coast and
    contains at least `min_pace_pixels` PACE pixels (2 excludes gap-filled and single-pixel cells)."""
    grids = load_grids(cfg)
    dist, _ = coast_distance_km(grids)
    z1, z2 = cfg["qc"]["zones_km"]
    rows = []
    for date, g in grids.items():
        la, lo = np.meshgrid(g.lat.values, g.lon.values, indexing="ij")
        chl_p, chl_o = g["pace_chlor_a"].values, g["olci_chlor_a"].values
        geom = {"pace_vza": g["pace_vza"].values if "pace_vza" in g else np.full(la.shape, np.nan),
                "pace_sza": g["pace_sza"].values if "pace_sza" in g else np.full(la.shape, np.nan),
                "olci_sza": g["olci_sza"].values if "olci_sza" in g else np.full(la.shape, np.nan)}
        chl_geo = np.sqrt(chl_p * chl_o)
        for v in [f"Rrs_{w:g}" for w in bands] + ["chlor_a"]:
            if f"pace_{v}" not in g or f"olci_{v}" not in g:
                continue
            p, o = g[f"pace_{v}"].values, g[f"olci_{v}"].values
            npx = g[f"pace_{v}_n"].values
            m = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0) & (dist > cfg["qc"]["exclude_coast_km"]) & (npx >= min_pace_pixels)
            df = pd.DataFrame({"date": date, "var": v, "lat": la[m], "lon": lo[m], "dist_km": dist[m], "pace": p[m], "olci": o[m],
                               "chl": chl_geo[m], "n_pace": g[f"pace_{v}_n"].values[m],
                               **{k: a[m] for k, a in geom.items()}})
            rows.append(df)
    m = pd.concat(rows, ignore_index=True)
    m["zone"] = pd.cut(m.dist_km, [0, z1, z2, np.inf], labels=[f"2-{z1} km", f"{z1}-{z2} km", f">{z2} km"])
    m["chl_class"] = pd.cut(m.chl, [0, 0.2, 0.5, 1, 3, np.inf], labels=["<0.2", "0.2-0.5", "0.5-1", "1-3", ">3"])
    return m


# ----------------------------------------------------------------------------- metrics
def rb(x, y, var=None):
    """r (log10 for chlorophyll), epsilon (%), beta (%), n. x = PACE, y = OLCI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return dict(n=int(x.size), r=np.nan, eps=np.nan, beta=np.nan)
    lr = np.log10(y / x)
    eps = 100 * (10 ** np.median(np.abs(lr)) - 1)
    beta = 100 * np.sign(np.median(lr)) * (10 ** abs(np.median(lr)) - 1)
    r = np.corrcoef(np.log10(x), np.log10(y))[0, 1] if var == "chlor_a" else np.corrcoef(x, y)[0, 1]
    return dict(n=int(x.size), r=float(r), eps=float(eps), beta=float(beta))


def boot(df, var, n=BOOT, seed=1):
    """95% intervals for r, eps, beta by resampling days (all cells of each resampled day)."""
    rng = np.random.default_rng(seed)
    days = {d: g for d, g in df.groupby("date")}
    if len(days) < 3:
        return {}
    sub = {d: g[["pace", "olci"]].to_numpy() for d, g in days.items()}
    keys = list(sub)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        a = np.vstack([sub[keys[i]] for i in pick])
        q = rb(a[:, 0], a[:, 1], var)
        out.append((q["r"], q["eps"], q["beta"]))
    lo, hi = np.nanpercentile(np.array(out), [2.5, 97.5], axis=0)
    return dict(r_lo=lo[0], r_hi=hi[0], eps_lo=lo[1], eps_hi=hi[1], beta_lo=lo[2], beta_hi=hi[2])


def stratified(m, by, vars_, with_ci=True, min_n=200, min_days=3):
    rows = []
    for v in vars_:
        mv = m[m["var"] == v]
        groups = [("all", mv)] if by is None else [(str(k), g) for k, g in mv.groupby(by, observed=True)]
        for k, g in groups:
            if len(g) < min_n:
                continue
            q = rb(g.pace, g.olci, v)
            row = dict(var=v, stratum=k, n=q["n"], days=g.date.nunique(), r=q["r"], eps=q["eps"], beta=q["beta"],
                       med_pace=float(np.median(g.pace)), med_diff=float(np.median(g.olci - g.pace)))
            if with_ci and g.date.nunique() >= min_days:
                row.update(boot(g, v))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- analyses
def per_day(m, vars_):
    rows = []
    for (v, d), g in m[m["var"].isin(vars_)].groupby(["var", "date"]):
        q = rb(g.pace, g.olci, v)
        rows.append(dict(var=v, date=d, n=q["n"], r=q["r"], eps=q["eps"], beta=q["beta"],
                         vza=g.pace_vza.median(), sza_p=g.pace_sza.median(), sza_o=g.olci_sza.median()))
    return pd.DataFrame(rows)


def geometry(m, outdir):
    """Does the OLCI-PACE difference depend on PACE view zenith angle or solar zenith angle?"""
    vb = [0, 45, 52, 58, 65, 90]
    m = m.copy()
    m["vza_bin"] = pd.cut(m.pace_vza, vb)
    s1 = stratified(m, "vza_bin", [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"])
    s1.to_csv(outdir / "geometry_vza_bins.csv", index=False)
    # within-day contrast: each day split into low/high VZA thirds; difference in beta (high - low)
    rows = []
    for (v, d), g in m[m["var"].isin([f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"])].groupby(["var", "date"]):
        g = g.dropna(subset=["pace_vza"])
        if len(g) < 600:
            continue
        lo_q, hi_q = g.pace_vza.quantile([1 / 3, 2 / 3])
        a, b = g[g.pace_vza <= lo_q], g[g.pace_vza >= hi_q]
        qa, qb = rb(a.pace, a.olci, v), rb(b.pace, b.olci, v)
        rows.append(dict(var=v, date=d, beta_low=qa["beta"], beta_high=qb["beta"], d_beta=qb["beta"] - qa["beta"],
                         vza_low=a.pace_vza.median(), vza_high=b.pace_vza.median()))
    w = pd.DataFrame(rows)
    summ = []
    for v, g in w.groupby("var"):
        x = g.d_beta.dropna()
        rng = np.random.default_rng(2)
        bs = [rng.choice(x.values, len(x)).mean() for _ in range(2000)]
        try:
            p = sst.wilcoxon(x).pvalue
        except ValueError:
            p = np.nan
        summ.append(dict(var=v, days=len(x), mean_d_beta=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         median_d_beta=x.median(), frac_positive=(x > 0).mean(), wilcoxon_p=p,
                         vza_low=g.vza_low.median(), vza_high=g.vza_high.median()))
    summ = pd.DataFrame(summ)
    summ.to_csv(outdir / "geometry_within_day.csv", index=False)
    # same contrast using only open ocean (> 50 km, chl < 1) to remove coastal/water-type confounding
    off = m[(m.dist_km > 50) & (m.chl < 1)]
    rows = []
    for (v, d), g in off[off["var"].isin([f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"])].groupby(["var", "date"]):
        g = g.dropna(subset=["pace_vza"])
        if len(g) < 600:
            continue
        lo_q, hi_q = g.pace_vza.quantile([1 / 3, 2 / 3])
        a, b = g[g.pace_vza <= lo_q], g[g.pace_vza >= hi_q]
        rows.append(dict(var=v, date=d, d_beta=rb(b.pace, b.olci, v)["beta"] - rb(a.pace, a.olci, v)["beta"]))
    w2 = pd.DataFrame(rows)
    s2 = []
    for v, g in w2.groupby("var"):
        x = g.d_beta.dropna()
        rng = np.random.default_rng(3)
        bs = [rng.choice(x.values, len(x)).mean() for _ in range(2000)]
        s2.append(dict(var=v, days=len(x), mean_d_beta=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(s2).to_csv(outdir / "geometry_within_day_openocean.csv", index=False)
    # day-level rank correlations of beta with day-median geometry and with overpass lag
    pdd = per_day(m, [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"])
    pdd.to_csv(outdir / "per_day.csv", index=False)
    rc = []
    for v, g in pdd.groupby("var"):
        for k in ("vza", "sza_p", "sza_o"):
            r = sst.spearmanr(g[k], g.beta, nan_policy="omit")
            rc.append(dict(var=v, driver=k, rho=r.statistic, p=r.pvalue, days=g[k].notna().sum()))
    pd.DataFrame(rc).to_csv(outdir / "geometry_day_correlations.csv", index=False)
    sz = m.assign(sza_bin=pd.cut(m.olci_sza, [0, 25, 35, 45, 60, 90]))
    stratified(sz, "sza_bin", [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"]).to_csv(outdir / "geometry_sza_bins.csv", index=False)
    return s1, summ, pd.DataFrame(rc)


def spectral(m, outdir):
    bands = [f"Rrs_{w:g}" for w in ALL_BANDS]
    s = stratified(m, None, bands)
    s["wl"] = [float(v.split("_")[1]) for v in s["var"]]
    s.to_csv(outdir / "spectral_all.csv", index=False)
    sz = stratified(m, "zone", bands, with_ci=False)
    sz["wl"] = [float(v.split("_")[1]) for v in sz["var"]]
    sz.to_csv(outdir / "spectral_by_zone.csv", index=False)
    sc = stratified(m, "chl_class", bands, with_ci=False)
    sc["wl"] = [float(v.split("_")[1]) for v in sc["var"]]
    sc.to_csv(outdir / "spectral_by_chl_class.csv", index=False)
    return s, sz, sc


def coverage(cfg, outdir):
    """Share of OLCI-valid chlorophyll cells with no valid PACE value, by distance from land (fine bins and the analysis zones)."""
    grids = load_grids(cfg)
    dist, _ = coast_distance_km(grids)
    out = {}
    for name, edges, labels in (("fine", [0, 2, 5, 10, 25, 50, 100, 1e9], None), ("zone", [0, 2, 10, 50, 1e9], ["0-2", "2-10", "10-50", ">50"])):
        tot = np.zeros(len(edges) - 1); miss = np.zeros(len(edges) - 1)
        for g in grids.values():
            o = np.isfinite(g["olci_chlor_a"].values); p = np.isfinite(g["pace_chlor_a"].values)
            for i in range(len(edges) - 1):
                k = o & (dist > edges[i]) & (dist <= edges[i + 1])
                tot[i] += k.sum(); miss[i] += (k & ~p).sum()
        df = pd.DataFrame(dict(d_from=edges[:-1], d_to=edges[1:], olci_cells=tot, pace_missing_pct=100 * miss / np.maximum(tot, 1)))
        if labels:
            df.index = labels
        df.to_csv(outdir / f"coverage_{name}.csv")
        out[name] = df
    return out["fine"], out["zone"]


def rma(x, y):
    """Reduced-major-axis slope and intercept (symmetric regression, no regression dilution)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sign(np.corrcoef(x, y)[0, 1]) * y.std() / x.std()
    return s, y.mean() - s * x.mean()


def regression(m, outdir, vars_, n=BOOT):
    """OLCI = a + b * PACE (RMA, log10 for chlorophyll); intervals by resampling days."""
    rng = np.random.default_rng(9)
    rows = []
    for v in vars_:
        g = m[m["var"] == v]
        tr = (lambda a: np.log10(a)) if v == "chlor_a" else (lambda a: a)
        b, a = rma(tr(g.pace), tr(g.olci))
        days = {d: h for d, h in g.groupby("date")}
        keys = list(days); out = []
        for _ in range(n):
            h = pd.concat([days[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            out.append(rma(tr(h.pace), tr(h.olci)))
        out = np.array(out)
        lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
        rows.append(dict(var=v, slope=b, slope_lo=lo[0], slope_hi=hi[0], intercept=a, int_lo=lo[1], int_hi=hi[1]))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "regression_rma.csv", index=False)
    return df


def run_core(cfg):
    outdir = cfg["paths"]["out"] / "final"; outdir.mkdir(exist_ok=True, parents=True)
    m = long_table(cfg)
    m.to_parquet(outdir / "long_table.parquet") if False else None
    prim = [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"]
    t_all = stratified(m, None, prim); t_all.to_csv(outdir / "pooled.csv", index=False)
    t_zone = stratified(m, "zone", prim); t_zone.to_csv(outdir / "by_zone.csv", index=False)
    t_chl = stratified(m, "chl_class", prim); t_chl.to_csv(outdir / "by_chl_class.csv", index=False)
    cov = coverage(cfg, outdir)
    coverage_independent(cfg, outdir)
    spec = spectral(m, outdir)
    geo = geometry(m, outdir)
    regression(m, outdir, [f"Rrs_{w:g}" for w in ALL_BANDS if w < 700] + ["chlor_a"])
    return m, t_all, t_zone, t_chl, cov, spec, geo


def coverage_independent(cfg, outdir):
    """Share of OLCI-valid chlorophyll cells inside the PACE swath without a valid PACE value on the days NOT chosen by the clear-sky rule."""
    import glob
    import xarray as xr
    from .coast import grid_distance_km
    files = sorted(glob.glob(str(cfg["paths"]["processed"].parent / "processed_insitu" / "grid_*.nc")))
    main_days = set(load_grids(cfg))
    files = [f for f in files if Path(f).stem[5:] not in main_days]
    edges, labels = [0, 2, 10, 50, 1e9], ["0-2", "2-10", "10-50", ">50"]
    tot = np.zeros(4); miss = np.zeros(4); dist = None; used = 0
    for f in files:
        g = xr.open_dataset(f).load()
        if "pace_chlor_a" not in g or "olci_chlor_a" not in g:
            continue
        used += 1
        if dist is None:
            dist = grid_distance_km(g.lat.values, g.lon.values)
        swath = np.isfinite(g["pace_vza"].values) if "pace_vza" in g else np.ones(dist.shape, bool)
        o = np.isfinite(g["olci_chlor_a"].values) & swath; p = np.isfinite(g["pace_chlor_a"].values)
        for i in range(4):
            k = o & (dist > edges[i]) & (dist <= edges[i + 1])
            tot[i] += k.sum(); miss[i] += (k & ~p).sum()
    df = pd.DataFrame(dict(d_from=edges[:-1], d_to=edges[1:], olci_cells=tot, pace_missing_pct=100 * miss / np.maximum(tot, 1)), index=labels)
    df["days"] = used
    df.to_csv(outdir / "coverage_independent.csv")
    return df
