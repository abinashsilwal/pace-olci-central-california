"""Satellite vs in situ: final statistics (protocol in Methods 'In situ matchups')."""
import numpy as np
import pandas as pd

from . import coast
from .final import rb
from .insitu import matchups
from .stats import coast_distance_km, load_grids

VARS = ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]
EDGES = [(-1, 10, "<10 km"), (10, 30, "10-30 km"), (30, 1e9, ">30 km")]


def _boot_days(g, fn, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    days = [h for _, h in g.groupby("date")]
    if len(days) < 3:
        return (np.nan, np.nan)
    out = []
    for _ in range(n):
        h = pd.concat([days[i] for i in rng.integers(0, len(days), len(days))])
        out.append(fn(h))
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def station_table(cfg, outdir):
    m = matchups(cfg, 0)
    m["dist_km"] = coast.distance_km(m.lat.values, m.lon.values)
    tp, to = _overpass_hours(cfg)
    dd = m["date"].astype(str).str.replace("-", "")
    # days without raw granules (no exact time) take the median overpass time of the days with granules
    m["dt_pace"] = np.abs(m.hour_utc - dd.map(tp).fillna(float(np.median(list(tp.values())))))
    m["dt_olci"] = np.abs(m.hour_utc - dd.map(to).fillna(float(np.median(list(to.values())))))
    m = m.dropna(subset=["insitu", "pace", "olci"])
    m = m[(m.insitu > 0) & (m.pace > 0) & (m.olci > 0)]
    m.to_csv(outdir / "insitu_pairs_sameday.csv", index=False)
    m = m[(m.dt_pace <= 3) & (m.dt_olci <= 3)].copy()      # primary: observation within 3 h of both overpasses
    m.to_csv(outdir / "insitu_pairs.csv", index=False)
    return m


def _stats(g, v):
    rows = []
    for sen in ("pace", "olci"):
        q = rb(g.insitu, g[sen], v)
        lo = _boot_days(g, lambda h: rb(h.insitu, h[sen], v)["beta"])
        elo = _boot_days(g, lambda h: rb(h.insitu, h[sen], v)["eps"])
        rows.append(dict(sensor=sen, n=q["n"], days=g.date.nunique(), r=q["r"], eps=q["eps"], eps_lo=elo[0], eps_hi=elo[1],
                         beta=q["beta"], beta_lo=lo[0], beta_hi=lo[1]))
    # which sensor is closer: median over stations of |log10(PACE/obs)| - |log10(OLCI/obs)| (> 0: OLCI closer)
    d = np.abs(np.log10(g.pace / g.insitu)) - np.abs(np.log10(g.olci / g.insitu))
    fn = lambda h: float(np.median(np.abs(np.log10(h.pace / h.insitu)) - np.abs(np.log10(h.olci / h.insitu))))
    lo = _boot_days(g, fn)
    for r in rows:
        r.update(closer_median=float(d.median()), closer_lo=lo[0], closer_hi=lo[1], frac_olci_closer=float((d > 0).mean()))
    return rows


def final_stats(cfg, outdir):
    m = station_table(cfg, outdir)
    surf = m[m.group != "airborne"]
    rows = []
    for v in VARS:
        gv = surf[surf["var"] == v]
        subsets = [("all", gv)] + [(lab, gv[(gv.dist_km > a) & (gv.dist_km <= b)]) for a, b, lab in EDGES]
        for lab, g in subsets:
            if len(g) >= 8 and g.date.nunique() >= 3:
                for r in _stats(g, v):
                    rows.append(dict(var=v, subset=lab, group="surface", **r))
        for kind, g in gv.groupby("kind"):
            if len(g) >= 8:
                for r in _stats(g, v):
                    rows.append(dict(var=v, subset=f"kind:{kind}", group="surface", **r))
    air = m[m.group == "airborne"]
    for v in VARS[:4]:
        g = air[air["var"] == v]
        if len(g) >= 8 and g.date.nunique() >= 3:
            for r in _stats(g, v):
                rows.append(dict(var=v, subset="all", group="airborne", **r))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "insitu_final.csv", index=False)
    return m, df


def _overpass_hours(cfg):
    """Median granule start hour (UTC) of PACE and OLCI over the study area, by date (YYYYMMDD)."""
    import glob, re
    t = {}
    for key in ("pace_aop", "olci_3a", "olci_3b"):
        for f in glob.glob(f"{cfg['paths']['raw']}/{key}/*.nc"):
            mm = re.search(r"\.(\d{8})T(\d{6})", f)
            if mm:
                t.setdefault(mm.group(1), {}).setdefault("p" if key == "pace_aop" else "o", []).append(int(mm.group(2)[:2]) + int(mm.group(2)[2:4]) / 60)
    # days whose raw granules were deleted after gridding: granule start times from the CMR catalog (overpass_cmr.fetch)
    cmr = cfg["paths"]["out"] / "final" / "overpass_times_cmr.csv"
    if cmr.exists():
        c = pd.read_csv(cmr, dtype={"date": str})
        for (d, s), g in c.groupby(["date", "sensor"]):
            if s not in t.get(d, {}):
                t.setdefault(d, {})[s] = list(g.hour)
    tp = {d: float(np.median(v["p"])) for d, v in t.items() if "p" in v}
    to = {d: float(np.median(v["o"])) for d, v in t.items() if "o" in v}
    return tp, to


def matchup_sensitivity(cfg, m, outdir):
    """Surface stations: primary (within 3 h of both overpasses), same day 15-24 UTC, and centre cell only (within 3 h)."""
    full = pd.read_csv(outdir / "insitu_pairs_sameday.csv")
    surf = m[m.group != "airborne"]
    full = full[full.group != "airborne"]
    c1 = matchups(cfg, 0, k=0, min_valid=1)
    c1["dist_km"] = coast.distance_km(c1.lat.values, c1.lon.values)
    c1 = c1[c1.group != "airborne"].dropna(subset=["insitu", "pace", "olci"])
    c1 = c1[(c1.insitu > 0) & (c1.pace > 0) & (c1.olci > 0)]
    tp, to = _overpass_hours(cfg)
    dd = c1["date"].astype(str).str.replace("-", "")
    c1["dt_pace"] = np.abs(c1.hour_utc - dd.map(tp).fillna(float(np.median(list(tp.values())))))
    c1["dt_olci"] = np.abs(c1.hour_utc - dd.map(to).fillna(float(np.median(list(to.values())))))
    c1 = c1[(c1.dt_pace <= 3) & (c1.dt_olci <= 3)]
    variants = {"Within 3 h of both overpasses, 3 x 3 window (as used)": surf,
                "Same day, 15-24 UTC": full,
                "Within 3 h, centre cell only": c1}
    rows = []
    for lab, d in variants.items():
        for v in VARS:
            g = d[d["var"] == v]
            if len(g) < 8:
                continue
            for sen in ("pace", "olci"):
                q = rb(g.insitu, g[sen], v)
                rows.append(dict(test=lab, var=v, sensor=sen, n=q["n"], days=g.date.nunique(), eps=q["eps"], beta=q["beta"]))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "insitu_sensitivity.csv", index=False)
    dt = surf.dropna(subset=["dt_pace", "dt_olci"])
    pd.DataFrame(dict(stat=["median_dt_pace_h", "median_dt_olci_h", "max_dt_pace_h", "max_dt_olci_h"],
                      value=[dt.dt_pace.median(), dt.dt_olci.median(), dt.dt_pace.max(), dt.dt_olci.max()])).to_csv(outdir / "insitu_time_offsets.csv", index=False)
    return df
