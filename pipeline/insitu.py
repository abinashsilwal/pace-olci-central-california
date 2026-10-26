"""Satellite vs in situ matchups (PACE-PAX, Plumes & Blooms, PVST_CALCOFI, PVST_SBCR, PVST_AIRSHARP, NMFS_IWCPS_AWR).

In situ side (prepared by the data-acquisition steps): data/insitu/{chl,rrs}_aoi.pkl and {chl,rrs}_aoi2.pkl.
Satellite side: gridded files in processed_insitu (same grid and processing as the main 20 days).

Rules (stated in the paper): surface samples (<= 5 m); in situ time 15-24 UTC (overpasses 17.8-21.2 UTC);
satellite value = median of the valid, non-land cells in the 3 x 3 window (0.06 deg) around the station,
at least 3 valid cells; both sensors required valid for a paired comparison.
Chlorophyll: HPLC total chlorophyll-a and bottle CHL; fluorescence proxies are excluded.
Rrs: station, cast, above-water and airborne spectra boxcar-averaged +/-5 nm at the four bands.
Airborne (C-AIR) segments are kept as a separate group.
"""
import numpy as np
import pandas as pd
import xarray as xr

from .extra import robust

INSITU = "data/insitu"
BANDS = ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665"]


def _window(ds, var, lat, lon, k=1, min_valid=3, min_pace_pixels=2):
    if var not in ds:
        return np.nan, 0
    i = int(np.abs(ds.lat.values - lat).argmin()); j = int(np.abs(ds.lon.values - lon).argmin())
    sl = (slice(max(i - k, 0), i + k + 1), slice(max(j - k, 0), j + k + 1))
    a = ds[var].values[sl].astype(float)
    if var.startswith("pace_") and var + "_n" in ds:
        a = np.where(ds[var + "_n"].values[sl] >= min_pace_pixels, a, np.nan)
    land = ds["land"].values[sl].astype(bool)
    v = a[np.isfinite(a) & ~land]
    return (float(np.median(v)) if v.size >= min_valid else np.nan), int(v.size)


def _stations(df, valcols):
    """Average replicates: same source, date, 0.02 deg position, same 30-min block."""
    d = df.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
    d["blk"] = d.time_utc.dt.floor("30min")
    d["la"] = (d.lat / 0.02).round(); d["lo"] = (d.lon / 0.02).round()
    g = d.groupby(["source", "group", "kind", "date", "blk", "la", "lo"], as_index=False)
    return g.agg({**{c: "mean" for c in valcols}, "lat": "mean", "lon": "mean", "hour": "mean"})


def load_insitu():
    chl = pd.concat([pd.read_pickle(f"{INSITU}/chl_aoi.pkl"), pd.read_pickle(f"{INSITU}/chl_aoi2.pkl")], ignore_index=True)
    chl = chl[chl.kind.isin(["HPLC_chl", "PnB_bottle_CHL"]) & (chl.depth_m <= 5) & chl.in_window].copy()
    chl["group"] = "HPLC/bottle"
    r1 = pd.read_pickle(f"{INSITU}/rrs_aoi.pkl")
    r1 = r1[r1.note.isin(["station", "cast", "above_water"]) & r1.in_window].copy()
    r2 = pd.read_pickle(f"{INSITU}/rrs_aoi2.pkl")
    r2 = r2[r2.in_window].copy()
    rrs = pd.concat([r1, r2], ignore_index=True)
    rrs["group"] = np.where(rrs.kind == "CAIR_Rrs", "airborne", "surface")
    return chl, rrs


def matchups(cfg, day_window=0, k=1, min_valid=3):
    """day_window=0: same UTC day only. 1: also accept a satellite grid one day before/after (sensitivity)."""
    gdirs = [cfg["paths"]["processed"], cfg["paths"]["processed"].parent / "processed_insitu"]   # final-run grids take precedence
    grids = {}
    for gd in gdirs:
        for f in sorted(gd.glob("grid_*.nc")):
            grids.setdefault(f.stem[5:], f)
    cache = {}

    def get(key):
        if key not in cache:
            cache[key] = xr.open_dataset(grids[key]).load()
        return cache[key]

    def grid_for(date):
        base = pd.Timestamp(date)
        for off in ([0] if day_window == 0 else [0, -1, 1]):
            k = (base + pd.Timedelta(days=off)).strftime("%Y%m%d")
            if k in grids:
                return get(k), off
        return None, None

    chl, rrs = load_insitu()
    rows = []
    for _, r in _stations(chl, ["value"]).iterrows():
        ds, off = grid_for(r["date"])
        if ds is None:
            continue
        p, npc = _window(ds, "pace_chlor_a", r.lat, r.lon, k, min_valid); o, noc = _window(ds, "olci_chlor_a", r.lat, r.lon, k, min_valid)
        rows.append(dict(var="chlor_a", source=r.source, kind=r.kind, group=r.group, date=r["date"], day_offset=off, lat=r.lat, lon=r.lon,
                         hour_utc=r.hour, insitu=r.value, pace=p, olci=o, n_pace=npc, n_olci=noc))
    for _, r in _stations(rrs, BANDS).iterrows():
        ds, off = grid_for(r["date"])
        if ds is None:
            continue
        for b in BANDS:
            p, npc = _window(ds, f"pace_{b}", r.lat, r.lon, k, min_valid); o, noc = _window(ds, f"olci_{b}", r.lat, r.lon, k, min_valid)
            rows.append(dict(var=b, source=r.source, kind=r.kind, group=r.group, date=r["date"], day_offset=off, lat=r.lat, lon=r.lon,
                             hour_utc=r.hour, insitu=r[b], pace=p, olci=o, n_pace=npc, n_olci=noc))
    return pd.DataFrame(rows)


def _stat_rows(m, label):
    out = []
    for (v, grp), g in m.groupby(["var", "group"]):
        both = g.dropna(subset=["insitu", "pace", "olci"])
        both = both[(both.insitu > 0) & (both.pace > 0) & (both.olci > 0)]
        for sensor in ("pace", "olci"):
            base = dict(set=label, var=v, group=grp, sensor=sensor, n_pairs=len(both), n_dates=both.date.nunique(),
                        n_cells_clusters=both.groupby(["date", "source"]).ngroups)
            if len(both) >= 5:
                q = robust(both["insitu"], both[sensor])
                base.update(r=q["r"], r_log=q["r_log"], eps=q["eps_pct"], beta=q["beta_pct"], RMSE=q["RMSE"], bias=q["bias"])
            out.append(base)
    return out


def insitu_stats(cfg):
    m0 = matchups(cfg, day_window=0)
    m1 = matchups(cfg, day_window=1)
    m0.to_csv(cfg["paths"]["out"] / "insitu_matchups_all.csv", index=False)
    m1.to_csv(cfg["paths"]["out"] / "insitu_matchups_pm1day.csv", index=False)
    s = pd.DataFrame(_stat_rows(m0, "same day") + _stat_rows(m1, "+/-1 day"))
    s.to_csv(cfg["paths"]["out"] / "insitu_stats.csv", index=False)
    return m0, s
