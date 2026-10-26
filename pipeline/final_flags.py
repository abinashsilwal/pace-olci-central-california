"""Why does PACE lose the coastal strip? Share of PACE pixels failing the quality mask, by distance from land and by flag."""
import numpy as np
import pandas as pd
import xarray as xr

from .common import Grid, _OC_BAD, granule_date
from .stats import coast_distance_km, load_grids

REPORT = ["LAND", "STRAYLIGHT", "CLDICE", "HISATZEN", "LOWLW", "HIGLINT", "ATMFAIL"]


def flags_by_distance(cfg, outdir):
    grid = Grid(cfg)
    grids = load_grids(cfg)
    dist, _ = coast_distance_km(grids)
    edges = [(0, 2, "0-2 km"), (2, 5, "2-5 km"), (5, 10, "5-10 km"), (10, 50, "10-50 km"), (50, 1e9, ">50 km")]
    days = [str(d) for d in cfg["dates"]]
    acc = {lab: dict(n=0, any=0, **{f: 0 for f in REPORT}) for _, _, lab in edges}
    n_gran = 0
    for f in sorted((cfg["paths"]["raw"] / "pace_aop").glob("*.nc")):
        if granule_date(f) not in days:
            continue
        nav = xr.open_dataset(f, group="navigation_data")
        geo = xr.open_dataset(f, group="geophysical_data")
        lat, lon = nav["latitude"].values, nav["longitude"].values
        fl = geo["l2_flags"]
        masks = np.atleast_1d(fl.attrs["flag_masks"]).astype("int64")
        names = str(fl.attrs["flag_meanings"]).split()
        bits = {n: int(m) for n, m in zip(names, masks)}
        v = fl.values.astype("int64")
        bad = np.zeros(v.shape, bool)
        for n in _OC_BAD:
            bad |= (v & bits[n]) != 0
        flat, ok = grid.index(lat, lon)
        d = np.full(lat.shape, np.nan)
        d[ok] = dist.ravel()[flat[ok]]
        n_gran += 1
        for a, b, lab in edges:
            k = ok & (d > a) & (d <= b)
            if not k.any():
                continue
            acc[lab]["n"] += int(k.sum()); acc[lab]["any"] += int(bad[k].sum())
            for name in REPORT:
                acc[lab][name] += int(((v[k] & bits[name]) != 0).sum())
    rows = []
    for _, _, lab in edges:
        a = acc[lab]
        rows.append(dict(zone=lab, pixels=a["n"], pct_any=100 * a["any"] / max(a["n"], 1), **{f"pct_{f}": 100 * a[f] / max(a["n"], 1) for f in REPORT}))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pace_flags_by_distance.csv", index=False)
    print("granules:", n_gran)
    return df
