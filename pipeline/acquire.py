"""Steps 1-2: discover overlap dates and download granules (earthaccess)."""
from datetime import date, timedelta

import earthaccess
import pandas as pd

from .common import granule_date

PRODUCT_KEYS = ["pace_aop", "pace_bgc", "olci_3a", "olci_3b"]


def _bbox(cfg):
    a = cfg["study_area"]
    return (a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"])


def login():
    earthaccess.login(strategy="all", persist=True)   # env vars, ~/.netrc, or prompt once


def discover(cfg):
    """Print the dataset short names earthaccess can see, to fix config if needed."""
    login()
    for q in ["PACE OCI L2", "OLCI L2 ocean color"]:
        for d in earthaccess.search_datasets(keyword=q, provider="OB_CLOUD")[:15]:
            print(d.get_umm("ShortName"), "|", d.get_umm("EntryTitle"))


def _search(cfg, key, start, end):
    return earthaccess.search_data(short_name=cfg["products"][key]["short_name"],
                                   bounding_box=_bbox(cfg), temporal=(start, end))


def find_dates(cfg):
    """Dates with a PACE AOP+BGC granule AND an OLCI granule over the box -> csv."""
    login()
    d0, d1 = (pd.Timestamp(x) for x in cfg["date_range"])
    rows = []
    for key in PRODUCT_KEYS:
        res = _search(cfg, key, d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d"))
        for g in res:
            name = g.data_links()[0].split("/")[-1] if g.data_links() else ""
            rows.append({"product": key, "date": granule_date(name)})
        print(key, len(res), "granules")
    df = pd.DataFrame(rows).dropna()
    piv = df.pivot_table(index="date", columns="product", aggfunc="size", fill_value=0)
    for k in PRODUCT_KEYS:
        if k not in piv:
            piv[k] = 0
    ok = piv[(piv.pace_aop > 0) & (piv.pace_bgc > 0) & ((piv.olci_3a > 0) | (piv.olci_3b > 0))]
    out = cfg["paths"]["out"] / "candidate_dates.csv"
    ok.to_csv(out)
    print(f"{len(ok)} candidate dates -> {out}\nNext: screen them for cloud (Worldview / quicklooks), "
          f"put the keepers in config.yaml `dates:`.")
    return ok


def _overlap_frac(cfg, g):
    """Fraction of the AOI box covered by the granule's footprint polygon (1.0 if unknown -> keep)."""
    from shapely.geometry import Polygon, box
    a = cfg["study_area"]
    aoi = box(a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"])
    try:
        geo = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if "GPolygons" in geo:
            poly = Polygon([(p["Longitude"], p["Latitude"]) for p in geo["GPolygons"][0]["Boundary"]["Points"]])
        else:
            r = geo["BoundingRectangles"][0]
            poly = box(r["WestBoundingCoordinate"], r["SouthBoundingCoordinate"],
                       r["EastBoundingCoordinate"], r["NorthBoundingCoordinate"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly.intersection(aoi).area / aoi.area
    except Exception:
        return 1.0


def download_date(cfg, d, min_overlap=0.15):
    """Download all granules for one date, skipping edge granules covering < min_overlap of the AOI."""
    d = str(d)
    for key in PRODUCT_KEYS:
        res = [g for g in _search(cfg, key, d + "T00:00:00", d + "T23:59:59") if _overlap_frac(cfg, g) >= min_overlap]
        if not res:
            print(d, key, "no granules")
            continue
        dest = cfg["paths"]["raw"] / key
        dest.mkdir(parents=True, exist_ok=True)
        earthaccess.download(res, str(dest))
        print(d, key, len(res), "granules")


def download(cfg):
    login()
    if not cfg["dates"]:
        raise SystemExit("config `dates` is empty: run find_dates, screen, then fill it in.")
    for d in cfg["dates"]:
        download_date(cfg, d)


_HDF5_LOCK = __import__("threading").Lock()   # HDF5/netCDF4 reads deadlock when threaded


def _batch_one(cfg, d):
    from .process import process_date
    out = cfg["paths"]["processed"] / f"grid_{d.replace('-', '')}.nc"
    if out.exists():
        return d, "already processed"
    try:
        download_date(cfg, d)
        with _HDF5_LOCK:
            process_date(cfg, d)
        status = "ok"
    except Exception as e:
        status = f"FAILED: {type(e).__name__}: {e}"
    for key in ([] if cfg["qc"].get("keep_raw") else PRODUCT_KEYS):   # free the disk unless keep_raw
        for p in (cfg["paths"]["raw"] / key).glob("*.nc"):
            if granule_date(p) == d:
                p.unlink()
    return d, status


def run_batch(cfg, workers=3):
    """Dates run in parallel: download -> grid/co-locate -> delete raw granules."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    login()
    dates = list(map(str, cfg["dates"]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_batch_one, cfg, d) for d in dates]
        for f in as_completed(futs):
            d, status = f.result()
            print(d, status, flush=True)


def screen(cfg):
    """Cloud screen: for each candidate date, fraction of ocean cells in the box with a valid
    PACE daily L3 (0.1 deg) chlorophyll value. Downloads one ~10 MB file at a time, then deletes it."""
    import tempfile
    from pathlib import Path

    import numpy as np
    import xarray as xr

    login()
    cand = pd.read_csv(cfg["paths"]["out"] / "candidate_dates.csv")["date"].astype(str).tolist()
    a = cfg["study_area"]
    out = cfg["paths"]["out"] / "screen_raw.csv"
    done = set(pd.read_csv(out)["date"].astype(str)) if out.exists() else set()
    rows, valid_any = [], None
    tmp = Path(tempfile.mkdtemp())
    for i, d in enumerate(cand):
        if d in done:
            continue
        try:
            res = earthaccess.search_data(short_name="PACE_OCI_L3M_BGC", temporal=(d, d))
            res = [g for g in res if any("DAY" in l and "0p1deg" in l for l in g.data_links())]
            if not res:
                continue
            f = earthaccess.download(res[:1], str(tmp))[0]
            ds = xr.open_dataset(f).sel(lat=slice(a["lat_max"], a["lat_min"]), lon=slice(a["lon_min"], a["lon_max"]))
            valid = np.isfinite(ds["chlor_a"].values)
            ds.close(); Path(f).unlink()
            rows.append({"date": d, "n_valid": int(valid.sum())})
            row = pd.DataFrame(rows[-1:])
            row.to_csv(out, mode="a", header=not out.exists(), index=False)
        except Exception as e:
            print(d, "failed:", e)
        if i % 25 == 0:
            print(f"{i}/{len(cand)}")
    df = pd.read_csv(out)
    ref = df.n_valid.quantile(0.98)     # ~ ocean cells in a clear day
    df["valid_frac"] = (df.n_valid / ref).clip(upper=1)
    df = df.sort_values("valid_frac", ascending=False)
    df.to_csv(cfg["paths"]["out"] / "screened_dates.csv", index=False)
    print(df.head(40).to_string(index=False))
