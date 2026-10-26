"""Granule start times over the study area from the NASA CMR catalog, for days whose raw granules are no longer on disk."""
import re

import earthaccess
import pandas as pd

from .acquire import _bbox, login

KEYS = {"pace_aop": "p", "olci_3a": "o", "olci_3b": "o"}


def fetch(cfg, dates, out_csv):
    """One row per granule: date (YYYYMMDD), sensor ('p' PACE, 'o' OLCI), granule name and start hour (UTC)."""
    login()
    rows = []
    for d in sorted(set(dates)):
        iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        for key, s in KEYS.items():
            res = earthaccess.search_data(short_name=cfg["products"][key]["short_name"], bounding_box=_bbox(cfg), temporal=(iso, iso))
            for g in res:
                name = g.data_links()[0].split("/")[-1] if g.data_links() else ""
                m = re.search(r"\.(\d{8})T(\d{6})", name)
                if m and m.group(1) == d:
                    rows.append(dict(date=d, sensor=s, product=key, granule=name, hour=int(m.group(2)[:2]) + int(m.group(2)[2:4]) / 60))
        print(d, sum(r["date"] == d for r in rows), "granules", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
