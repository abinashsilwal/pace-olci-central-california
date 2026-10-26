"""Fetch OLCI-A / OLCI-B spectral response functions (EUMETSAT NWP SAF tabulation) into data/srf."""
import io
import os
import tarfile

import requests

os.makedirs("data/srf", exist_ok=True)
base = "https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/"
for sat, n in (("a", 1), ("b", 2)):
    name = f"rtcoef_sentinel3_{n}_olci_srf"
    r = requests.get(f"{base}{name}/{name}.tar.gz", timeout=300)
    print(sat, r.status_code, len(r.content))
    tarfile.open(fileobj=io.BytesIO(r.content)).extractall(f"data/srf/{sat}")
for root, _, files in os.walk("data/srf/a"):
    txt = sorted(f for f in files if f.endswith(".txt"))
    print(root, len(txt), txt[:3], txt[-2:])
    if txt:
        for f in (txt[0], txt[20] if len(txt) > 20 else txt[-1], txt[-1]):
            print("--", f)
            print("".join(open(os.path.join(root, f)).readlines()[:6]))
        break
