"""Grid PACE and OLCI for the days with in situ stations (dates in insitu_days.txt) into data/processed_insitu."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.acquire import run_batch
from pipeline.common import load_config

cfg = load_config()
c = copy.deepcopy(cfg)
c["dates"] = [l.strip() for l in open(Path(__file__).with_name("insitu_days.txt")) if l.strip()]
c["qc"]["keep_raw"] = False
c["paths"]["processed"] = cfg["paths"]["processed"].parent / "processed_insitu"
c["paths"]["processed"].mkdir(exist_ok=True)
run_batch(c, workers=2)
