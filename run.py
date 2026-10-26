"""Usage: python run.py <step>   steps: discover | find_dates | screen | download | batch | process | stats | extra | figures | all"""
import sys

from pipeline.common import load_config


def main(step):
    cfg = load_config()
    if step == "discover":
        from pipeline.acquire import discover; discover(cfg)
    elif step == "find_dates":
        from pipeline.acquire import find_dates; find_dates(cfg)
    elif step == "screen":
        from pipeline.acquire import screen; screen(cfg)
    elif step == "download":
        from pipeline.acquire import download; download(cfg)
    elif step == "batch":
        from pipeline.acquire import run_batch; run_batch(cfg)
    elif step == "process":
        from pipeline.process import process_all; process_all(cfg)
    elif step == "stats":
        from pipeline.stats import run_stats; run_stats(cfg)
    elif step == "extra":
        from pipeline.extra import run_extra; run_extra(cfg)
    elif step == "figures":
        from pipeline.figures import run_figures; run_figures(cfg)
    elif step == "all":   # after dates are chosen in config.yaml
        for s in ("download", "process", "stats", "figures"):
            main(s)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
