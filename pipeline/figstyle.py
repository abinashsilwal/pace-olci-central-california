"""One typeface for every figure of the manuscript: Source Sans 3 (text, tick labels, legends) and the same face in mathtext.

The semibold file is registered as the bold weight of the family, so panel tags and headings are bold; matplotlib would otherwise
find only the regular and italic faces and silently ignore fontweight="bold".
"""
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"
FAMILY = "Source Sans 3"
_done = False


def apply():
    global _done
    if not _done:
        for f in sorted(FONT_DIR.glob("*.ttf")):
            fm.fontManager.addfont(str(f))
        entries = []
        for e in fm.fontManager.ttflist:
            if "SourceSans3" in Path(e.fname).name:
                if "Semibold" in e.fname:
                    e = replace(e, name=FAMILY, weight=700, style="normal")
                elif "-It" in e.fname:
                    e = replace(e, name=FAMILY, weight=400, style="italic")
                else:
                    e = replace(e, name=FAMILY, weight=400, style="normal")
            entries.append(e)
        fm.fontManager.ttflist[:] = entries
        for cached in (getattr(fm.fontManager, "_findfont_cached", None), getattr(fm, "_findfont_cached", None)):
            if hasattr(cached, "cache_clear"):
                cached.cache_clear()
        _done = True
    mpl.rcParams.update({
        "font.family": FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": f"{FAMILY}:normal",
        "mathtext.it": f"{FAMILY}:italic",
        "mathtext.bf": f"{FAMILY}:bold",
        "mathtext.default": "it",
        "axes.unicode_minus": True,
    })
