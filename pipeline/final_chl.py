"""Why does chlorophyll agree when reflectance does not? Propagate the measured reflectance differences through the two
spectral forms the chlorophyll algorithm uses: the colour index (curvature) at low chlorophyll and the blue-green ratio at higher chlorophyll."""
import numpy as np
import pandas as pd

BANDS = ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665"]


def _wide(m):
    keys = ["date", "lat", "lon"]
    w = None
    for v in BANDS + ["chlor_a"]:
        t = m[m["var"] == v][keys + ["pace", "olci"]].rename(columns={"pace": f"p_{v}", "olci": f"o_{v}"})
        w = t if w is None else w.merge(t, on=keys, how="inner")
    return w


def ci(r442, r560, r665):
    return r560 - (r442 + (560 - 442.5) / (665 - 442.5) * (r665 - r442))


def chl_sensitivity(m, outdir):
    w = _wide(m)
    for s in ("p", "o"):
        w[f"{s}_ci"] = ci(w[f"{s}_Rrs_442.5"], w[f"{s}_Rrs_560"], w[f"{s}_Rrs_665"])
        w[f"{s}_x"] = np.log10(np.maximum(w[f"{s}_Rrs_490"], 1e-6) / np.maximum(w[f"{s}_Rrs_560"], 1e-6))
        w[f"{s}_lc"] = np.log10(w[f"{s}_chlor_a"])
    out = {"n_cells": len(w)}
    # low-chlorophyll regime: colour index
    lo = w[w.p_chlor_a < 0.25]
    a1 = np.polyfit(lo.p_ci, lo.p_lc, 1)[0]
    d_ci = float(np.median(lo.o_ci - lo.p_ci))
    out.update(n_lo=len(lo), ci_slope=a1, ci_delta_e4=d_ci * 1e4, ci_spread_e4=float(np.subtract(*np.percentile(lo.p_ci, [75, 25])) * 1e4),
               ci_pred_pct=100 * (10 ** (a1 * d_ci) - 1), ci_obs_pct=100 * (10 ** float(np.median(lo.o_lc - lo.p_lc)) - 1))
    # higher chlorophyll: blue-green ratio
    hi = w[w.p_chlor_a > 0.35]
    b1 = np.polyfit(hi.p_x, hi.p_lc, 1)[0]
    d_x = float(np.median(hi.o_x - hi.p_x))
    out.update(n_hi=len(hi), ratio_slope=b1, ratio_delta=d_x, ratio_spread=float(np.subtract(*np.percentile(hi.p_x, [75, 25]))),
               ratio_pred_pct=100 * (10 ** (b1 * d_x) - 1), ratio_obs_pct=100 * (10 ** float(np.median(hi.o_lc - hi.p_lc)) - 1),
               rel_bias_490=100 * float(np.median((hi.o_Rrs_490 - hi.p_Rrs_490) / hi.p_Rrs_490)), rel_bias_560=100 * float(np.median((hi.o_Rrs_560 - hi.p_Rrs_560) / hi.p_Rrs_560)))
    df = pd.DataFrame([out]); df.to_csv(outdir / "chl_sensitivity.csv", index=False)
    return out
