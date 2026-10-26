"""Run the whole final analysis: results -> tables, figures, numbers for the paper. Usage: python run_final.py [core|insitu|sens|assemble|all]"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common import load_config
from pipeline import final_chl, final_maps, final, final_figs, final_insitu, final_numbers, final_sens, final_supp, final_tables, final_tables2, final_tables2

cfg = load_config()
OUT = cfg["paths"]["out"] / "final"; OUT.mkdir(parents=True, exist_ok=True)
PAPER = Path(cfg["paths"]["out"]) / "manuscript"
FIG = PAPER / "figures"; SEC = PAPER / "sections"
FIG.mkdir(parents=True, exist_ok=True); SEC.mkdir(parents=True, exist_ok=True)


def step(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def do_core():
    step("core: long table, pooled, strata, spectrum, geometry")
    m, t_all, t_zone, t_chl, (cov_fine, cov_zone), spec, geo = final.run_core(cfg)
    m.to_pickle(OUT / "long_table.pkl")
    final_chl.chl_sensitivity(m, OUT)
    return m


def load_long():
    return pd.read_pickle(OUT / "long_table.pkl")


def do_insitu():
    step("in situ statistics")
    m, df = final_insitu.final_stats(cfg, OUT)
    final_insitu.matchup_sensitivity(cfg, m, OUT)
    return df


def do_sens(m):
    step("sensitivity: coast / fill / band matching")
    final_sens.run_sensitivity(cfg, m, OUT)
    step("sensitivity: value limits on 10 days")
    days = [str(d) for d in cfg["dates"]][::4]
    final_sens.value_limit_variants(cfg, days, OUT)
    step("replication on in situ days")
    final_sens.replication(cfg, OUT)


def lag_table():
    import glob, re
    t = {}
    for key in ("pace_aop", "olci_3a", "olci_3b"):
        for f in glob.glob(f"{cfg['paths']['raw']}/{key}/*.nc"):
            mm = re.search(r"\.(\d{8})T(\d{6})", f)
            if mm:
                t.setdefault(mm.group(1), {}).setdefault("p" if key == "pace_aop" else "o", []).append(int(mm.group(2)[:2]) + int(mm.group(2)[2:4]) / 60)
    return t


def assemble():
    step("assemble tables, figures, numbers")
    from scipy import stats as sst
    pooled = pd.read_csv(OUT / "pooled.csv"); zone = pd.read_csv(OUT / "by_zone.csv"); chl = pd.read_csv(OUT / "by_chl_class.csv")
    cov_zone = pd.read_csv(OUT / "coverage_zone.csv", index_col=0); cov_fine = pd.read_csv(OUT / "coverage_fine.csv")
    s_all = pd.read_csv(OUT / "spectral_all.csv"); s_zone = pd.read_csv(OUT / "spectral_by_zone.csv"); s_chl = pd.read_csv(OUT / "spectral_by_chl_class.csv")
    vza = pd.read_csv(OUT / "geometry_vza_bins.csv"); within = pd.read_csv(OUT / "geometry_within_day.csv")
    within_off = pd.read_csv(OUT / "geometry_within_day_openocean.csv"); rho = pd.read_csv(OUT / "geometry_day_correlations.csv")
    sza = pd.read_csv(OUT / "geometry_sza_bins.csv"); per_day = pd.read_csv(OUT / "per_day.csv", dtype={"date": str})
    m = load_long()
    for _d in (zone, chl, s_all, s_zone, s_chl):
        _d["med_diff_e4"] = _d.med_diff * 1e4; _d["med_pace_e4"] = _d.med_pace * 1e4

    # lag per day and its correlation with beta
    t = lag_table(); lag = {d: np.median(v["p"]) - np.median(v["o"]) for d, v in t.items() if "p" in v and "o" in v}
    per_day["lag_h"] = per_day.date.map(lag)
    lag_rows = []
    for v, g in per_day.groupby("var"):
        r = sst.spearmanr(g.lag_h, g.beta, nan_policy="omit"); lag_rows.append(dict(var=v, driver="lag", rho=r.statistic, p=r.pvalue, days=g.lag_h.notna().sum()))
    rho = pd.concat([rho, pd.DataFrame(lag_rows)], ignore_index=True); rho.to_csv(OUT / "geometry_day_correlations_all.csv", index=False)

    # figures
    final_figs.fig_agreement(m, per_day, FIG)
    final_figs.fig_spectral(s_all, s_zone, s_chl, FIG)
    final_figs.fig_geometry(vza, within, per_day, FIG)
    ins = pd.read_csv(OUT / "insitu_final.csv")
    final_figs.fig_insitu(ins[ins.group == "surface"], FIG)
    final_supp.s_perday(per_day, FIG); final_supp.s_coverage(cov_fine, FIG); final_supp.s_sza(sza, FIG); final_supp.s_kind(ins, FIG)
    best, worst = final_supp.s_maps(cfg, per_day, FIG)
    typ, nmap = final_maps.fig_maps(cfg, per_day, FIG)
    final_maps.supp_gallery(cfg, FIG)
    final_maps.supp_chl_gallery(cfg, FIG)
    final_figs.fig_insitu_scatter(pd.read_csv(OUT / "insitu_pairs.csv"), FIG)

    # tables
    final_tables.table_pooled(pooled, SEC / "tab_pooled.tex")
    final_tables.table_strata(zone, chl, cov_zone, SEC / "tab_strata.tex")
    final_tables.table_insitu(ins, SEC / "tab_insitu.tex")
    reg = pd.read_csv(OUT / "regression_rma.csv"); final_tables.table_rma(reg, SEC / "tab_rma.tex")
    final_tables2.table_flags(pd.read_csv(OUT / "pace_flags_by_distance.csv"), SEC / "tab_flags.tex")
    final_tables2.table_sens(pd.read_csv(OUT / "sensitivity.csv"), pd.read_csv(OUT / "sensitivity_value_limits.csv"), SEC / "tab_sens.tex")
    final_tables2.table_rep(pd.read_csv(OUT / "replication_summary.csv"), SEC / "tab_rep.tex")
    final_tables2.table_air(ins, SEC / "tab_air.tex")
    sc = pd.read_csv(cfg["paths"]["out"] / "screened_dates.csv", dtype={"date": str}).set_index("date")
    days = [str(d) for d in cfg["dates"]]
    cells = m[m["var"] == "chlor_a"].groupby("date").size()
    info = pd.DataFrame({"date": sorted(days), "valid_frac": [sc.loc[d, "valid_frac"] for d in sorted(days)],
                         "cells": [cells.get(d.replace("-", ""), 0) for d in sorted(days)]})
    final_tables.table_days(info, SEC / "tab_days.tex")

    # numbers
    N = final_numbers.Macros()
    N.frame("pooled", pooled.assign(stratum="all"), ["var", "stratum"], ["n", "days", "r", "r_lo", "r_hi", "eps", "eps_lo", "eps_hi", "beta", "beta_lo", "beta_hi"])
    N.frame("zone", zone, ["var", "stratum"], ["n", "r", "eps", "beta", "med_diff_e4", "med_pace_e4"])
    N.frame("chlc", chl, ["var", "stratum"], ["n", "r", "eps", "beta", "med_diff_e4", "med_pace_e4"])
    N.frame("spec", s_all.assign(var=s_all.wl.map(lambda w: f"{w:g}")), ["var"], ["eps", "beta", "beta_lo", "beta_hi", "r", "med_pace_e4", "med_diff_e4", "n"])
    N.frame("specz", s_zone.assign(var=s_zone.wl.map(lambda w: f"{w:g}")), ["var", "stratum"], ["eps", "beta", "med_diff_e4"])
    N.frame("specc", s_chl.assign(var=s_chl.wl.map(lambda w: f"{w:g}")), ["var", "stratum"], ["eps", "beta", "med_diff_e4", "med_pace_e4"])
    N.frame("cov", cov_zone.reset_index().rename(columns={"index": "var"}).assign(stratum="x"), ["var"], ["pace_missing_pct"])
    N.frame("covf", cov_fine.assign(var=cov_fine.d_from.map(lambda x: f"d{x:g}")), ["var"], ["pace_missing_pct"])
    N.frame("vza", vza, ["var", "stratum"], ["n", "eps", "beta", "beta_lo", "beta_hi"])
    N.frame("within", within, ["var"], ["days", "mean_d_beta", "lo", "hi", "median_d_beta", "frac_positive", "wilcoxon_p", "vza_low", "vza_high"])
    N.frame("withinoff", within_off, ["var"], ["days", "mean_d_beta", "lo", "hi"])
    N.frame("rho", rho, ["var", "driver"], ["rho", "p", "days"])
    N.frame("sza", sza, ["var", "stratum"], ["n", "eps", "beta"])
    rg = reg.copy(); rg["intercept_e4"] = np.where(rg["var"] == "chlor_a", rg.intercept, rg.intercept * 1e4)
    rg["int_lo_e4"] = np.where(rg["var"] == "chlor_a", rg.int_lo, rg.int_lo * 1e4); rg["int_hi_e4"] = np.where(rg["var"] == "chlor_a", rg.int_hi, rg.int_hi * 1e4)
    N.frame("rma", rg, ["var"], ["slope", "slope_lo", "slope_hi", "intercept_e4", "int_lo_e4", "int_hi_e4"])
    # day-level summaries
    for v, g in per_day.groupby("var"):
        k = final_numbers.SHORT[v]
        N.put(f"day_{k}_olci_higher", float((g.beta > 0).sum()), "n"); N.put(f"day_{k}_med_r", float(g.r.median()), "r")
        N.put(f"day_{k}_min_r", float(g.r.min()), "r"); N.put(f"day_{k}_max_r", float(g.r.max()), "r"); N.put(f"day_{k}_ndays", float(len(g)), "n")
        N.put(f"day_{k}_olci_lower", float((g.beta < 0).sum()), "n")
        N.put(f"day_{k}_med_beta", float(g.beta.median()), "beta")
        N.put(f"day_{k}_sd_beta", float(g.beta.std()), "eps")
    # overpass timing and seasons
    lags = np.array(list(lag.values()))
    N.put("lag_med", float(np.median(lags)), "eps"); N.put("lag_min", float(lags.min()), "eps"); N.put("lag_max", float(lags.max()), "eps")
    allp = [h for v in t.values() for h in v.get("p", [])]; allo = [h for v in t.values() for h in v.get("o", [])]
    N.put("pace_utc_min", f"{min(allp):.1f}"); N.put("pace_utc_max", f"{max(allp):.1f}"); N.put("olci_utc_min", f"{min(allo):.1f}"); N.put("olci_utc_max", f"{max(allo):.1f}")
    mon = pd.to_datetime(pd.Series(days)).dt.month
    N.put("n_djf", str(int(mon.isin([12, 1, 2]).sum()))); N.put("n_mam", str(int(mon.isin([3, 4, 5]).sum())))
    N.put("n_jja", str(int(mon.isin([6, 7, 8]).sum()))); N.put("n_son", str(int(mon.isin([9, 10, 11]).sum())))
    N.put("ndays", str(m.date.nunique())); N.put("nsel", str(len(days)))
    # sensitivity, replication, in situ
    for fn, pre in (("sensitivity.csv", "sens"), ("sensitivity_value_limits.csv", "vl")):
        p = OUT / fn
        if p.exists():
            d = pd.read_csv(p)
            code = {"Baseline (response-function average; 2 km; cells with >= 2 PACE pixels)": "base", "Coastal exclusion 5 km": "coast5",
                    "Single-pixel and gap-filled PACE cells included": "fill", "Coarser grid (0.06 deg blocks)": "coarse",
                    "PACE view zenith below 25 deg": "lowvza", "PACE band: nearest": "nearest", "PACE band: boxcar10": "boxcar10",
                    "Value limits: as used (10 days)": "vlbase", "Value limits: no value limits (10 days)": "vlnone", "Value limits: tight limits (10 days)": "vltight", "QC: Rrs mask without CHLFAIL and COCCOLITH (10 days)": "vlqc"}
            for _, r in d.iterrows():
                key = f"{pre}_{code.get(r['test'], r['test'])}_{final_numbers.SHORT[r['var']]}"
                for s in ("n", "r", "eps", "beta"):
                    N.put(f"{key}_{s}", float(r[s]), s)
    p = OUT / "replication_summary.csv"
    if p.exists():
        _rep = pd.read_csv(p); _rep["pct_olci_higher"] = _rep.frac_olci_higher * 100
        N.frame("rep", _rep, ["var"], ["days", "median_beta", "lo", "hi", "pct_olci_higher", "median_eps", "median_r"])
    ins = ins.assign(**{f"{c}_x100": ins[c] * 100 for c in ("closer_median", "closer_lo", "closer_hi")})   # same scale as Table 3
    N.frame("ins", ins.assign(stratum=ins.subset.str.replace("kind:", "k_")), ["var", "stratum", "sensor", "group"],
            ["n", "days", "r", "eps", "eps_lo", "eps_hi", "beta", "beta_lo", "beta_hi", "closer_median", "closer_lo", "closer_hi", "frac_olci_closer",
             "closer_median_x100", "closer_lo_x100", "closer_hi_x100"])
    fl = pd.read_csv(OUT / "pace_flags_by_distance.csv")
    N.frame("flag", fl.rename(columns={"zone": "var"}), ["var"], ["pixels", "pct_any", "pct_LAND", "pct_STRAYLIGHT", "pct_CLDICE", "pct_HISATZEN", "pct_LOWLW"])
    # chlorophyll sensitivity to the reflectance differences
    cx = pd.read_csv(OUT / "chl_sensitivity.csv").iloc[0]
    for k, stat in (("n_cells", "n"), ("n_lo", "n"), ("n_hi", "n"), ("ci_delta_e4", "beta"), ("ci_spread_e4", "eps"), ("ci_pred_pct", "beta"), ("ci_obs_pct", "beta"),
                    ("ci_slope", "eps"), ("ratio_pred_pct", "beta"), ("ratio_obs_pct", "beta"), ("rel_bias_490", "beta"), ("rel_bias_560", "beta"), ("ratio_slope", "eps")):
        N.put("chlx_" + k, float(cx[k]), stat)
    N.put("chlx_ratio_delta_e3", float(cx["ratio_delta"]) * 1e3, "beta"); N.put("chlx_ratio_spread_e3", float(cx["ratio_spread"]) * 1e3, "eps")
    # coverage on the independent days, in situ matchup sensitivity and time offsets
    ci_ = pd.read_csv(OUT / "coverage_independent.csv", index_col=0)
    N.frame("covi", ci_.reset_index().rename(columns={"index": "var"}).assign(stratum="x"), ["var"], ["pace_missing_pct"]); N.put("covi_days", float(ci_["days"].iloc[0]), "n")
    isn = pd.read_csv(OUT / "insitu_sensitivity.csv")
    short = {"Within 3 h of both overpasses, 3 x 3 window (as used)": "base", "Same day, 15-24 UTC": "sameday", "Within 3 h, centre cell only": "centre"}
    N.frame("insens", isn.assign(test=isn.test.map(short)), ["test", "var", "sensor"], ["n", "days", "eps", "beta"])
    final_tables2.table_insens(isn, SEC / "tab_insens.tex")
    to_ = pd.read_csv(OUT / "insitu_time_offsets.csv").set_index("stat").value
    N.put("dt_pace_med", float(to_["median_dt_pace_h"]), "eps"); N.put("dt_olci_med", float(to_["median_dt_olci_h"]), "eps")
    N.put("dt_pace_max", float(to_["max_dt_pace_h"]), "eps"); N.put("dt_olci_max", float(to_["max_dt_olci_h"]), "eps")
    N.put("map_typ", f"{typ[:4]}-{typ[4:6]}-{typ[6:]}"); N.put("map_best", f"{best[:4]}-{best[4:6]}-{best[6:]}"); N.put("map_worst", f"{worst[:4]}-{worst[4:6]}-{worst[6:]}")
    N.write(PAPER / "numbers.tex", PAPER / "numbers_keys.txt")
    print("macros:", len(N.d), "best/worst chl day:", best, worst)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("core", "all"):
        m = do_core()
    if what in ("insitu", "all"):
        do_insitu()
    if what in ("sens", "all"):
        do_sens(load_long())
    if what in ("assemble", "all"):
        assemble()
