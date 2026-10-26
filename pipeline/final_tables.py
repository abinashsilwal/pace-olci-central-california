"""LaTeX tables generated from the result files (never edited by hand)."""
import numpy as np
import pandas as pd

NAME = {"Rrs_442.5": r"$R_{rs}$ 442.5 nm", "Rrs_490": r"$R_{rs}$ 490 nm", "Rrs_560": r"$R_{rs}$ 560 nm", "Rrs_665": r"$R_{rs}$ 665 nm",
        "chlor_a": "Chl-a"}
ROW = r" \\"


def _wrap(lines, caption, label, cols, head, small=True, star=False):
    out = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}"]
    if small:
        out.append(r"\small")
    out += [r"\begin{tabular}{" + cols + "}", r"\hline", head + ROW, r"\hline"] + lines + [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def f(x, k=0, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if round(float(x), k) == 0:          # no signed zero (-0.0, +0)
        x, sign = 0.0, False
    s = f"{x:+.{k}f}" if sign else f"{x:.{k}f}"
    return s.replace("-", "$-$")


def thousands(n):
    return f"{int(n):,}".replace(",", r"\,")


def table_pooled(t, path):
    lines = []
    for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]:
        q = t[t["var"] == v].iloc[0]
        lines.append(f"{NAME[v]} & {thousands(q.n)} & {f(q.r,2)} [{f(q.r_lo,2)}, {f(q.r_hi,2)}] & {f(q.eps,1)} [{f(q.eps_lo,1)}, {f(q.eps_hi,1)}] & {f(q.beta,1,True)} [{f(q.beta_lo,1,True)}, {f(q.beta_hi,1,True)}]" + ROW)
    cap = (r"Agreement between OLCI ($y$) and PACE ($x$), all analyzed days pooled. $n$: cells valid in both sensors and more than 2~km from the coast. "
           r"$r$: Pearson correlation (for chlorophyll, of $\log_{10}$ values). $\epsilon$: median symmetric accuracy. $\beta$: signed symmetric "
           r"percentage bias (positive: OLCI higher). Brackets: 95\% intervals from resampling days.")
    open(path, "w", encoding="utf-8", newline="\n").write(_wrap(lines, cap, "tab:pooled", "lrccc", r"Variable & $n$ & $r$ & $\epsilon$ (\%) & $\beta$ (\%)"))


def table_strata(zone, chl, cov, path):
    tot_z = zone[zone["var"] == "chlor_a"].set_index("stratum").n
    tot_c = chl[chl["var"] == "chlor_a"].set_index("stratum").n
    lines = [r"\multicolumn{7}{l}{\textit{Distance from the coast}}" + ROW]
    for s, covrow in (("2-10 km", "2-10"), ("10-50 km", "10-50"), (">50 km", ">50")):
        c = cov.loc[covrow]
        cells = []
        for v in ["chlor_a", "Rrs_442.5", "Rrs_490", "Rrs_560"]:
            q = zone[(zone["var"] == v) & (zone.stratum == s)]
            cells.append(f"{f(q.eps.iloc[0],0)} / {f(q.beta.iloc[0],0,True)}" if len(q) else "--")
        lines.append(f"{s.replace('-', '--').replace('>', '$>$')} & {100*tot_z[s]/tot_z.sum():.0f} & {f(c.pace_missing_pct,0)} & " + " & ".join(cells) + ROW)
    lines.append(r"\multicolumn{7}{l}{\textit{Chlorophyll-a class (mg\,m$^{-3}$)}}" + ROW)
    for s in ["<0.2", "0.2-0.5", "0.5-1", "1-3", ">3"]:
        cells = []
        for v in ["chlor_a", "Rrs_442.5", "Rrs_490", "Rrs_560"]:
            q = chl[(chl["var"] == v) & (chl.stratum == s)]
            cells.append(f"{f(q.eps.iloc[0],0)} / {f(q.beta.iloc[0],0,True)}" if len(q) else "--")
        lines.append(f"{s.replace('-', '--').replace('>', '$>$').replace('<', '$<$')} & {100*tot_c[s]/tot_c.sum():.0f} & -- & " + " & ".join(cells) + ROW)
    cap = (r"Agreement by distance from the coast and by chlorophyll class. Entries are $\epsilon$ / $\beta$ in percent. Cells: share of the matched cells. "
           r"PACE missing: share of OLCI-valid chlorophyll cells with no valid PACE pixel within 3.5~km of the cell center; for 0--2~km (excluded from all statistics) it is "
           + f"{cov.loc['0-2'].pace_missing_pct:.0f}" + r"\%.")
    open(path, "w", encoding="utf-8", newline="\n").write(_wrap(lines, cap, "tab:strata", "lrrcccc",
          r"Stratum & Cells (\%) & PACE missing (\%) & Chl-a & $R_{rs}$ 442.5 & $R_{rs}$ 490 & $R_{rs}$ 560"))


def table_insitu(df, path):
    lines = []
    for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]:
        first = True
        for sub in ["all", "<10 km", "10-30 km", ">30 km"]:
            a = df[(df["var"] == v) & (df["subset"] == sub) & (df.group == "surface")]
            if a.empty:
                continue
            p = a[a.sensor == "pace"].iloc[0]; o = a[a.sensor == "olci"].iloc[0]
            lab = sub.replace("-", "--").replace("<", "$<$").replace(">", "$>$")
            lines.append(f"{NAME[v] if first else ''} & {lab} & {int(p.n)} & {int(p.days)} & {f(p.eps,0)} & {f(p.beta,0,True)} & {f(o.eps,0)} & {f(o.beta,0,True)} & {f(o.closer_median*100,0,True)}" + ROW)
            first = False
        lines.append(r"\hline")
    lines = lines[:-1]
    cap = (r"Comparison with in situ observations at stations where both sensors were valid. $\epsilon$: median symmetric accuracy; $\beta$: signed symmetric percentage bias; "
           r"both relative to the observation (negative $\beta$: satellite lower). $\Delta_{\log}$: median over stations of "
           r"$[|\log_{10}(\mathrm{PACE}/\mathrm{obs})|-|\log_{10}(\mathrm{OLCI}/\mathrm{obs})|]\times100$; positive values mean OLCI is closer. "
           r"$n$: stations; $d$: days. Subsets are distances from the coast (km). Airborne data are in the Supplement.")
    lines = [l.replace(r"& $-$0 \\", r"& 0 \\").replace(r"& $-$0 &", r"& 0 &") for l in lines]   # no negative zero
    open(path, "w", encoding="utf-8", newline="\n").write(_wrap(lines, cap, "tab:insitu", "llrrrrrrr",
          r" & & & & \multicolumn{2}{c}{PACE} & \multicolumn{2}{c}{OLCI} & \\" + "\n" + r"Variable & Subset & $n$ & $d$ & $\epsilon$ & $\beta$ & $\epsilon$ & $\beta$ & $\Delta_{\log}$"))


def table_days(days_info, path):
    """days_info: DataFrame with columns date, valid_frac, season, cells."""
    lines = []
    d = days_info.reset_index(drop=True)
    half = (len(d) + 1) // 2
    for i in range(half):
        a = d.iloc[i]; b = d.iloc[i + half] if i + half < len(d) else None
        right = f"{b.date} & {b.valid_frac:.2f} & {thousands(b.cells)}" if b is not None else " & & "
        lines.append(f"{a.date} & {a.valid_frac:.2f} & {thousands(a.cells)} & {right}" + ROW)
    cap = r"The 40 selected days (no OLCI pixel passed quality control on one of them; see text). Clear-sky fraction: valid ocean fraction of the PACE Level-3 chlorophyll map used for selection. Cells: number of $0.02^\circ$ cells with valid chlorophyll in both sensors, before the coastal exclusion."
    open(path, "w", encoding="utf-8", newline="\n").write(_wrap(lines, cap, "tab:days", "lrr|lrr", r"Date (UTC) & Clear-sky & Cells & Date (UTC) & Clear-sky & Cells"))


def table_rma(reg, path):
    """OLCI = a + b * PACE, reduced-major-axis regression (log10 for chlorophyll)."""
    lines = []
    for _, q in reg.iterrows():
        v = q["var"]
        if v == "chlor_a":
            lab, a, alo, ahi, un = "Chl-a (log$_{10}$)", q.intercept, q.int_lo, q.int_hi, ""
        else:
            lab = r"$R_{rs}$ " + v.split("_")[1] + " nm"
            a, alo, ahi = q.intercept * 1e4, q.int_lo * 1e4, q.int_hi * 1e4
        lines.append(f"{lab} & {f(q.slope,2)} [{f(q.slope_lo,2)}, {f(q.slope_hi,2)}] & {f(a,1,True)} [{f(alo,1,True)}, {f(ahi,1,True)}]" + ROW)
    cap = (r"Reduced-major-axis regression of OLCI on PACE, OLCI $= a + b\,$PACE, all analyzed days pooled. Intercepts $a$ for $R_{rs}$ are in $10^{-4}$~sr$^{-1}$; "
           r"for chlorophyll the regression is in $\log_{10}$ units. Brackets: 95\% intervals from resampling days.")
    open(path, "w", encoding="utf-8").write(_wrap(lines, cap, "tab:rma", "lcc", r"Variable & Slope $b$ & Intercept $a$"))

