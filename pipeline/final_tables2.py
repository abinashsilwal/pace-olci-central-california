"""Supplementary LaTeX tables: PACE flags by distance, sensitivity tests, replication, airborne comparison."""
from .final_tables import NAME, ROW, _wrap, f, thousands


def table_flags(fl, path):
    lines = []
    for _, q in fl.iterrows():
        zone = q.zone.replace("-", "--").replace(">", "$>$")
        lines.append(f"{zone} & {thousands(q.pixels)} & {f(q.pct_any,1)} & {f(q.pct_STRAYLIGHT,1)} & {f(q.pct_CLDICE,1)} & {f(q.pct_LAND,1)} & {f(q.pct_LOWLW,1)}" + ROW)
    cap = (r"Share of PACE ocean pixels failing the quality mask, by distance from the coast, over all PACE granules used. "
           r"The first column of percentages is the share failing any flag of the mask; the others are the shares carrying each flag, and a pixel can carry several. "
           r"Stray light: STRAYLIGHT; cloud/ice: CLDICE; land: LAND (mixed pixels); low water-leaving radiance: LOWLW.")
    head = r"Distance (km) & Pixels & Any (\%) & Stray light & Cloud/ice & Land & Low $L_w$"
    open(path, "w", encoding="utf-8").write(_wrap(lines, cap, "tab:flags", "lrrrrrr", head))


def table_sens(sens, vl, path):
    names = {
        "Baseline (response-function average; 2 km; cells with >= 2 PACE pixels)": "Baseline (as in Table 1)",
        "Coastal exclusion 5 km": "Coastal exclusion 5 km",
        "Single-pixel and gap-filled PACE cells included": "Single-pixel and filled PACE cells",
        "Coarser grid (0.06 deg blocks)": r"Coarser grid ($0.06^\circ$ blocks)",
        "PACE view zenith below 25 deg": r"PACE view zenith angle $<25^\circ$",
        "PACE band: nearest": "Nearest PACE band",
        "PACE band: boxcar10": "10 nm boxcar",
    }
    lines = []

    def row(label, d):
        cells = []
        for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]:
            q = d[d["var"] == v]
            cells.append(f"{f(q.eps.iloc[0],0)} / {f(q.beta.iloc[0],0,True)}" if len(q) else "--")
        lines.append(f"{label} & " + " & ".join(cells) + ROW)

    for t, lab in names.items():
        d = sens[sens.test == t]
        if len(d):
            row(lab, d)
    lines.append(r"\hline")
    short = {"Value limits: as used (10 days)": "Baseline on the 10-day subset",
             "Value limits: no value limits (10 days)": "Subset without value limits",
             "Value limits: tight limits (10 days)": "Subset with tighter value limits",
             "QC: Rrs mask without CHLFAIL and COCCOLITH (10 days)": "Subset with reduced $R_{rs}$ mask"}
    for t in vl.test.unique():
        row(short.get(t, t), vl[vl.test == t])
    cap = (r"Sensitivity of the pooled agreement to analysis choices. Entries are $\epsilon$ / $\beta$ in percent. "
           r"The last four rows use a subset of ten days (every fourth selected day) that was reprocessed for the value-limit and quality-mask tests; the reduced reflectance mask omits the CHLFAIL and COCCOLITH flags.")
    head = r"Test & $R_{rs}$ 442.5 & $R_{rs}$ 490 & $R_{rs}$ 560 & $R_{rs}$ 665 & Chl-a"
    txt = _wrap(lines, cap, "tab:sens", "lccccc", head).replace("\\small\n", "\\small\n\\setlength{\\tabcolsep}{4pt}\n", 1)
    open(path, "w", encoding="utf-8").write(txt)


def table_rep(rep, path):
    lines = []
    for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]:
        q = rep[rep["var"] == v].iloc[0]
        lines.append(f"{NAME[v]} & {int(q.days)} & {f(q.median_beta,1,True)} [{f(q.lo,1,True)}, {f(q.hi,1,True)}] & {f(100*q.frac_olci_higher,0)} & {f(q.median_eps,1)} & {f(q.median_r,2)}" + ROW)
    cap = (r"Agreement on additional days selected independently of the clear-sky criterion (days processed for the in situ comparison). "
           r"$\beta$: median of the daily values with a 95\% interval from resampling days; OLCI higher: share of days with $\beta>0$; "
           r"$\epsilon$ and $r$: medians of the daily values.")
    head = r"Variable & Days & Median $\beta$ (\%) & OLCI higher (\%) & $\epsilon$ (\%) & $r$"
    open(path, "w", encoding="utf-8").write(_wrap(lines, cap, "tab:rep", "lrcccc", head))


def table_air(df, path):
    lines = []
    for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665"]:
        a = df[(df["var"] == v) & (df.group == "airborne") & (df.subset == "all")]
        if a.empty:
            continue
        p = a[a.sensor == "pace"].iloc[0]
        o = a[a.sensor == "olci"].iloc[0]
        lines.append(f"{NAME[v]} & {int(p.n)} & {int(p.days)} & {f(p.eps,0)} & {f(p.beta,0,True)} & {f(o.eps,0)} & {f(o.beta,0,True)}" + ROW)
    cap = (r"Comparison with airborne reflectance (C-AIR, 30~s segments) where both sensors were valid. $\epsilon$ and $\beta$ are relative to the airborne value. "
           r"The airborne data are an initial public release, have no bidirectional-reflectance correction and come from few days.")
    head = (r" & & & \multicolumn{2}{c}{PACE} & \multicolumn{2}{c}{OLCI} \\" + "\n"
            + r"Variable & Segments & Days & $\epsilon$ & $\beta$ & $\epsilon$ & $\beta$")
    open(path, "w", encoding="utf-8").write(_wrap(lines, cap, "tab:air", "lrrrrrr", head))


def table_insens(df, path):
    """In situ matchup sensitivity: error and bias of each sensor for three matchup rules."""
    lines = []
    first_test = True
    for t in df.test.unique():
        if not first_test:
            lines.append(r"\hline")
        first_test = False
        lines.append(r"\multicolumn{7}{l}{\textit{" + t.replace("x", r"$\times$") + "}}" + ROW)
        for v in ["Rrs_442.5", "Rrs_490", "Rrs_560", "chlor_a"]:
            q = df[(df.test == t) & (df["var"] == v)]
            p = q[q.sensor == "pace"]; o = q[q.sensor == "olci"]
            if p.empty or o.empty:
                continue
            p, o = p.iloc[0], o.iloc[0]
            lines.append(f"{NAME[v]} & {int(p.n)} & {int(p.days)} & {f(p.eps,0)} & {f(p.beta,0,True)} & {f(o.eps,0)} & {f(o.beta,0,True)}" + ROW)
    cap = (r"Sensitivity of the comparison with surface in situ observations to the matchup rules. $n$: stations; $d$: days; "
           r"$\epsilon$ and $\beta$ in percent relative to the observation.")
    head = (r" & & & \multicolumn{2}{c}{PACE} & \multicolumn{2}{c}{OLCI} \\" + "\n" + r"Variable & $n$ & $d$ & $\epsilon$ & $\beta$ & $\epsilon$ & $\beta$")
    open(path, "w", encoding="utf-8").write(_wrap(lines, cap, "tab:insens", "lrrrrrr", head))
