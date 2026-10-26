"""Supplementary figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figstyle import apply as _apply_figstyle
_apply_figstyle()
import numpy as np
import pandas as pd

from .final import C_OLCI, C_PACE
from .stats import load_grids

NAMES = {"Rrs_442.5": "442.5", "Rrs_490": "490", "Rrs_560": "560", "Rrs_665": "665", "chlor_a": "Chl-a"}


def s_perday(per_day, fig):
    f, axs = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True, sharex=True)
    d = sorted(per_day.date.unique())
    for v, c in zip(NAMES, ["#4c78a8", "#59a14f", "#edc948", "#e15759", "#b07aa1"]):
        q = per_day[per_day["var"] == v].set_index("date").reindex(d)
        axs[0].plot(range(len(d)), q.r, "o-", ms=3, lw=0.9, color=c, label=NAMES[v])
        axs[1].plot(range(len(d)), q.beta, "o-", ms=3, lw=0.9, color=c)
    for a, y in zip(axs, ("$r$ (log for chl-a)", "$\\beta$ (%)")):
        a.set_ylabel(y)
    lab = [x[:4] + "-" + x[4:6] + "-" + x[6:] if "-" not in x else x for x in d]
    axs[1].set_xticks(range(0, len(d), 2)); axs[1].set_xticklabels(lab[::2], rotation=60, ha="right", fontsize=8)
    axs[1].axhline(0, color="k", lw=0.6)
    axs[0].legend(ncol=5, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, title="Band (nm) / variable", title_fontsize=9)
    f.savefig(fig / "figS1_perday.png", dpi=300); plt.close(f)


def s_maps(cfg, per_day, fig):
    from .final_maps import supp_maps
    pdc = per_day[per_day["var"] == "chlor_a"].dropna(subset=["r"])
    best, worst = pdc.sort_values("r").iloc[-1].date, pdc.sort_values("r").iloc[0].date
    supp_maps(cfg, [best, worst], fig)
    return best, worst


def s_coverage(cov, fig):
    f, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    lab = [f"{int(a)}-{int(b)}" if b < 1e8 else f">{int(a)}" for a, b in zip(cov.d_from, cov.d_to)]
    ax.bar(range(len(cov)), cov.pace_missing_pct, color=C_PACE)
    ax.set_xticks(range(len(cov))); ax.set_xticklabels(lab); ax.set_xlabel("Distance from the coast (km)")
    ax.set_ylabel("OLCI-valid cells without PACE (%)")
    f.savefig(fig / "figS3_coverage.png", dpi=300); plt.close(f)


def s_sza(sza, fig):
    f, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    for v, c in zip(NAMES, ["#4c78a8", "#59a14f", "#edc948", "#e15759", "#b07aa1"]):
        q = sza[sza["var"] == v]
        ax.plot(range(len(q)), q.beta, "o-", ms=3, lw=1, color=c, label=NAMES[v])
    ax.set_xticks(range(len(q))); ax.set_xticklabels(q.stratum, fontsize=8.5); ax.set_xlabel("OLCI solar zenith angle ($^\\circ$)")
    ax.set_ylabel("$\\beta$ (%)"); ax.axhline(0, color="k", lw=0.6); ax.legend(ncol=3, fontsize=8.5)
    f.savefig(fig / "figS6_sza.png", dpi=300); plt.close(f)


def s_kind(df, fig):
    k = df[df.subset.str.startswith("kind:")]
    f, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), constrained_layout=True)
    for ax, v in zip(axs, ["Rrs_442.5", "Rrs_490", "Rrs_560"]):
        q = k[k["var"] == v]
        kinds = list(q.subset.unique())
        for off, sen, c in ((-0.15, "pace", C_PACE), (0.15, "olci", C_OLCI)):
            r = q[q.sensor == sen].set_index("subset").reindex(kinds)
            ax.errorbar(np.arange(len(kinds)) + off, r.beta, yerr=[(r.beta - r.beta_lo).fillna(0), (r.beta_hi - r.beta).fillna(0)], fmt="o", color=c, ms=3.5, capsize=2, lw=0.8, label=sen.upper())
        ax.set_xticks(range(len(kinds))); ax.set_xticklabels([x.replace("kind:", "").replace("_Rrs", "").replace("Rrs_station", "PACE-PAX station").replace("PACE-PAX(above_water)", "PACE-PAX above-water").replace("PACE-PAX(station)", "PACE-PAX station") for x in kinds], rotation=40, ha="right", fontsize=8.5)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"$R_{{\mathrm{{rs}}}}$ {NAMES[v]} nm", fontsize=10.5)
    axs[0].set_ylabel("$\\beta$ vs in situ (%)"); axs[0].legend(fontsize=8.5)
    f.savefig(fig / "figS4_insitu_kind.png", dpi=300); plt.close(f)
