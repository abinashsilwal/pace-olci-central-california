"""Main-text and supplementary figures (see docs/analysis-protocol.md for what each answers)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figstyle import apply as _apply_figstyle
_apply_figstyle()
import numpy as np
import pandas as pd

from .final import rma, ALL_BANDS, C_OLCI, C_PACE, PRIMARY, rb

PANEL = "abcdefghij"


def _tag(ax, i):
    ax.text(0.0, 1.04, f"({PANEL[i]})", transform=ax.transAxes, fontweight="bold", fontsize=12, ha="left", va="bottom")


def fig_agreement(m, per_day, fig):
    """Fig 3: density scatter for the four bands and chl-a, plus day-by-day bias."""
    vars_ = [f"Rrs_{w:g}" for w in PRIMARY] + ["chlor_a"]
    f, axs = plt.subplots(2, 3, figsize=(7.2, 5.6), constrained_layout=True)
    axs = axs.ravel()
    for i, v in enumerate(vars_):
        ax = axs[i]; g = m[m["var"] == v]
        log = v == "chlor_a"
        if len(g) > 300000:
            g = g.sample(300000, random_state=1)
        if not log:
            g = g.assign(pace=g.pace * 1e3, olci=g.olci * 1e3)
        ax.hexbin(g.pace, g.olci, gridsize=45, bins="log", mincnt=1, cmap="viridis", xscale="log" if log else "linear",
                  yscale="log" if log else "linear", linewidths=0)
        lim = [g[["pace", "olci"]].min().min(), g[["pace", "olci"]].max().max()]
        ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
        mv = m[m["var"] == v]
        xs = np.linspace(lim[0], lim[1], 50)
        if log:
            sl, ic = rma(np.log10(mv.pace), np.log10(mv.olci)); ax.plot(xs, 10 ** (ic + sl * np.log10(xs)), color="#c0392b", lw=1.2, label="RMA")
        else:
            sl, ic = rma(mv.pace * 1e3, mv.olci * 1e3); ax.plot(xs, ic + sl * xs, color="#c0392b", lw=1.2, label="RMA")
        q = rb(m[m["var"] == v].pace, m[m["var"] == v].olci, v)
        name = "Chl-a" if log else f"$R_{{\mathrm{{rs}}}}$ {v.split('_')[1]} nm"
        ax.set_title(name, fontsize=10.5)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.text(0.96, 0.04, f"r = {q['r']:.2f}\n$\\epsilon$ = {q['eps']:.0f}%\n$\\beta$ = {q['beta']:+.0f}%", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.9, pad=2))
        ax.set_xlabel("PACE (mg m$^{-3}$)" if log else "PACE ($10^{-3}$ sr$^{-1}$)"); ax.set_ylabel("OLCI (mg m$^{-3}$)" if log else "OLCI ($10^{-3}$ sr$^{-1}$)"); _tag(ax, i)
    ax = axs[5]
    cols = ["#4c78a8", "#59a14f", "#edc948", "#e15759", "#b07aa1"]
    for k, v in enumerate(vars_):
        g = per_day[per_day["var"] == v]
        ax.scatter(np.full(len(g), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(g)), g.beta, s=7, color=cols[k], alpha=0.8, lw=0)
        ax.hlines(g.beta.median(), k - 0.3, k + 0.3, color="k", lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(["442.5", "490", "560", "665", "Chl-a"], rotation=40, ha="right")
    ax.set_ylabel("Daily $\\beta$ (%)"); ax.set_xlabel("Band (nm)"); _tag(ax, 5)
    ax.set_title("Daily bias", fontsize=10.5)
    f.savefig(fig / "fig4_agreement.png", dpi=300); plt.close(f)


def fig_spectral(spec_all, spec_zone, spec_chl, fig):
    """Fig 4: spectra, relative offset by zone, absolute offset by chlorophyll class."""
    plot = spec_all[spec_all.wl < 700]
    f = plt.figure(figsize=(7.2, 5.8), constrained_layout=True); gs = f.add_gridspec(2, 2, height_ratios=[1, 0.75]); axs = [f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]), f.add_subplot(gs[1, :])]
    ax = axs[0]
    ax.plot(plot.wl, plot.med_pace * 1e3, "o-", color=C_PACE, ms=3, lw=1, label="PACE")
    ax.plot(plot.wl, (plot.med_pace + plot.med_diff) * 1e3, "s--", color=C_OLCI, ms=3, lw=1, label="OLCI")
    ax.set_ylabel("Median $R_{\mathrm{rs}}$ ($10^{-3}$ sr$^{-1}$)"); ax.set_xlabel("Wavelength (nm)"); ax.legend(loc="upper right"); _tag(ax, 0)
    ax = axs[1]
    for z, c in (("2-10 km", "#d62728"), ("10-50 km", "#ff7f0e"), (">50 km", "#1f77b4")):
        q = spec_zone[(spec_zone.stratum == z) & (spec_zone.wl < 700)]
        ax.plot(q.wl, q.beta, "o-", ms=3, lw=1, color=c, label=z)
    ax.fill_between(plot.wl, plot.beta_lo, plot.beta_hi, color="k", alpha=0.12, lw=0)
    ax.plot(plot.wl, plot.beta, "k-", lw=1.4, label="all")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("$\\beta$: OLCI relative to PACE (%)"); ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim(-25, 68); ax.legend(title="Distance from the coast", ncol=2, fontsize=9, title_fontsize=9, loc="upper right"); _tag(ax, 1)
    ax = axs[2]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
    for c, cl in zip(cols, ["<0.2", "0.2-0.5", "0.5-1", "1-3", ">3"]):
        q = spec_chl[(spec_chl.stratum == cl) & (spec_chl.wl < 700)]
        ax.plot(q.wl, q.med_diff * 1e4, "o-", ms=3, lw=1, color=c, label=cl)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Median OLCI $-$ PACE ($10^{-4}$ sr$^{-1}$)"); ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim(-5.5, 8.5); ax.legend(title="Chl-a (mg m$^{-3}$)", ncol=3, fontsize=9, title_fontsize=9, loc="upper right"); _tag(ax, 2)
    f.savefig(fig / "fig5_spectrum.png", dpi=300); plt.close(f)


def fig_geometry(vza_bins, within, per_day_df, fig):
    """Fig 5: dependence on PACE view zenith angle."""
    f = plt.figure(figsize=(7.2, 5.8), constrained_layout=True); gs = f.add_gridspec(2, 2, height_ratios=[1, 0.75]); axs = [f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]), f.add_subplot(gs[1, :])]
    ax = axs[0]
    cols = {"Rrs_442.5": "#4c78a8", "Rrs_490": "#59a14f", "Rrs_560": "#edc948", "Rrs_665": "#e15759", "chlor_a": "#b07aa1"}
    names = {"Rrs_442.5": "442.5", "Rrs_490": "490", "Rrs_560": "560", "Rrs_665": "665", "chlor_a": "Chl-a"}
    edges = [0, 45, 52, 58, 65, 90]
    mids = {f"({a}, {b}]": (a + min(b, 70)) / 2 if b < 90 else 67.5 for a, b in zip(edges[:-1], edges[1:])}
    for v, c in cols.items():
        q = vza_bins[vza_bins["var"] == v].copy()
        q["x"] = q.stratum.map(mids)
        q = q.sort_values("x")
        ax.errorbar(q.x, q.beta, yerr=[q.beta - q.beta_lo, q.beta_hi - q.beta], fmt="o-", ms=3, lw=1, capsize=1.5, color=c, label=names[v])
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("PACE view zenith angle ($^\\circ$)"); ax.set_ylabel("$\\beta$ (%)")
    ax.legend(ncol=3, fontsize=8.5, loc="upper left"); ax.set_ylim(top=250); _tag(ax, 0)
    ax = axs[1]
    g = per_day_df[per_day_df["var"] == "Rrs_442.5"]
    ax.scatter(g.vza, g.beta, s=14, color="#4c78a8", lw=0)
    ax.set_xlabel("Day-median PACE view zenith ($^\\circ$)"); ax.set_ylabel("Daily $\\beta$ at 442.5 nm (%)"); ax.axhline(0, color="k", lw=0.6); _tag(ax, 1)
    ax = axs[2]
    order = ["Rrs_442.5", "Rrs_490", "Rrs_560", "Rrs_665", "chlor_a"]
    w = within.set_index("var").loc[order]
    ax.errorbar(range(5), w.mean_d_beta, yerr=[w.mean_d_beta - w.lo, w.hi - w.mean_d_beta], fmt="o", color="k", capsize=2, ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([names[v] for v in order]); ax.set_xlabel("Band (nm)")
    ax.set_ylabel("$\\beta$(high VZA) $-$ $\\beta$(low VZA) (pp)"); _tag(ax, 2)
    f.savefig(fig / "figS5_geometry.png", dpi=300); plt.close(f)


def fig_insitu(tab, fig):
    """Fig 6: bias of each sensor against in situ observations for all stations and by distance from the coast."""
    panels = ["Rrs_442.5", "Rrs_490", "Rrs_560", "chlor_a"]
    ttl = {"Rrs_442.5": "$R_{\mathrm{rs}}$ 442.5 nm", "Rrs_490": "$R_{\mathrm{rs}}$ 490 nm", "Rrs_560": "$R_{\mathrm{rs}}$ 560 nm", "chlor_a": "Chl-a (HPLC/bottle)"}
    f, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True, sharey=False); axs = axs.ravel()
    cats = ["all", "<10 km", "10-30 km"]
    for ax, v, i in zip(axs, panels, range(4)):
        q = tab[tab["var"] == v]
        for j, (sen, c, off) in enumerate((("pace", C_PACE, -0.12), ("olci", C_OLCI, 0.12))):
            r = q[q.sensor == sen].set_index("subset").reindex(cats)
            x = np.arange(len(cats)) + off
            ok = r.beta.notna()
            lo = (r.beta - r.beta_lo).where(ok, 0).fillna(0); hi = (r.beta_hi - r.beta).where(ok, 0).fillna(0)
            ax.errorbar(x[ok.values], r.beta[ok], yerr=[lo[ok], hi[ok]], fmt="o", color=c, ms=4, capsize=2, lw=1, label=sen.upper() if sen == "olci" else "PACE")
        labs = []
        for k, cat in enumerate(cats):
            n = q[(q.sensor == "pace") & (q.subset == cat)]
            labs.append(cat.replace(" km", "") + ("\n(n=%d)" % int(n.n.iloc[0]) if len(n) else ""))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(labs); ax.set_xlabel("Distance from coast (km)")
        ax.set_title(ttl[v], fontsize=10.5, pad=6); _tag(ax, i)
        if i % 2 == 0:
            ax.set_ylabel("$\\beta$ vs in situ (%)")
    h, l = axs[1].get_legend_handles_labels(); f.legend(h, l, loc="outside lower center", ncol=2, fontsize=10, frameon=False)
    f.savefig(fig / "fig7_insitu_bias.png", dpi=300); plt.close(f)


def fig_insitu_scatter(pairs, fig):
    """Fig 7: satellite against in situ values at the matched stations."""
    panels = [("Rrs_442.5", "$R_{\mathrm{rs}}$ 442.5 nm", False), ("Rrs_490", "$R_{\mathrm{rs}}$ 490 nm", False), ("Rrs_560", "$R_{\mathrm{rs}}$ 560 nm", False), ("chlor_a", "Chl-a", True)]
    f, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True); axs = axs.ravel()
    for i, (ax, (v, ttl, log)) in enumerate(zip(axs, panels)):
        q = pairs[(pairs["var"] == v) & (pairs.group == ("HPLC/bottle" if log else "surface"))]
        q = q.dropna(subset=["insitu", "pace", "olci"])
        q = q[(q.insitu > 0) & (q.pace > 0) & (q.olci > 0)]
        k = 1.0 if log else 1e3
        x = q.insitu * k
        ax.scatter(x, q.pace * k, s=14, color=C_PACE, alpha=0.75, lw=0, label="PACE")
        ax.scatter(x, q.olci * k, s=14, color=C_OLCI, alpha=0.75, lw=0, label="OLCI")
        lo = min(x.min(), (q.pace * k).min(), (q.olci * k).min()); hi = max(x.max(), (q.pace * k).max(), (q.olci * k).max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("In situ (mg m$^{-3}$)" if log else "In situ ($10^{-3}$ sr$^{-1}$)")
        ax.set_ylabel("Satellite (mg m$^{-3}$)" if log else "Satellite ($10^{-3}$ sr$^{-1}$)")
        ax.set_title(ttl, fontsize=10.5, pad=6); _tag(ax, i)
        ax.text(0.96, 0.04, f"n = {x.size}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    h, l = axs[0].get_legend_handles_labels(); f.legend(h, l, loc="outside lower center", ncol=2, fontsize=10, frameon=False, markerscale=1.6)
    f.savefig(fig / "fig6_insitu_scatter.png", dpi=300); plt.close(f)
