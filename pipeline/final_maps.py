"""Map figures: one typical day (rows 1-2) plus all-day summaries (row 3), and a gallery of all days for the supplement."""
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figstyle import apply as _apply_figstyle
_apply_figstyle()
import numpy as np

from .stats import load_grids

MIN_DAYS = 5  # cells valid on fewer days are left blank in the all-day means


def pick_day(per_day, grids, min_cells=15000):
    """Day with the median chlorophyll correlation among days with more than min_cells matched cells."""
    q = per_day[per_day["var"] == "chlor_a"].dropna(subset=["r"])
    q = q[q.date.isin([d for d, g in grids.items() if int(np.isfinite(g["olci_chlor_a"].values).sum()) > min_cells])]
    med = q.r.median()
    return q.iloc[(q.r - med).abs().argsort()].iloc[0].date


def _style(a, g, left):
    a.contour(g.lon, g.lat, g.land.values.astype(float), [0.5], colors="k", linewidths=0.6)
    a.set_aspect(1 / np.cos(np.deg2rad(35.5))); a.grid(False)
    a.set_xticks([-123, -121, -119]); a.set_xticklabels(["123°W", "121°W", "119°W"], fontsize=10)
    a.set_yticks([34, 35, 36, 37])
    a.set_yticklabels([f"{x}°N" for x in (34, 35, 36, 37)] if left else [], fontsize=10)


def _tag(a, text):
    a.text(0.0, 1.04, text, transform=a.transAxes, fontweight="bold", fontsize=12, va="bottom", ha="left")


def _row(f, axs, g, v, name, log, lab, cm, r):
    p, o = g[f"pace_{v}"].values.astype(float), g[f"olci_{v}"].values.astype(float)
    if log:
        p, o = np.log10(p), np.log10(o)
    else:
        p, o = p * 1e3, o * 1e3
    both = np.r_[p[np.isfinite(p)], o[np.isfinite(o)]]
    vmin, vmax = np.percentile(both, [2, 98])
    for c, (a, t) in enumerate(((p, "PACE"), (o, "OLCI"))):
        im = axs[c].pcolormesh(g.lon, g.lat, a, vmin=vmin, vmax=vmax, cmap=cm, rasterized=True)
        axs[c].set_title(t, fontsize=10, pad=6, loc="right")
    f.colorbar(im, ax=axs[:2], orientation="horizontal", shrink=0.8, pad=0.02, aspect=30, label=lab, ticks=matplotlib.ticker.MaxNLocator(5))
    diff = o - p
    lim = np.nanpercentile(np.abs(diff), 95)
    im2 = axs[2].pcolormesh(g.lon, g.lat, diff, vmin=-lim, vmax=lim, cmap="RdBu_r", rasterized=True)
    axs[2].set_title("OLCI $-$ PACE", fontsize=10, pad=6, loc="right")
    f.colorbar(im2, ax=axs[2], orientation="horizontal", shrink=0.9, pad=0.02, aspect=15,
               label=("difference in log$_{10}$" if log else "difference (10$^{-3}$ sr$^{-1}$)"))
    axs[0].set_ylabel(name, fontsize=10)
    for c in range(3):
        _style(axs[c], g, c == 0)
        _tag(axs[c], "(" + "abcdefghi"[r * 3 + c] + ")")


CHL = ("chlor_a", "Chl-a", True, "log$_{10}$ Chl-a (mg m$^{-3}$)", "turbo")
RRS = ("Rrs_442.5", "$R_{\mathrm{rs}}$ 442.5 nm", False, "$R_{\mathrm{rs}}$ (10$^{-3}$ sr$^{-1}$)", "viridis")


def all_day_means(grids):
    """Number of days with both sensors valid, and mean OLCI - PACE for chlorophyll (log10) and Rrs 442.5 nm (1e-3 sr-1)."""
    keys = [d for d in sorted(grids) if np.isfinite(grids[d]["olci_chlor_a"].values).any()]
    shape = grids[keys[0]]["olci_chlor_a"].shape
    n = np.zeros(shape); s_chl = np.zeros(shape); s_rrs = np.zeros(shape)
    nr = np.zeros(shape)
    for d in keys:
        g = grids[d]
        pc, oc = g["pace_chlor_a"].values.astype(float), g["olci_chlor_a"].values.astype(float)
        k = np.isfinite(pc) & np.isfinite(oc) & (pc > 0) & (oc > 0)
        n += k; s_chl += np.where(k, np.log10(np.where(k, oc, 1)) - np.log10(np.where(k, pc, 1)), 0)
        pr, orr = g["pace_Rrs_442.5"].values.astype(float), g["olci_Rrs_442.5"].values.astype(float)
        kr = np.isfinite(pr) & np.isfinite(orr)
        nr += kr; s_rrs += np.where(kr, (orr - pr) * 1e3, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_chl = np.where(n >= MIN_DAYS, s_chl / n, np.nan)
        m_rrs = np.where(nr >= MIN_DAYS, s_rrs / nr, np.nan)
    land = grids[keys[0]].land.values.astype(bool)
    n = np.where(land, np.nan, n)
    return keys, n, m_chl, m_rrs


def fig_maps(cfg, per_day, fig):
    grids = load_grids(cfg)
    d = pick_day(per_day, grids)
    g = grids[d]
    f, axs = plt.subplots(3, 3, figsize=(7.2, 7.8), constrained_layout=True)
    for r, spec in enumerate((CHL, RRS)):
        _row(f, axs[r], g, *spec, r)
    # row 3: all days
    keys, n, m_chl, m_rrs = all_day_means(grids)
    im = axs[2, 0].pcolormesh(g.lon, g.lat, n, cmap="viridis", rasterized=True)
    axs[2, 0].set_title("Valid days", fontsize=10, pad=6, loc="right")
    f.colorbar(im, ax=axs[2, 0], orientation="horizontal", shrink=0.9, pad=0.02, aspect=15, label="number of days", ticks=matplotlib.ticker.MaxNLocator(4))
    for c, (m, lab, ttl) in enumerate(((m_chl, "difference in log$_{10}$", "Mean difference"), (m_rrs, "difference (10$^{-3}$ sr$^{-1}$)", "Mean difference"))):
        lim = np.nanpercentile(np.abs(m), 98)
        imm = axs[2, c + 1].pcolormesh(g.lon, g.lat, m, vmin=-lim, vmax=lim, cmap="RdBu_r", rasterized=True)
        axs[2, c + 1].set_title(ttl, fontsize=10, pad=6, loc="right")
        f.colorbar(imm, ax=axs[2, c + 1], orientation="horizontal", shrink=0.9, pad=0.02, aspect=15, label=lab, ticks=matplotlib.ticker.MaxNLocator(4))
    axs[2, 0].set_ylabel("All days", fontsize=10)
    for c in range(3):
        _style(axs[2, c], g, c == 0)
        _tag(axs[2, c], "(" + "ghi"[c] + ")")
    f.savefig(fig / "fig3_maps.png", dpi=300); plt.close(f)
    return d, len(keys)


def supp_maps(cfg, days, fig):
    """Chlorophyll maps for the given days (one row each)."""
    grids = load_grids(cfg)
    f, axs = plt.subplots(len(days), 3, figsize=(7.2, 2.9 * len(days)), constrained_layout=True)
    for r, d in enumerate(days):
        _row(f, axs[r], grids[d], *CHL, r)
        axs[r, 0].set_ylabel(f"{d[:4]}-{d[4:6]}-{d[6:]}", fontsize=10)
        axs[r, 0].yaxis.set_label_position("left")
    f.savefig(fig / "figS2_maps.png", dpi=300); plt.close(f)


def supp_gallery(cfg, fig, ncol=8):
    """OLCI - PACE at 442.5 nm for every analysed day."""
    grids = load_grids(cfg)
    keys = [d for d in sorted(grids) if np.isfinite(grids[d]["olci_Rrs_442.5"].values).any()]
    nrow = int(np.ceil(len(keys) / ncol))
    f, axs = plt.subplots(nrow, ncol, figsize=(7.2, 1.05 * nrow + 0.9), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for a in axs.ravel():
        a.axis("off")
    for a, d in zip(axs.ravel(), keys):
        g = grids[d]
        diff = (g["olci_Rrs_442.5"].values.astype(float) - g["pace_Rrs_442.5"].values.astype(float)) * 1e3
        im = a.pcolormesh(g.lon, g.lat, diff, vmin=-2, vmax=2, cmap="RdBu_r", rasterized=True)
        a.set_aspect(1 / np.cos(np.deg2rad(35.5)))
        a.set_title(f"{d[:4]}-{d[4:6]}-{d[6:]}", fontsize=6.5, pad=2)
    f.colorbar(im, ax=axs, orientation="horizontal", shrink=0.5, aspect=40, pad=0.02, label="OLCI $-$ PACE, $R_{\mathrm{rs}}$ 442.5 nm (10$^{-3}$ sr$^{-1}$)", extend="both")
    f.savefig(fig / "figS7_gallery.png", dpi=300); plt.close(f)
    return len(keys)


def supp_chl_gallery(cfg, fig, ncol=4):
    """Chlorophyll-a from PACE and OLCI side by side for every analysed day."""
    grids = load_grids(cfg)
    keys = [d for d in sorted(grids) if np.isfinite(grids[d]["olci_chlor_a"].values).any()]
    nrow = int(np.ceil(len(keys) / ncol))
    f, axs = plt.subplots(nrow, 2 * ncol, figsize=(7.2, 0.95 * nrow + 0.9), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for a in axs.ravel():
        a.axis("off")
    for i, d in enumerate(keys):
        g = grids[d]; r, c = divmod(i, ncol)
        for j, (sen, name) in enumerate((("pace", "PACE"), ("olci", "OLCI"))):
            a = axs[r, 2 * c + j]
            im = a.pcolormesh(g.lon, g.lat, np.log10(g[f"{sen}_chlor_a"].values.astype(float)), vmin=-1, vmax=0.8, cmap="turbo", rasterized=True)
            a.set_aspect(1 / np.cos(np.deg2rad(35.5)))
            a.set_title(f"{d[:4]}-{d[4:6]}-{d[6:]} {name}" if j == 0 else name, fontsize=6, pad=1.5, loc="left" if j == 0 else "center")
    f.colorbar(im, ax=axs, orientation="horizontal", shrink=0.5, aspect=40, pad=0.02, label="log$_{10}$ Chl-a (mg m$^{-3}$)", extend="both")
    f.savefig(fig / "figS8_chl_gallery.png", dpi=300); plt.close(f)
