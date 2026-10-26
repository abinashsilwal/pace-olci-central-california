"""Step 5: scatter panels, side-by-side maps, transects."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .stats import load_grids, metrics, variables

PRETTY = {"chlor_a": "Chl-a (mg m$^{-3}$)"}


def _label(v):
    return PRETTY.get(v, v.replace("Rrs_", "Rrs ") + " nm (sr$^{-1}$)")


def scatter_panels(m, cfg, fname="fig_scatter.png"):
    vs = variables(cfg)
    fig, axs = plt.subplots(1, len(vs), figsize=(4 * len(vs), 4))
    for ax, v in zip(np.atleast_1d(axs), vs):
        g = m[m["var"] == v]
        log = v == "chlor_a" and cfg["stats"]["chl_log10"]
        ax.hexbin(g.pace, g.olci, gridsize=50, bins="log", mincnt=1, cmap="viridis",
                  xscale="log" if log else "linear", yscale="log" if log else "linear")
        lim = [min(g.pace.min(), g.olci.min()), max(g.pace.max(), g.olci.max())]
        ax.plot(lim, lim, "k--", lw=1)
        s = metrics(g.pace, g.olci, log)
        ax.set_title(f"{_label(v)}\nr={s.get('r', np.nan):.2f}  RMSE={s.get('RMSE', np.nan):.3g}  "
                     f"bias={s.get('bias', np.nan):.2g}  n={s['n']}", fontsize=8)
        ax.set_xlabel("PACE OCI"); ax.set_ylabel("Sentinel-3 OLCI")
    fig.tight_layout()
    fig.savefig(cfg["paths"]["out"] / fname, dpi=300)
    plt.close(fig)


def maps(cfg, var="chlor_a"):
    """Per-date PACE | OLCI | difference maps."""
    for date, g in load_grids(cfg).items():
        p, o = g[f"pace_{var}"], g[f"olci_{var}"]
        if int(np.isfinite(p.values + o.values).sum()) < cfg["qc"]["min_common_cells"]:
            continue
        log = var == "chlor_a"
        t = (lambda a: np.log10(a)) if log else (lambda a: a)
        fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
        vmin, vmax = np.nanpercentile(t(np.r_[p.values.ravel(), o.values.ravel()]), [2, 98])
        for ax, a, ttl in zip(axs[:2], (p, o), ("PACE OCI", "Sentinel-3 OLCI")):
            im = ax.pcolormesh(g.lon, g.lat, t(a), vmin=vmin, vmax=vmax, cmap="turbo")
            ax.set_title(f"{ttl}  {date}")
        fig.colorbar(im, ax=axs[:2], label=("log10 " if log else "") + _label(var), shrink=0.8)
        d = t(o) - t(p)
        lim = np.nanpercentile(np.abs(d.values), 95)
        im2 = axs[2].pcolormesh(g.lon, g.lat, d, vmin=-lim, vmax=lim, cmap="RdBu_r")
        axs[2].set_title("OLCI - PACE")
        fig.colorbar(im2, ax=axs[2], shrink=0.8)
        for ax in axs:
            ax.contour(g.lon, g.lat, g.land.values.astype(float), [0.5], colors="k", linewidths=0.6)
            ax.set_aspect(1 / np.cos(np.deg2rad(g.lat.values.mean())))
        fig.savefig(cfg["paths"]["out"] / f"map_{var}_{date}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def transects(cfg, var="chlor_a", npts=60):
    grids = load_grids(cfg)
    for tr in cfg["transects"]:
        lons = np.linspace(tr["start"][0], tr["end"][0], npts)
        lats = np.linspace(tr["start"][1], tr["end"][1], npts)
        km = np.hypot((lons - lons[0]) * 111 * np.cos(np.deg2rad(lats[0])), (lats - lats[0]) * 111)
        fig, ax = plt.subplots(figsize=(7, 4))
        for date, g in grids.items():
            iy = (lats - g.lat.values[0]) / (g.lat.values[1] - g.lat.values[0])
            ix = (lons - g.lon.values[0]) / (g.lon.values[1] - g.lon.values[0])
            for s, c in (("pace", "tab:blue"), ("olci", "tab:red")):
                a = g[f"{s}_{var}"].values
                z = ndimage.map_coordinates(np.nan_to_num(a, nan=-999), [iy, ix], order=0)
                z = np.where(z == -999, np.nan, z)
                if np.isfinite(z).sum() > npts // 2:
                    ax.plot(km, z, color=c, alpha=0.35, lw=1)
        ax.plot([], [], "tab:blue", label="PACE"); ax.plot([], [], "tab:red", label="OLCI")
        if var == "chlor_a":
            ax.set_yscale("log")
        ax.set_xlabel("distance offshore (km)"); ax.set_ylabel(_label(var))
        ax.set_title(f"Transect: {tr['name']}"); ax.legend()
        fig.tight_layout()
        fig.savefig(cfg["paths"]["out"] / f"transect_{tr['name']}_{var}.png", dpi=200)
        plt.close(fig)


def run_figures(cfg, m=None):
    from .stats import build_matchups
    m = m if m is not None else build_matchups(cfg)
    scatter_panels(m, cfg)
    maps(cfg)
    transects(cfg)
    print("figures ->", cfg["paths"]["out"])
