"""Viewing / solar geometry and OLCI spectral response functions.

PACE L2 files carry no per-pixel angles, so the view zenith angle (VZA) is computed from the
cross-track ground distance to the scan-line centre (nadir) for a spherical Earth and the PACE
altitude. The along-track tilt of OCI (+/-20 deg) is ignored, so the VZA is a cross-track value.
Solar zenith angle (SZA) uses the NOAA low-precision solar position formulas.
"""
import glob
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np

R_EARTH = 6371.0
H_PACE = 676.5   # km

# OLCI band number -> nominal centre (nm)
OLCI_BANDS = {1: 400.0, 2: 412.5, 3: 442.5, 4: 490.0, 5: 510.0, 6: 560.0, 7: 620.0, 8: 665.0,
              9: 673.75, 10: 681.25, 11: 708.75}
BAND_OF = {v: k for k, v in OLCI_BANDS.items()}


def pace_vza(lat, lon):
    """Cross-track view zenith angle (deg) for every pixel of a PACE swath array (lines x pixels)."""
    ic = lat.shape[1] // 2
    la0 = np.deg2rad(lat[:, ic])[:, None]; lo0 = np.deg2rad(lon[:, ic])[:, None]
    la = np.deg2rad(lat); lo = np.deg2rad(lon)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    phi = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))                       # central angle (rad)
    alpha = np.arctan2(R_EARTH * np.sin(phi), R_EARTH + H_PACE - R_EARTH * np.cos(phi))
    return np.degrees(alpha + phi)


def granule_time(path, minutes_offset=0.0):
    m = re.search(r"\.(\d{8})T(\d{6})", os.path.basename(path))
    t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return t + timedelta(minutes=minutes_offset)


def solar_zenith(lat, lon, when):
    """Solar zenith angle (deg); lat/lon in degrees (east positive), `when` a UTC datetime."""
    doy = when.timetuple().tm_yday
    frac_h = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (frac_h - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = frac_h * 60 + eqt + 4 * lon                                      # true solar time (min)
    ha = np.deg2rad(tst / 4 - 180)
    la = np.deg2rad(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cosz, -1, 1)))


_SRF = {}


def load_srf(root="data/srf"):
    """Band-mean OLCI relative spectral response (average over camera modules and CCD columns, S3A and S3B).
    Returns {band_number: (wavelength_nm, response)} on a 0.1 nm grid, normalised to unit area."""
    if _SRF:
        return _SRF
    acc = {}
    for sat in ("a", "b"):
        for f in glob.glob(f"{root}/{sat}/**/*.txt", recursive=True):
            head = open(f).readline()
            m = re.search(r"band (\d+) module", head)
            if not m:
                continue
            b = int(m.group(1))
            rows = [l.split() for l in open(f).read().split("Filter response")[1].strip().splitlines()]
            wn = np.array([float(r[0]) for r in rows]); resp = np.array([float(r[1]) for r in rows])
            wl = 1e7 / wn
            o = np.argsort(wl)
            acc.setdefault(b, []).append((wl[o], resp[o]))
    for b, lst in acc.items():
        grid = np.arange(min(w[0] for w, _ in lst), max(w[-1] for w, _ in lst), 0.1)
        s = np.mean([np.interp(grid, w, r, left=0, right=0) for w, r in lst], axis=0)
        keep = s > 1e-4 * s.max()
        grid, s = grid[keep], s[keep]
        _SRF[b] = (grid, s / np.trapezoid(s, grid))
    return _SRF


def srf_centroid_fwhm(b):
    w, s = load_srf()[b]
    c = np.trapezoid(w * s, w)
    half = s >= s.max() / 2
    return float(c), float(w[half][-1] - w[half][0])
