"""Extract the 54-vane exhaust PWL spectrum from Fernandes Fig. 18(c).

Fernandes.pdf page 24 (report p. 24) is vector.  Panel (c) is the exhaust-
contribution PWL spectrum.  The experimental curve is the thick black solid
path; the three ZDES curves are thin black dashed, grey dashed and red solid.
The extraction uses the path geometry only, not raster interpretation.

Units: the Fernandes y axis is PWL [dB/Hz], whereas the NASA TM-2002 figure
is labelled "Exhaust PWL Spectrum, dB".  The two are therefore compared by
shape, not by absolute level; see the provenance and QA notes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pymupdf as fitz

from common import (DIG, OVER, PROV, QA, ROOT, render_page, write_csv,
                    write_json, sha256)

PDF = ROOT / "Fernandes.pdf"
PAGE = 24
BPF_HZ = 7808.0 * 22.0 / 60.0

# Panel (c) frame in PDF points, from the vector frame paths.
FRAME = (327.9, 319.5, 526.5, 445.5)
X_LIMITS = (0.3, 10.0)   # BPF multiples at the left/right frame
Y_LIMITS = (100.0, 55.0)  # dB/Hz at the top/bottom frame

SIGMA_DB = 0.03
FREQ_REL_SIGMA = 0.001


def panel_frame_and_ticks(page) -> dict:
    """Detect the panel frame and the 5 dB horizontal gridlines from vectors."""
    drawings = page.get_drawings()
    # frame: long thin horizontal/vertical black strokes at the panel boundary
    x0, y0, x1, y1 = FRAME
    vlines, hlines = [], []
    for dd in drawings:
        r = dd["rect"]
        if dd["type"] != "s":
            continue
        # Only accept strokes that span this panel: neighbouring panel (b)
        # has gridlines at almost the same y, so a y-only filter mixes them in.
        if (abs(r.x1 - r.x0) < 0.2 and abs(r.y1 - r.y0) > 100
                and abs(r.y0 - y0) < 1.0 and abs(r.y1 - y1) < 1.0):
            vlines.append(round(r.x0, 2))
        if (abs(r.y1 - r.y0) < 0.2 and abs(r.x1 - r.x0) > 100
                and x0 - 1 <= r.x0 and r.x1 <= x1 + 1):
            hlines.append(round(r.y0, 2))
    vlines = sorted(set(vlines))
    hlines = sorted(set(hlines))
    if not any(abs(x - x0) < 1.0 for x in vlines) or not any(abs(x - x1) < 1.0 for x in vlines):
        raise RuntimeError(f"panel c vertical frame not found: {vlines}")
    if not any(abs(y - y0) < 1.0 for y in hlines) or not any(abs(y - y1) < 1.0 for y in hlines):
        raise RuntimeError(f"panel c horizontal frame not found: {hlines}")

    # horizontal gridlines inside the frame, every 5 dB
    grid = sorted(y for y in hlines if y0 - 1 <= y <= y1 + 1)
    if len(grid) != 10:
        raise RuntimeError(f"expected 10 horizontal gridlines, got {grid}")
    yvals = np.linspace(Y_LIMITS[0], Y_LIMITS[1], len(grid))
    ycoef = np.polyfit(grid, yvals, 1)
    yres = np.polyval(ycoef, grid) - yvals

    # vertical gridlines every 0.1 BPF below 1, then integer BPF above 1
    vgrid = sorted(x for x in vlines if x0 - 1 <= x <= x1 + 1)
    if len(vgrid) != 17:
        raise RuntimeError(f"expected 17 vertical gridlines, got {vgrid}")
    # log positions of the known tick sequence
    tick_bpf = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                2, 3, 4, 5, 6, 7, 8, 9, 10]
    xcoef = np.polyfit(vgrid, np.log10(tick_bpf), 1)
    xres = np.polyval(xcoef, vgrid) - np.log10(tick_bpf)
    return {
        "frame_pt": [x0, y0, x1, y1],
        "x_gridlines_pt": vgrid,
        "x_tick_bpf": tick_bpf,
        "x_fit": {
            "slope_decades_per_pt": float(xcoef[0]),
            "intercept_decades": float(xcoef[1]),
            "max_abs_residual_decades": float(np.max(np.abs(xres))),
            "residual_decades": xres.tolist(),
        },
        "y_gridlines_pt": grid,
        "y_tick_db_per_hz": yvals.tolist(),
        "y_fit": {
            "slope_db_per_pt": float(ycoef[0]),
            "intercept_db": float(ycoef[1]),
            "max_abs_residual_db": float(np.max(np.abs(yres))),
            "residual_db": yres.tolist(),
        },
    }


def is_dashed(dd: dict) -> bool:
    """PyMuPDF returns dashes as a string such as '[] 0' or '[ 3 1 ] 0'."""
    spec = (dd.get("dashes") or "").strip()
    if spec.startswith("["):
        spec = spec[1:].split("]")[0]
    return bool(spec.strip())


def find_experimental_curve(page) -> tuple[int, dict]:
    """Pick the thick black solid path that spans the panel."""
    x0, y0, x1, y1 = FRAME
    candidates = []
    for i, dd in enumerate(page.get_drawings()):
        r = dd["rect"]
        if dd["type"] != "s":
            continue
        color = dd.get("color")
        if color is None or max(color) > 0.01:  # black only
            continue
        if dd.get("width", 0) < 0.6:
            continue
        if is_dashed(dd):
            continue
        if r.width < 150 or r.height < 20:
            continue
        if r.x0 > x0 + 5 or r.x1 < x1 - 5:
            continue
        npts = sum(len(it) for it in dd["items"])
        candidates.append((npts, i, dd))
    if not candidates:
        raise RuntimeError("no thick black solid curve found in panel c")
    npts, idx, dd = max(candidates, key=lambda q: q[0])
    if len(candidates) > 1:
        print(f"warning: {len(candidates)} thick black candidates; chose drawing {idx} with {npts} points")
    return idx, dd


def curve_points(dd: dict) -> np.ndarray:
    pts = []
    for it in dd["items"]:
        if it[0] == "l":
            pts.extend([it[1], it[2]])
    a = np.array([[q.x, q.y] for q in pts], dtype=float)
    # remove consecutive duplicates
    keep = [0]
    for i in range(1, len(a)):
        if abs(a[i, 0] - a[keep[-1], 0]) > 1e-9 or abs(a[i, 1] - a[keep[-1], 1]) > 1e-9:
            keep.append(i)
    return a[keep]


def make_overlay(gray: np.ndarray, pts: np.ndarray, title: str, outpath) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, x1, y1 = FRAME
    # 400 dpi: 1 pt = 400/72 px
    s = 400.0 / 72.0
    L, T, R, B = [int(round(v * s)) for v in FRAME]
    m = 20
    crop = gray[max(0, T - m):B + m, max(0, L - m):R + m]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.plot(pts[:, 0] * s - (L - m), pts[:, 1] * s - (T - m),
            ".", ms=2, color="red", alpha=0.8, label="extracted experimental path")
    ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> dict:
    doc = fitz.open(PDF)
    page = doc[PAGE - 1]
    cal = panel_frame_and_ticks(page)
    idx, dd = find_experimental_curve(page)
    a = curve_points(dd)
    x0, y0, x1, y1 = FRAME
    m = (a[:, 0] >= x0 - 0.05) & (a[:, 0] <= x1 + 0.05) & (a[:, 1] >= y0 - 0.05) & (a[:, 1] <= y1 + 0.05)
    a = a[m]
    if len(a) < 100:
        raise RuntimeError(f"too few curve points inside frame: {len(a)}")

    xf, yf = cal["x_fit"], cal["y_fit"]
    bpf = 10.0 ** (xf["slope_decades_per_pt"] * a[:, 0] + xf["intercept_decades"])
    pwl = yf["slope_db_per_pt"] * a[:, 1] + yf["intercept_db"]

    # duplicate x positions -> mean y
    ux = np.unique(np.round(a[:, 0], 3))
    vals = []
    for x in ux:
        sel = np.abs(a[:, 0] - x) < 0.02
        vals.append((x, float(np.mean(a[sel, 1]))))
    vals = np.array(vals)
    bpf = 10.0 ** (xf["slope_decades_per_pt"] * vals[:, 0] + xf["intercept_decades"])
    pwl = yf["slope_db_per_pt"] * vals[:, 1] + yf["intercept_db"]
    order = np.argsort(bpf)
    bpf, pwl = bpf[order], pwl[order]
    freq = bpf * BPF_HZ

    rows = []
    raw = vals[order]
    for b, f, p, (xpt, ypt) in zip(bpf, freq, pwl, raw):
        rows.append({
            "source_pdf": PDF.name,
            "source_page": PAGE,
            "figure": "Figure 18(c)",
            "dataset_id": "fernandes_fig18c_54vane_exhaust_pwl",
            "method": "vector path extraction (PyMuPDF get_drawings), thick black experimental curve",
            "BPF_order": f"{b:.6f}",
            "frequency_Hz": f"{f:.3f}",
            "frequency_sigma_Hz": f"{f * FREQ_REL_SIGMA:.3f}",
            "PWL_dB_per_Hz": f"{p:.4f}",
            "PWL_sigma_dB": f"{SIGMA_DB:.3f}",
            "x_pt": f"{xpt:.3f}",
            "y_pt": f"{ypt:.3f}",
        })
    write_csv(DIG / "fernandes_fig18c_54vane_exhaust_pwl.csv", rows, list(rows[0].keys()))

    # 1/12-octave summary
    per_oct = 12
    nbin = max(1, int(round(per_oct * np.log2(bpf.max() / bpf.min()))))
    edges = np.logspace(np.log10(bpf.min()), np.log10(bpf.max()), nbin + 1)
    oct_rows = []
    for k in range(nbin):
        sel = (bpf >= edges[k]) & (bpf < edges[k + 1])
        if not sel.any():
            continue
        fc = float(np.sqrt(edges[k] * edges[k + 1]))
        oct_rows.append({
            "source_pdf": PDF.name,
            "source_page": PAGE,
            "figure": "Figure 18(c)",
            "dataset_id": "fernandes_fig18c_54vane_exhaust_pwl",
            "method": "1/12-octave median of vector path",
            "BPF_order": f"{fc:.6f}",
            "frequency_Hz": f"{fc * BPF_HZ:.3f}",
            "frequency_sigma_Hz": f"{fc * BPF_HZ * FREQ_REL_SIGMA:.3f}",
            "PWL_dB_per_Hz": f"{float(np.median(pwl[sel])):.4f}",
            "PWL_sigma_dB": f"{SIGMA_DB:.3f}",
            "PWL_min_dB": f"{float(pwl[sel].min()):.4f}",
            "PWL_max_dB": f"{float(pwl[sel].max()):.4f}",
            "n_points": int(sel.sum()),
        })
    write_csv(DIG / "fernandes_fig18c_54vane_exhaust_pwl_1_12oct.csv",
              oct_rows, list(oct_rows[0].keys()))

    # QA overlay
    gray = render_page(PDF, PAGE, dpi=400)
    make_overlay(gray, np.c_[vals[:, 0], vals[:, 1]],
                 "Fernandes Fig. 18(c) — 54-vane exhaust experimental PWL",
                 OVER / "fernandes_fig18c_54vane_exhaust_overlay.png")

    # validation
    b2 = (bpf >= 1.9) & (bpf <= 2.1)
    b4 = (bpf >= 3.9) & (bpf <= 4.1)
    checks = [
        {
            "check": "2BPF peak present",
            "dataset": "fernandes_fig18c_54vane_exhaust_pwl",
            "value": f"{float(pwl[b2].max()):.2f}",
            "criterion": "local maximum near 2 BPF with PWL > 82 dB/Hz",
            "pass": bool(pwl[b2].max() > 82.0),
            "note": f"2 BPF = {2 * BPF_HZ:.0f} Hz",
        },
        {
            "check": "high-frequency decline",
            "dataset": "fernandes_fig18c_54vane_exhaust_pwl",
            "value": f"{float(pwl[-1]):.2f}",
            "criterion": "PWL at 10 BPF < PWL at 4 BPF - 10 dB",
            "pass": bool(pwl[-1] < pwl[b4].mean() - 10.0),
            "note": f"4 BPF mean {pwl[b4].mean():.2f}, 10 BPF {pwl[-1]:.2f} dB/Hz",
        },
        {
            "check": "axis calibration",
            "dataset": "fernandes_fig18c_54vane_exhaust_pwl",
            "value": f"x {xf['max_abs_residual_decades']:.2e} dec, y {yf['max_abs_residual_db']:.2e} dB",
            "criterion": "x residual < 1e-3 dec; y residual < 0.01 dB",
            "pass": bool(xf["max_abs_residual_decades"] < 1e-3 and yf["max_abs_residual_db"] < 0.01),
            "note": "17 x gridlines and 10 y gridlines fitted",
        },
    ]

    meta = {
        "dataset_id": "fernandes_fig18c_54vane_exhaust_pwl",
        "pdf": PDF.name,
        "pdf_sha256": sha256(PDF),
        "pdf_page": PAGE,
        "figure": "Figure 18(c)",
        "caption": "Sound power level spectra - exhaust contribution",
        "curve_drawing_index": idx,
        "curve_width_pt": dd.get("width"),
        "curve_color": dd.get("color"),
        "curve_dashes": dd.get("dashes"),
        "n_points": len(rows),
        "bpf_range": [float(bpf.min()), float(bpf.max())],
        "frequency_range_hz": [float(freq.min()), float(freq.max())],
        "calibration": cal,
        "uncertainty": {
            "PWL_sigma_db": SIGMA_DB,
            "frequency_rel_sigma": FREQ_REL_SIGMA,
            "basis": "vector coordinate quantization + gridline fit residuals",
        },
        "caveat": (
            "Fernandes Fig. 18(c) is PWL [dB/Hz] from the LAVA simulation / "
            "experiment breakdown; NASA TM-2002 Fig. 9 is labelled Exhaust PWL "
            "Spectrum, dB.  Compare spectral shape and BPF trend, not absolute "
            "level, unless the processing bandwidths are reconciled."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "fernandes_fig18c.json", meta)
    write_json(QA / "fernandes_validation_checks.json", checks)

    print(f"Fernandes Fig. 18(c): drawing {idx}, {len(rows)} points, "
          f"{freq.min():.0f}-{freq.max():.0f} Hz, PWL {pwl.min():.2f}-{pwl.max():.2f} dB/Hz")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['value']}")
    return meta


if __name__ == "__main__":
    main()
