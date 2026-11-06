"""Digitise Podboy Fig. 26 (B2): upwash PSDs at 61.7 % speed.

Native image: 498 x 348 px, indexed CalRGB with exact black/red/white
palette.  Report p. 8 states black = HW1 and red = HW2.  Panels are, left to
right, 97 %, 81 % and 25 % span (verified against the text-layer titles).

Axes: x = 10^2 .. 10^4 Hz (two decades); y = 10^1 .. 10^-4 ft^2/s^2/Hz
(five decades).  The figure stops at 10 kHz, so the requested 20 kHz
integrals necessarily use a power-law extrapolation from 7--10 kHz for the
10--20 kHz segment; the fully measured 0.1--10 kHz integral is reported
separately.

No vision model: each colour is an exact palette mask, the frame and ticks
are found by run-length analysis, and each column is reduced to the median
and top of its curve pixels.
"""
from __future__ import annotations

import json

import numpy as np
import pymupdf as fitz
from scipy import ndimage

from common import DIG, OVER, PROV, QA, ROOT, write_csv, write_json
from raster_common import (colour_masks, components, detect_panels,
                           load_embedded, page_to_image, remove_ticks)

PDF = ROOT / "20030065891.pdf"
PAGE = 33
IMAGE_INDEX = 0
BPF = 22.0 * 7808.0 / 60.0
TONE_THRESHOLD_DEC = 0.5      # factor ~3.2 above the local floor
FIT_LO_HZ = 6000.0
FIT_HI_HZ = 10000.0
CUT_HZ = 20000.0


def panel_titles() -> list[tuple[str, float]]:
    doc = fitz.open(PDF)
    page = doc[PAGE - 1]
    out = []
    for w in page.get_text("words"):
        if w[4] in ("97%", "81%", "25%") and abs(w[1] - 39) < 6:
            out.append((w[4].replace("%", ""), (w[0] + w[2]) / 2.0))
    return sorted(out, key=lambda t: t[1])


def remove_legends(mask: np.ndarray, panel: dict) -> list[dict]:
    """Remove hollow rectangular legend boxes (only Fig. 26 panel 2 has one)."""
    sub = mask[panel["T"] + 2:panel["B"] - 1, panel["L"] + 2:panel["R"] - 1]
    lab, _ = ndimage.label(sub, np.ones((3, 3)))
    removed = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        w = sl[1].stop - sl[1].start
        h = sl[0].stop - sl[0].start
        if area < 80 or w < 30 or h < 12:
            continue
        filled = int(ndimage.binary_fill_holes(m).sum())
        hole = (filled - area) / filled if filled else 0.0
        if hole < 0.5:
            continue
        x0 = panel["L"] + 2 + sl[1].start - 1
        x1 = panel["L"] + 2 + sl[1].stop + 0
        y0 = panel["T"] + 2 + sl[0].start - 1
        y1 = panel["T"] + 2 + sl[0].stop + 0
        mask[y0:y1, x0:x1] = False
        removed.append({"x0": int(x0), "x1": int(x1), "y0": int(y0),
                        "y1": int(y1), "area": area, "hole_fraction": round(hole, 3)})
    return removed


def trace(mask: np.ndarray, panel: dict) -> list[dict]:
    """One row per pixel column: median/top/bottom curve y."""
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    rows = []
    for x in range(L + 2, R - 1):
        ys = np.where(mask[T + 4:B - 3, x])[0] + T + 4
        if len(ys) == 0:
            continue
        rows.append({"x": x, "y_med": float(np.median(ys)),
                     "y_top": float(np.min(ys)), "y_bot": float(np.max(ys)),
                     "n_px": int(len(ys))})
    return rows


def add_calibration(rows: list[dict], panel: dict) -> None:
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    for r in rows:
        r["log10_f"] = 2.0 + (r["x"] - L) / (R - L) * 2.0
        r["frequency_Hz"] = 10.0 ** r["log10_f"]
        r["log10_psd_med"] = 1.0 - (r["y_med"] - T) / (B - T) * 5.0
        r["log10_psd_top"] = 1.0 - (r["y_top"] - T) / (B - T) * 5.0
        r["PSD"] = 10.0 ** r["log10_psd_med"]


def detect_tones(rows: list[dict]) -> list[dict]:
    x = np.array([r["x"] for r in rows], dtype=float)
    top = np.array([r["log10_psd_top"] for r in rows], dtype=float)
    med = np.array([r["log10_psd_med"] for r in rows], dtype=float)
    floor = np.array([np.median(med[np.abs(x - xi) <= 6]) for xi in x])
    prom = top - floor
    # local maxima only; contiguous columns above threshold form one tone
    cand = prom > TONE_THRESHOLD_DEC
    groups = []
    start = None
    for i, flag in enumerate(cand):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            groups.append((start, i - 1))
            start = None
    if start is not None:
        groups.append((start, len(cand) - 1))
    tones = []
    for a, b in groups:
        j = int(a + np.argmax(prom[a:b + 1]))
        tones.append({
            "f_Hz": rows[j]["frequency_Hz"],
            "peak_log10_psd": float(top[j]),
            "floor_log10_psd": float(floor[j]),
            "prominence_decades": float(prom[j]),
            "width_columns": int(b - a + 1),
            "x_px": int(rows[j]["x"]),
        })
    return tones, prom


def floor_curve(rows: list[dict], tones: list[dict]) -> np.ndarray:
    """Replace tone columns (expanded by one) and interpolate in log-f."""
    x = np.array([r["x"] for r in rows], dtype=float)
    med = np.array([r["log10_psd_med"] for r in rows], dtype=float)
    bad = np.zeros(len(rows), dtype=bool)
    for t in tones:
        bad |= np.abs(x - t["x_px"]) <= (t["width_columns"] / 2.0 + 1.0)
    good = ~bad
    if good.sum() < 5:
        return med.copy()
    return np.interp(x, x[good], med[good])


def integrate(rows: list[dict], logpsd: np.ndarray) -> dict:
    f = np.array([r["frequency_Hz"] for r in rows], dtype=float)
    psd = 10.0 ** logpsd
    f_lo, f_hi = float(f.min()), float(f.max())
    # fully measured integral over the actual digitised range
    i_meas = float(np.trapezoid(psd, f))
    # power-law fit over the last octave that exists in the figure
    hi = min(FIT_HI_HZ, f_hi)
    lo = max(FIT_LO_HZ, f_lo)
    sel = (f >= lo) & (f <= hi)
    if sel.sum() < 8:
        sel = (f >= 0.8 * lo) & (f <= f_hi)
    b, a = np.polyfit(np.log10(f[sel]), np.log10(psd[sel]), 1)
    pred = a + b * np.log10(f[sel])
    ss_res = float(np.sum((np.log10(psd[sel]) - pred) ** 2))
    ss_tot = float(np.sum((np.log10(psd[sel]) - np.mean(np.log10(psd[sel]))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    amp = 10.0 ** a

    def tail(f0: float, f1: float) -> float:
        if b >= -1.0:
            return float("nan")
        return float(amp * (f1 ** (b + 1) - f0 ** (b + 1)) / (b + 1))

    seg_to_20k = tail(f_hi, CUT_HZ) if f_hi < CUT_HZ else 0.0
    if not np.isfinite(seg_to_20k):
        # documented von Karman fallback, slope -5/3 anchored at f_hi
        b_fb = -5.0 / 3.0
        p_hi = float(np.interp(np.log10(f_hi), np.log10(f), logpsd))
        amp_fb = 10.0 ** (p_hi - b_fb * np.log10(f_hi))
        seg_to_20k = float(amp_fb * (CUT_HZ ** (b_fb + 1) - f_hi ** (b_fb + 1)) / (b_fb + 1))
        tail_20_inf = float(amp_fb * CUT_HZ ** (b_fb + 1) / (-b_fb - 1))
        fallback = True
    else:
        tail_20_inf = tail(CUT_HZ, np.inf)
        fallback = False
    return {
        "integral_measured_full": i_meas,
        "integral_20k_truncated": i_meas + seg_to_20k,
        "integral_extrapolated_inf": i_meas + seg_to_20k + tail_20_inf,
        "tail_measured_to_20k": seg_to_20k,
        "tail_20k_inf": tail_20_inf,
        "slope_loglog": float(b),
        "fit_r2": float(r2),
        "fit_lo_Hz": float(f[sel].min()),
        "fit_hi_Hz": float(f[sel].max()),
        "von_karman_fallback": int(fallback),
        "f_min_measured_Hz": f_lo,
        "f_max_measured_Hz": f_hi,
        "n_points": int(len(f)),
    }


def make_plot(rows_by_curve, tones_by_curve, panels, outpath) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True)
    for ax, panel, title in zip(axes, panels, ["97 % span", "81 % span", "25 % span"]):
        for station, colour in (("HW1", "black"), ("HW2", "red")):
            rows = rows_by_curve[(panel["L"], station)]
            f = [r["frequency_Hz"] for r in rows]
            y = [r["PSD"] for r in rows]
            ax.plot(f, y, color=colour, lw=1.0, label=station)
            for t in tones_by_curve[(panel["L"], station)]:
                ax.plot(t["f_Hz"], 10 ** t["peak_log10_psd"], "x", color=colour, ms=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(100, 10000); ax.set_ylim(1e-4, 10)
        ax.set_title(title)
        ax.set_xlabel("Frequency, Hz")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    axes[0].set_ylabel("PSD, ft$^2$/s$^2$/Hz")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Podboy Fig. 26 — upwash PSD, 61.7 % speed (black HW1, red HW2)")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> dict:
    rgb, path, info, rect = load_embedded(PDF, PAGE, IMAGE_INDEX)
    masks = colour_masks(rgb)
    panels = detect_panels(masks["black"])
    if len(panels) != 3:
        raise RuntimeError(f"expected 3 panels, found {len(panels)}")
    titles = panel_titles()
    # map title x to panel index
    ordered = []
    for label, xpt in titles:
        xi, _ = page_to_image(xpt, 0, rect, info["width"], info["height"])
        which = [i for i, p in enumerate(panels) if p["L"] <= xi <= p["R"]]
        if len(which) != 1:
            raise RuntimeError(f"title {label} at image x={xi:.1f} not in one panel")
        ordered.append((label, which[0]))
    if [lab for lab, _ in ordered] != ["97", "81", "25"]:
        raise RuntimeError(f"unexpected panel title order: {ordered}")

    rows_by_curve = {}
    tones_by_curve = {}
    all_rows = []
    tone_rows = []
    integrals = []
    for i, panel in enumerate(panels):
        span = ordered[i][0]
        # detect the legend box once from black, then remove it from every colour
        probe = masks["black"].copy()
        legends = remove_legends(probe, panel)
        for colour, station in (("black", "HW1"), ("red", "HW2")):
            mask = masks[colour].copy()
            for b in legends:
                mask[b["y0"]:b["y1"], b["x0"]:b["x1"]] = False
            mask = remove_ticks(mask, panel)
            rows = trace(mask, panel)
            add_calibration(rows, panel)
            tones, prom = detect_tones(rows)
            rows_by_curve[(panel["L"], station)] = rows
            tones_by_curve[(panel["L"], station)] = tones
            floor = floor_curve(rows, tones)
            for j, r in enumerate(rows):
                all_rows.append({
                    "panel": span,
                    "span_percent": span,
                    "station": station,
                    "colour": colour,
                    "x_px": r["x"],
                    "y_px": f"{r['y_med']:.2f}",
                    "y_top_px": f"{r['y_top']:.2f}",
                    "frequency_Hz": f"{r['frequency_Hz']:.3f}",
                    "BPF_order": f"{r['frequency_Hz'] / BPF:.5f}",
                    "PSD_ft2_s2_Hz": f"{r['PSD']:.6e}",
                    "log10_PSD": f"{r['log10_psd_med']:.5f}",
                    "log10_PSD_top": f"{r['log10_psd_top']:.5f}",
                    "floor_log10_PSD": f"{floor[j]:.5f}",
                    "tone_prominence_decades": f"{prom[j]:.4f}",
                    "tone_flag": int(prom[j] > TONE_THRESHOLD_DEC),
                    "n_pixels": r["n_px"],
                })
            for t in tones:
                tone_rows.append({
                    "panel": span,
                    "span_percent": span,
                    "station": station,
                    "frequency_Hz": f"{t['f_Hz']:.3f}",
                    "BPF_order": f"{t['f_Hz'] / BPF:.5f}",
                    "peak_log10_PSD": f"{t['peak_log10_psd']:.5f}",
                    "floor_log10_PSD": f"{t['floor_log10_psd']:.5f}",
                    "prominence_decades": f"{t['prominence_decades']:.4f}",
                    "prominence_factor": f"{10 ** t['prominence_decades']:.3f}",
                    "width_columns": t["width_columns"],
                    "method": "local top above 13-column median; threshold 0.5 decade",
                })
            # primary integral: 97 % uses the whole curve, 81/25 use the floor
            use_floor = span in ("81", "25")
            logpsd = floor if use_floor else np.array([r["log10_psd_med"] for r in rows])
            integ = integrate(rows, logpsd)
            integ.update({
                "panel": span,
                "span_percent": span,
                "station": station,
                "curve_used": "floor between tones" if use_floor else "whole curve",
                "source_limitation": "figure stops at 10 kHz; 10-20 kHz segment is "
                                     "power-law extrapolation, not measured",
            })
            integrals.append(integ)
            print(f"Fig26 {span}% {station}: {len(rows)} cols, {len(tones)} tones, "
                  f"I_meas={integ['integral_measured_full']:.3e}, "
                  f"I_20k={integ['integral_20k_truncated']:.3e}, "
                  f"I_inf={integ['integral_extrapolated_inf']:.3e}, "
                  f"slope={integ['slope_loglog']:.2f}, R2={integ['fit_r2']:.3f}")

    write_csv(DIG / "podboy_fig26_psd.csv", all_rows, list(all_rows[0].keys()))
    if tone_rows:
        write_csv(DIG / "podboy_fig26_tones.csv", tone_rows, list(tone_rows[0].keys()))
    write_csv(DIG / "podboy_fig26_variance_integrals.csv", integrals,
              list(integrals[0].keys()))
    make_plot(rows_by_curve, tones_by_curve, panels,
              OVER / "podboy_fig26_psd_overlay.png")

    meta = {
        "pdf": PDF.name,
        "pdf_page": PAGE,
        "report_page": 29,
        "image": info,
        "panels": panels,
        "panel_order": ordered,
        "palette": "black/red/white",
        "station_mapping": "black = HW1, red = HW2 (report p. 8 text)",
        "axes": {"x": "10^2 to 10^4 Hz", "y": "10^1 to 10^-4 ft2/s2/Hz"},
        "method": "exact palette masks; per-column median/top; tone threshold 0.5 decade",
        "source_limitation": "Fig. 26 stops at 10 kHz, so 20 kHz integrals use a "
                             "power-law tail fit over 6-10 kHz",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "podboy_fig26.json", meta)
    write_json(QA / "podboy_fig26_summary.json",
               {"integrals": integrals, "n_psd_rows": len(all_rows),
                "n_tones": len(tone_rows)})
    return meta


if __name__ == "__main__":
    main()
