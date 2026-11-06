"""Digitise the measured exhaust spectra from NASA/TM-2002-211798 page 14.

Panels (report p. 10):
  TL  Fig.  9  54 radial vanes, approach, exhaust
  BL  Fig. 10  26 radial vanes, approach, exhaust
  TR  Fig. 11  26 swept  vanes, approach, exhaust

Method: 400 dpi grayscale render; fixed panel frames and one-time label boxes
from calibration JSON; connected-component morphology keeps the small solid
dashes of the measured curve and rejects the solid theory line and open-circle
symbols.  Each dash centroid is converted with the fitted axis calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (CAL, DIG, QA, ROOT, OVER, extract_dashes, render_page,
                    write_csv, write_json)

PDF = ROOT / "20030003692.pdf"
PAGE = 14
DPI = 400
BPF_HZ = 7808.0 * 22.0 / 60.0  # 2862.933... Hz at 61.7% speed
PWL_SIGMA_RASTER = 0.12       # dB, k=1 (pixel + calibration + line thickness)
FREQ_REL_SIGMA = 0.0025       # 0.25 %, k=1

DATASETS = {
    "TL": {
        "dataset_id": "tm2002_p14_fig09_54vane_exhaust",
        "figure": "Figure 9",
        "panel": "TL",
        "vanes": 54,
        "vane_type": "radial",
        "outfile": "tm2002_p14_fig09_54vane_exhaust_measured.csv",
        "dashfile": "tm2002_p14_fig09_54vane_exhaust_dashes.csv",
        "octfile": "tm2002_p14_fig09_54vane_exhaust_1_6oct.csv",
    },
    "BL": {
        "dataset_id": "tm2002_p14_fig10_26vane_exhaust",
        "figure": "Figure 10",
        "panel": "BL",
        "vanes": 26,
        "vane_type": "radial",
        "outfile": "tm2002_p14_fig10_26vane_exhaust_measured.csv",
        "dashfile": "tm2002_p14_fig10_26vane_exhaust_dashes.csv",
        "octfile": "tm2002_p14_fig10_26vane_exhaust_1_6oct.csv",
    },
    "TR": {
        "dataset_id": "tm2002_p14_fig11_26swept_exhaust",
        "figure": "Figure 11",
        "panel": "TR",
        "vanes": 26,
        "vane_type": "swept 30 deg",
        "outfile": "tm2002_p14_fig11_26swept_exhaust_measured.csv",
        "dashfile": "tm2002_p14_fig11_26swept_exhaust_dashes.csv",
        "octfile": "tm2002_p14_fig11_26swept_exhaust_1_6oct.csv",
    },
}

COMMON = {
    "source_pdf": PDF.name,
    "source_page": PAGE,
    "report_page": 10,
    "method": "raster-dash morphology at 400 dpi",
}


def f_from_x(x: np.ndarray | float, panel: dict) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xf = panel["x_fit"]
    return 10.0 ** (xf["slope_decades_per_px"] * x + xf["intercept_decades"])


def db_from_y(y: np.ndarray | float, panel: dict) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yf = panel["y_fit"]
    return yf["slope_db_per_px"] * y + yf["intercept_db"]


def local_median(dashes: list[dict], i: int, half: int = 15) -> float:
    x = dashes[i]["x_px"]
    vals = [d["PWL_dB"] for d in dashes if abs(d["x_px"] - x) <= half]
    return float(np.median(vals)) if vals else float("nan")


def bin_spectrum(dashes: list[dict], panel: dict, bin_px: float = 3.0) -> list[dict]:
    L, T, R, B = panel["frame_px"]
    centers = np.arange(L + 12 + bin_px / 2.0, R - 8.0, bin_px)
    xs = np.array([d["x_px"] for d in dashes])
    ys = np.array([d["y_px"] for d in dashes])
    dbs = np.array([d["PWL_dB"] for d in dashes])

    rows: list[dict] = []
    for c in centers:
        sel = np.abs(xs - c) <= bin_px / 2.0
        n = int(sel.sum())
        if n:
            med_y = float(np.median(ys[sel]))
            med_db = float(np.median(dbs[sel]))
            f = float(f_from_x(c, panel))
            rows.append({
                "x_px": float(c),
                "y_px": med_y,
                "frequency_Hz": f,
                "frequency_sigma_Hz": f * FREQ_REL_SIGMA,
                "BPF_order": f / BPF_HZ,
                "PWL_dB": med_db,
                "PWL_sigma_dB": PWL_SIGMA_RASTER,
                "PWL_min_dB": float(np.min(dbs[sel])),
                "PWL_max_dB": float(np.max(dbs[sel])),
                "n_dashes": n,
                "bin_width_px": bin_px,
                "interpolated": 0,
                "note": "",
            })
        else:
            f = float(f_from_x(c, panel))
            rows.append({
                "x_px": float(c),
                "y_px": float("nan"),
                "frequency_Hz": f,
                "frequency_sigma_Hz": f * FREQ_REL_SIGMA,
                "BPF_order": f / BPF_HZ,
                "PWL_dB": float("nan"),
                "PWL_sigma_dB": PWL_SIGMA_RASTER,
                "PWL_min_dB": float("nan"),
                "PWL_max_dB": float("nan"),
                "n_dashes": 0,
                "bin_width_px": bin_px,
                "interpolated": 0,
                "note": "no dash centroid in bin",
            })

    # trim leading/trailing empty bins
    valid = [i for i, r in enumerate(rows) if r["n_dashes"] > 0]
    if not valid:
        return []
    rows = rows[valid[0]:valid[-1] + 1]

    # linear interpolation in (log10 f, dB) across short gaps only
    logf = np.log10([r["frequency_Hz"] for r in rows])
    val = np.array([r["PWL_dB"] for r in rows])
    for i in range(len(rows)):
        if np.isfinite(val[i]):
            continue
        prev = i - 1
        while prev >= 0 and not np.isfinite(val[prev]):
            prev -= 1
        nxt = i + 1
        while nxt < len(rows) and not np.isfinite(val[nxt]):
            nxt += 1
        if prev >= 0 and nxt < len(rows) and (nxt - prev) <= 4:
            w = (logf[i] - logf[prev]) / (logf[nxt] - logf[prev])
            rows[i]["PWL_dB"] = float(val[prev] + w * (val[nxt] - val[prev]))
            rows[i]["y_px"] = float(rows[prev]["y_px"] + w * (rows[nxt]["y_px"] - rows[prev]["y_px"]))
            rows[i]["interpolated"] = 1
            rows[i]["note"] = f"linear interpolation across {nxt - prev - 1} empty bin(s)"
    return rows


def octave_summary(dashes: list[dict], panel: dict, per_octave: int = 6) -> list[dict]:
    f = np.array([d["frequency_Hz"] for d in dashes])
    db = np.array([d["PWL_dB"] for d in dashes])
    fmin, fmax = float(f.min()), float(f.max())
    nbin = max(1, int(round(per_octave * np.log2(fmax / fmin))))
    edges = np.logspace(np.log10(fmin), np.log10(fmax), nbin + 1)
    rows = []
    for k in range(nbin):
        sel = (f >= edges[k]) & (f < edges[k + 1])
        if not sel.any():
            continue
        fc = float(np.sqrt(edges[k] * edges[k + 1]))
        rows.append({
            "frequency_Hz": fc,
            "frequency_sigma_Hz": fc * FREQ_REL_SIGMA,
            "BPF_order": fc / BPF_HZ,
            "PWL_dB": float(np.median(db[sel])),
            "PWL_sigma_dB": PWL_SIGMA_RASTER,
            "PWL_min_dB": float(db[sel].min()),
            "PWL_max_dB": float(db[sel].max()),
            "n_dashes": int(sel.sum()),
            "fractional_octave": f"1/{per_octave}",
        })
    return rows


def make_overlay(gray: np.ndarray, panel: dict, dashes: list[dict],
                 spectrum: list[dict], title: str, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, T, R, B = panel["frame_px"]
    m = 18
    crop = gray[max(0, T - m):B + m, max(0, L - m):R + m]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ox, oy = max(0, L - m), max(0, T - m)
    ax.scatter([d["x_px"] - ox for d in dashes],
               [d["y_px"] - oy for d in dashes],
               s=5, c="red", alpha=0.75, label="dash centroids (measured)")
    sx = [r["x_px"] - ox for r in spectrum if np.isfinite(r["PWL_dB"])]
    sy = [r["y_px"] - oy for r in spectrum if np.isfinite(r["PWL_dB"])]
    ax.plot(sx, sy, "-", color="cyan", lw=1.0, alpha=0.9, label="3 px binned median")
    ax.set_xlim(0, crop.shape[1])
    ax.set_ylim(crop.shape[0], 0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> list[dict]:
    cal = json.loads((CAL / "p14_panels.json").read_text())
    gray = render_page(PDF, PAGE, dpi=DPI)

    all_checks: list[dict] = []
    summary: list[dict] = []

    for key, meta in DATASETS.items():
        panel = cal["panels"][key]
        dashes = extract_dashes(gray, panel, threshold=cal["threshold_gray"])
        if len(dashes) < 50:
            raise RuntimeError(f"{key}: only {len(dashes)} dashes found")

        for d in dashes:
            d["frequency_Hz"] = float(f_from_x(d["x_px"], panel))
            d["BPF_order"] = d["frequency_Hz"] / BPF_HZ
            d["PWL_dB"] = float(db_from_y(d["y_px"], panel))

        # local residual flag: low outliers are usually solid-line fragments
        for i, d in enumerate(dashes):
            med = local_median(dashes, i, half=15)
            d["local_median_dB"] = med
            d["residual_dB"] = d["PWL_dB"] - med if np.isfinite(med) else float("nan")
            if np.isfinite(d["residual_dB"]) and d["residual_dB"] < -2.5:
                d["quality_flag"] = "low_outlier"
            elif np.isfinite(d["residual_dB"]) and d["residual_dB"] > 2.5:
                d["quality_flag"] = "tone_or_peak"
            else:
                d["quality_flag"] = "ok"

        spectrum = bin_spectrum(dashes, panel, bin_px=3.0)
        if len(spectrum) < 50:
            raise RuntimeError(f"{key}: spectrum too short")

        # ---- per-dash CSV ----
        dash_rows = []
        for d in dashes:
            row = dict(COMMON)
            row.update({
                "dataset_id": meta["dataset_id"],
                "figure": meta["figure"],
                "panel": meta["panel"],
                "x_px": f"{d['x_px']:.2f}",
                "y_px": f"{d['y_px']:.2f}",
                "frequency_Hz": f"{d['frequency_Hz']:.3f}",
                "frequency_sigma_Hz": f"{d['frequency_Hz'] * FREQ_REL_SIGMA:.3f}",
                "BPF_order": f"{d['BPF_order']:.5f}",
                "PWL_dB": f"{d['PWL_dB']:.3f}",
                "PWL_sigma_dB": f"{PWL_SIGMA_RASTER:.3f}",
                "area_px": d["area_px"],
                "width_px": d["width_px"],
                "height_px": d["height_px"],
                "hole_fraction": f"{d['hole_fraction']:.4f}",
                "local_median_dB": f"{d['local_median_dB']:.3f}",
                "residual_dB": f"{d['residual_dB']:.3f}",
                "quality_flag": d["quality_flag"],
            })
            dash_rows.append(row)
        write_csv(DIG / meta["dashfile"], dash_rows, list(dash_rows[0].keys()))

        # ---- spectrum CSV ----
        spec_rows = []
        for r in spectrum:
            row = dict(COMMON)
            row.update({
                "dataset_id": meta["dataset_id"],
                "figure": meta["figure"],
                "panel": meta["panel"],
                "frequency_Hz": f"{r['frequency_Hz']:.3f}",
                "frequency_sigma_Hz": f"{r['frequency_sigma_Hz']:.3f}",
                "BPF_order": f"{r['BPF_order']:.5f}",
                "PWL_dB": f"{r['PWL_dB']:.3f}" if np.isfinite(r["PWL_dB"]) else "",
                "PWL_sigma_dB": f"{r['PWL_sigma_dB']:.3f}",
                "PWL_min_dB": f"{r['PWL_min_dB']:.3f}" if np.isfinite(r["PWL_min_dB"]) else "",
                "PWL_max_dB": f"{r['PWL_max_dB']:.3f}" if np.isfinite(r["PWL_max_dB"]) else "",
                "n_dashes": r["n_dashes"],
                "x_px": f"{r['x_px']:.2f}",
                "y_px": f"{r['y_px']:.2f}" if np.isfinite(r["y_px"]) else "",
                "bin_width_px": f"{r['bin_width_px']:.1f}",
                "interpolated": r["interpolated"],
                "note": r["note"],
            })
            spec_rows.append(row)
        write_csv(DIG / meta["outfile"], spec_rows, list(spec_rows[0].keys()))

        # ---- 1/6-octave summary ----
        oct_rows = []
        for r in octave_summary(dashes, panel):
            row = dict(COMMON)
            row.update({
                "dataset_id": meta["dataset_id"],
                "figure": meta["figure"],
                "panel": meta["panel"],
                "frequency_Hz": f"{r['frequency_Hz']:.3f}",
                "frequency_sigma_Hz": f"{r['frequency_sigma_Hz']:.3f}",
                "BPF_order": f"{r['BPF_order']:.5f}",
                "PWL_dB": f"{r['PWL_dB']:.3f}",
                "PWL_sigma_dB": f"{r['PWL_sigma_dB']:.3f}",
                "PWL_min_dB": f"{r['PWL_min_dB']:.3f}",
                "PWL_max_dB": f"{r['PWL_max_dB']:.3f}",
                "n_dashes": r["n_dashes"],
                "fractional_octave": r["fractional_octave"],
            })
            oct_rows.append(row)
        write_csv(DIG / meta["octfile"], oct_rows, list(oct_rows[0].keys()))

        # ---- QA overlay ----
        title = f"{meta['dataset_id']} — {meta['figure']} ({meta['vanes']}-vane {meta['vane_type']})"
        make_overlay(gray, panel, dashes, spectrum, title,
                     OVER / f"{meta['dataset_id']}_overlay.png")

        # ---- validation metrics ----
        fp = np.array([d["frequency_Hz"] for d in dashes])
        db = np.array([d["PWL_dB"] for d in dashes])
        bpf_band = (fp >= 0.95 * BPF_HZ) & (fp <= 1.05 * BPF_HZ)
        b2_band = (fp >= 1.95 * BPF_HZ) & (fp <= 2.05 * BPF_HZ)
        bpf_max = float(db[bpf_band].max()) if bpf_band.any() else float("nan")
        b2_max = float(db[b2_band].max()) if b2_band.any() else float("nan")
        all_checks.append({
            "check": "BPF band maximum",
            "dataset": meta["dataset_id"],
            "value": f"{bpf_max:.2f}",
            "criterion": "54-vane < 102 dB; 26-vane > 108 dB (report: BPF cut off/cut on)",
            "pass": (bpf_max < 102.0) if meta["vanes"] == 54 else (bpf_max > 108.0),
            "note": f"band 0.95-1.05 BPF; 2BPF band max {b2_max:.2f} dB",
        })
        all_checks.append({
            "check": "spectral extent",
            "dataset": meta["dataset_id"],
            "value": f"{fp.max():.0f}",
            "criterion": "maximum frequency in 40-55 kHz (report computations to 50 kHz)",
            "pass": 40000.0 <= fp.max() <= 55000.0,
            "note": f"minimum {fp.min():.0f} Hz; max BPF order {fp.max() / BPF_HZ:.2f}",
        })

        summary.append({
            "dataset_id": meta["dataset_id"],
            "figure": meta["figure"],
            "vanes": meta["vanes"],
            "vane_type": meta["vane_type"],
            "n_dashes": len(dashes),
            "n_spectrum_bins": len(spectrum),
            "f_min": float(fp.min()),
            "f_max": float(fp.max()),
            "PWL_min": float(db.min()),
            "PWL_max": float(db.max()),
            "BPF_band_max": bpf_max,
            "two_BPF_band_max": b2_max,
            "low_outliers": sum(1 for d in dashes if d["quality_flag"] == "low_outlier"),
            "tone_points": sum(1 for d in dashes if d["quality_flag"] == "tone_or_peak"),
        })

    # cross-panel low-frequency consistency (report: TL and BL similar < 5.5 kHz)
    tl = [r for r in summary if r["dataset_id"].endswith("54vane_exhaust")][0]
    bl = [r for r in summary if r["dataset_id"].endswith("26vane_exhaust")][0]
    # recompute medians directly from the spectra below 2.5 kHz
    def median_below(path: Path, fmax: float) -> float:
        from common import read_csv
        rows = read_csv(path)
        vals = []
        for r in rows:
            if not r["PWL_dB"]:
                continue
            try:
                v = float(r["PWL_dB"])
            except ValueError:
                continue
            if np.isfinite(v) and float(r["frequency_Hz"]) <= fmax:
                vals.append(v)
        return float(np.median(vals))
    tl_low = median_below(DIG / DATASETS["TL"]["outfile"], 2500.0)
    bl_low = median_below(DIG / DATASETS["BL"]["outfile"], 2500.0)
    all_checks.append({
        "check": "low-frequency TL vs BL",
        "dataset": "tm2002_p14_fig09_54vane_exhaust vs tm2002_p14_fig10_26vane_exhaust",
        "value": f"{bl_low - tl_low:+.2f}",
        "criterion": "|difference| < 5 dB below 2.5 kHz (report says not significantly different < 5.5 kHz)",
        "pass": abs(bl_low - tl_low) < 5.0,
        "note": f"median TL {tl_low:.2f} dB, BL {bl_low:.2f} dB",
    })

    write_csv(QA / "validation_checks.csv", all_checks, list(all_checks[0].keys()))
    write_json(QA / "p14_extraction_summary.json", summary)

    print("\nExtraction summary")
    for s in summary:
        print(f"  {s['dataset_id']}: {s['n_dashes']} dashes, {s['n_spectrum_bins']} bins, "
              f"{s['f_min']:.0f}-{s['f_max']:.0f} Hz, PWL {s['PWL_min']:.1f}-{s['PWL_max']:.1f} dB, "
              f"BPF max {s['BPF_band_max']:.1f}, 2BPF max {s['two_BPF_band_max']:.1f}")
    print("\nValidation")
    for c in all_checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['value']} ({c['note']})")
    return summary


if __name__ == "__main__":
    main()
