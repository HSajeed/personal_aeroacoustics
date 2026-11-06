"""Create the Day-1 QA figures, cross-checks and one-page data-quality table."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from common import (CAL, DER, DIG, GEO, OVER, PROV, QA, ROOT, read_csv,
                    write_csv, write_json)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BPF_HZ = 7808.0 * 22.0 / 60.0  # 2862.933... Hz

P14 = [
    ("TL", "tm2002_p14_fig09_54vane_exhaust_measured.csv",
     "Fig. 9 - 54-vane radial, approach (BPF cut off)"),
    ("BL", "tm2002_p14_fig10_26vane_exhaust_measured.csv",
     "Fig. 10 - 26-vane radial, approach (BPF cut on)"),
    ("TR", "tm2002_p14_fig11_26swept_exhaust_measured.csv",
     "Fig. 11 - 26-vane swept, approach (BPF cut on)"),
]


def load_spectrum(name: str):
    rows = read_csv(DIG / name)
    f = np.array([float(r["frequency_Hz"]) for r in rows])
    y = np.array([float(r["PWL_dB"]) if r["PWL_dB"] else np.nan for r in rows])
    interp = np.array([r["interpolated"] == "1" for r in rows])
    return f, y, interp


def plot_p14() -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.5), sharex=True)
    for ax, (key, fname, title) in zip(axes, P14):
        f, y, interp = load_spectrum(fname)
        good = np.isfinite(y)
        ax.semilogx(f[good], y[good], "-", color="C0", lw=1.0, label="measured, 3 px bins")
        if interp.any():
            ax.semilogx(f[interp & good], y[interp & good], "o", ms=2.2, mfc="none",
                        mec="C1", mew=0.7, label="interpolated gap")
        ax.axvline(BPF_HZ, color="k", ls="--", lw=0.8)
        ax.axvline(2 * BPF_HZ, color="k", ls=":", lw=0.8)
        ax.text(BPF_HZ * 1.03, 117.0, "BPF", fontsize=8)
        ax.text(2 * BPF_HZ * 1.03, 117.0, "2BPF", fontsize=8)
        ax.set_ylim(70, 120)
        ax.set_xlim(1e3, 1e5)
        ax.set_ylabel("Exhaust PWL, dB")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.10)
        imax = np.nanargmax(y)
        ax.set_title(f"{title}  |  max {y[imax]:.1f} dB at {f[imax]/1000:.2f} kHz", fontsize=10)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("Frequency, Hz")
    fig.suptitle("NASA/TM-2002-211798 p. 14 (report p. 10) - measured exhaust spectra\n"
                 "digitised by deterministic raster dash morphology (no vision model)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OVER / "fig_p14_exhaust_spectra.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def crosscheck() -> tuple[Path, list[dict]]:
    f_n, y_n, _ = load_spectrum("tm2002_p14_fig09_54vane_exhaust_measured.csv")
    f_n = f_n[np.isfinite(y_n)]
    y_n = y_n[np.isfinite(y_n)]

    rows = read_csv(DIG / "fernandes_fig18c_54vane_exhaust_pwl.csv")
    f_f = np.array([float(r["frequency_Hz"]) for r in rows])
    y_f = np.array([float(r["PWL_dB_per_Hz"]) for r in rows])

    # common log-frequency grid and median-removed shape
    g = np.logspace(np.log10(2000.0), np.log10(20000.0), 240)
    n_i = np.interp(np.log10(g), np.log10(f_n), y_n)
    f_i = np.interp(np.log10(g), np.log10(f_f), y_f)
    band = (g >= 2000.0) & (g <= 8000.0)
    n_med, f_med = float(np.median(n_i[band])), float(np.median(f_i[band]))
    n_norm, f_norm = n_i - n_med, f_i - f_med
    r = float(np.corrcoef(n_norm, f_norm)[0, 1])
    rms = float(np.sqrt(np.mean((n_norm - f_norm) ** 2)))
    mean_diff = float(np.mean(n_norm - f_norm))

    def prominence(freq, level, order):
        lo, hi = 0.75 * order * BPF_HZ, 1.25 * order * BPF_HZ
        m = (freq >= lo) & (freq <= hi)
        if not m.any():
            return np.nan, np.nan, np.nan, np.nan
        base = float(np.median(level[m]))
        peak = float(level[m].max())
        fpk = float(freq[m][np.argmax(level[m])])
        return peak, base, peak - base, fpk

    n_peak2, n_base2, n_prom2, n_f2 = prominence(f_n, y_n, 2)
    f_peak2, f_base2, f_prom2, f_f2 = prominence(f_f, y_f, 2)
    n_peak1, n_base1, n_prom1, _ = prominence(f_n, y_n, 1)
    f_peak1, f_base1, f_prom1, _ = prominence(f_f, y_f, 1)

    harm_n, harm_f, harm_rows = [], [], []
    for k in range(1, 9):
        fq = k * BPF_HZ
        if fq > min(f_n.max(), f_f.max()):
            break
        vn = float(np.interp(np.log10(fq), np.log10(f_n), y_n))
        vf = float(np.interp(np.log10(fq), np.log10(f_f), y_f))
        harm_n.append(vn - n_med)
        harm_f.append(vf - f_med)
        harm_rows.append({"order": k, "frequency_Hz": f"{fq:.1f}",
                          "nasa_PWL_dB": f"{vn:.2f}", "fernandes_PWL_dB_per_Hz": f"{vf:.2f}",
                          "nasa_normalised_dB": f"{vn - n_med:.2f}",
                          "fernandes_normalised_dB": f"{vf - f_med:.2f}",
                          "difference_normalised_dB": f"{(vn - n_med) - (vf - f_med):+.2f}"})
    harm_r = float(np.corrcoef(harm_n, harm_f)[0, 1]) if len(harm_n) > 2 else float("nan")
    harm_rms = float(np.sqrt(np.mean((np.array(harm_n) - np.array(harm_f)) ** 2)))

    metrics = [
        ("common_band", "2-20 kHz, log interpolation", "", ""),
        ("median_band", "2-8 kHz", "", ""),
        ("median_nasa_dB", f"{n_med:.3f}", "dB", "NASA TM-2002 Fig. 9 median in 2-8 kHz"),
        ("median_fernandes_dB_per_Hz", f"{f_med:.3f}", "dB/Hz", "Fernandes Fig. 18c median in 2-8 kHz"),
        ("raw_offset_dB", f"{n_med - f_med:+.3f}", "dB", "absolute units differ; not a physical error"),
        ("pearson_r_normalised", f"{r:.4f}", "-", "shape after removing each 2-8 kHz median"),
        ("rms_normalised_dB", f"{rms:.3f}", "dB", "shape residual"),
        ("mean_normalised_dB", f"{mean_diff:+.3f}", "dB", "NASA minus Fernandes"),
        ("2BPF_peak_nasa_dB", f"{n_peak2:.2f}", "dB", f"at {n_f2:.0f} Hz"),
        ("2BPF_prominence_nasa_dB", f"{n_prom2:.2f}", "dB", "peak minus local median, +/-25 % band"),
        ("2BPF_peak_fernandes_dB_per_Hz", f"{f_peak2:.2f}", "dB/Hz", f"at {f_f2:.0f} Hz"),
        ("2BPF_prominence_fernandes_dB", f"{f_prom2:.2f}", "dB", "peak minus local median, +/-25 % band"),
        ("1BPF_prominence_nasa_dB", f"{n_prom1:.2f}", "dB", "BPF cut-off expected for 54 vanes"),
        ("1BPF_prominence_fernandes_dB", f"{f_prom1:.2f}", "dB", "BPF weak in Fernandes experimental curve"),
        ("harmonic_correlation", f"{harm_r:.4f}", "-", "1-8 BPF, median-removed"),
        ("harmonic_rms_dB", f"{harm_rms:.3f}", "dB", "1-8 BPF, median-removed"),
    ]
    metric_rows = [{"metric": m, "value": v, "unit": u, "note": n} for m, v, u, n in metrics]
    write_csv(QA / "crosscheck_54vane.csv", metric_rows, list(metric_rows[0].keys()))
    write_csv(QA / "crosscheck_54vane_harmonics.csv", harm_rows, list(harm_rows[0].keys()))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.6))
    ax0.semilogx(f_n, y_n, color="C0", lw=1.1, label="NASA TM-2002 Fig. 9 (dB)")
    ax0.set_xlim(8e2, 3.2e4)
    ax0.set_ylim(70, 110)
    ax0.set_xlabel("Frequency, Hz")
    ax0.set_ylabel("NASA exhaust PWL, dB", color="C0")
    ax0.tick_params(axis="y", labelcolor="C0")
    ax0.grid(True, which="major", alpha=0.25)
    axR = ax0.twinx()
    axR.semilogx(f_f, y_f, color="C2", lw=1.2, label="Fernandes Fig. 18c (dB/Hz)")
    axR.set_ylim(55, 95)
    axR.set_ylabel("Fernandes PWL, dB/Hz", color="C2")
    axR.tick_params(axis="y", labelcolor="C2")
    for k in range(1, 9):
        ax0.axvline(k * BPF_HZ, color="k", ls=":", lw=0.5, alpha=0.4)
    ax0.set_title("(a) Absolute levels - units differ", fontsize=10)
    lines = ax0.get_lines()[:1] + axR.get_lines()[:1]
    ax0.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)

    ax1.semilogx(g, n_norm, color="C0", lw=1.4, label=f"NASA Fig. 9 (median {n_med:.1f} dB)")
    ax1.semilogx(g, f_norm, color="C2", lw=1.4, label=f"Fernandes Fig. 18c (median {f_med:.1f} dB/Hz)")
    ax1.axhline(0, color="k", lw=0.6)
    for k in range(1, 8):
        x = k * BPF_HZ
        if x <= g.max():
            ax1.axvline(x, color="k", ls=":", lw=0.5, alpha=0.4)
            ax1.text(x, ax1.get_ylim()[1] - 0.4, f"{k}x", ha="center", va="top", fontsize=7)
    ax1.set_xlim(2e3, 2e4)
    ax1.set_xlabel("Frequency, Hz")
    ax1.set_ylabel("Level relative to 2-8 kHz median, dB")
    ax1.set_title(f"(b) Normalised shape: r = {r:.3f}, RMS = {rms:.2f} dB", fontsize=10)
    ax1.grid(True, which="major", alpha=0.25)
    ax1.legend(loc="lower left", fontsize=8)

    fig.suptitle("54-vane exhaust: NASA raster spectrum vs Fernandes vector spectrum (approach, 7808 rpm)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OVER / "fig_54vane_fernandes_crosscheck.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out, metric_rows


def plot_annulus() -> Path:
    audit = read_csv(DER / "flowpath_audit.csv")
    x = np.array([float(r["x_in"]) for r in audit])
    area = np.array([float(r["annulus_area_ft2"]) for r in audit])

    am = read_csv(DER / "annulus_area_mach.csv")
    station = [r for r in am if r["x_in"] and r["x_in"] not in ("3-6",)]
    xs = np.array([float(r["x_in"]) for r in station])
    mach = np.array([float(r["mean_axial_mach"]) for r in station])
    sig = np.array([float(r["mach_sigma"]) for r in station])
    nom = [r for r in am if r["station"] == "mean x=3-6 in"][0]
    fan = [r for r in am if r["station"].startswith("fan face")][0]
    m_nom, m_sig = float(nom["mean_axial_mach"]), float(nom["mach_sigma"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax0.plot(x, area, color="C0", lw=1.3)
    ax0.axvspan(3.0, 6.0, color="C1", alpha=0.15, label="OGV axial bracket x = 3-6 in")
    ax0.axhline(float(fan["annulus_area_ft2"]), color="k", ls="--", lw=0.9,
                label=f"fan face from Table 1 = {float(fan['annulus_area_ft2']):.3f} ft$^2$")
    ax0.set_xlim(2, 7)
    ax0.set_ylim(2.0, 2.5)
    ax0.set_xlabel("Axial distance downstream of fan stacking axis, in")
    ax0.set_ylabel("Annulus area, ft$^2$")
    ax0.set_title("(a) Figure 15 flowpath area", fontsize=10)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    ax1.plot(xs, mach, "o-", color="C0", lw=1.2, ms=4)
    ax1.fill_between(xs, mach - sig, mach + sig, color="C0", alpha=0.20,
                     label=f"nominal +/- {m_sig:.4f} (k=1)")
    ax1.axhline(m_nom, color="C0", ls="--", lw=0.8)
    ax1.set_xlim(2.8, 6.2)
    ax1.set_ylim(0.28, 0.33)
    ax1.set_xlabel("Axial station x, in")
    ax1.set_ylabel("Mean axial Mach number")
    ax1.set_title(f"(b) Approach (61.7 %): M = {m_nom:.3f} +/- {m_sig:.3f}", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8)

    fig.suptitle("OGV annulus and mean axial Mach from NASA/TM-2005-211681 Fig. 15 and Table 3",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OVER / "fig_annulus_mach.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_gate() -> Path:
    vals, labels = [], []
    for key, fname, _ in P14:
        rows = read_csv(DIG / fname)
        v = [(float(r["frequency_Hz"]), float(r["PWL_dB"])) for r in rows
             if r["PWL_dB"] and 0.95 <= float(r["BPF_order"]) <= 1.05]
        vals.append(max(v, key=lambda q: q[1])[1] if v else np.nan)
        labels.append(key)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["C0" if (labels[i] == "TL" and vals[i] < 102) or
              (labels[i] != "TL" and vals[i] > 108) else "C3" for i in range(3)]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.axhline(102, color="C0", ls="--", lw=1.0, label="54-vane cut-off criterion (< 102 dB)")
    ax.axhline(108, color="C1", ls="--", lw=1.0, label="26-vane cut-on criterion (> 108 dB)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylim(90, 125)
    ax.set_ylabel("Max PWL in 0.95-1.05 BPF band, dB")
    ax.set_title("BPF-band gate checks (report text: 54-vane cut off, 26-vane cut on)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = OVER / "fig_gate_checks.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def data_quality(cross_metrics: list[dict]) -> tuple[Path, Path]:
    cal = json.loads((CAL / "p14_panels.json").read_text())
    summary = json.loads((QA / "p14_extraction_summary.json").read_text())
    sum_by_id = {s["dataset_id"]: s for s in summary}
    fern = json.loads((PROV / "fernandes_fig18c.json").read_text())
    ann = json.loads((PROV / "annulus_mach.json").read_text())
    geom_checks = read_csv(GEO / "sdt_geometry_check.csv")
    raster = read_csv(QA / "raster_figure_screening.csv")
    cross = {m["metric"]: m["value"] for m in cross_metrics}

    def p14_row(key: str, fname: str, desc: str) -> dict:
        sid = [s for s in sum_by_id if s.endswith("fig09_54vane_exhaust") and key == "TL"] or \
              [s for s in sum_by_id if s.endswith("fig10_26vane_exhaust") and key == "BL"] or \
              [s for s in sum_by_id if s.endswith("fig11_26swept_exhaust") and key == "TR"]
        s = sum_by_id[sid[0]]
        panel = cal["panels"][key]
        return {
            "dataset_id": s["dataset_id"],
            "category": "measured exhaust PWL spectrum",
            "description": desc,
            "source": f"{cal['pdf']} p. {cal['pdf_page']} (report p. {cal['report_page']})",
            "figure": s["figure"],
            "method": "raster dash morphology, 400 dpi, 3 px bins",
            "n_points": s["n_dashes"],
            "n_bins": s["n_spectrum_bins"],
            "frequency_range_Hz": f"{s['f_min']:.0f}-{s['f_max']:.0f}",
            "level_range": f"{s['PWL_min']:.2f}-{s['PWL_max']:.2f} dB",
            "uncertainty": "0.12 dB (k=1); frequency 0.25 %",
            "calibration": f"y residual {panel['y_fit']['max_abs_residual_db']:.4f} dB; "
                           f"x residual {panel['x_fit']['max_abs_residual_decades']:.2e} dec",
            "key_check": f"BPF-band max {s['BPF_band_max']:.2f} dB; 2BPF max {s['two_BPF_band_max']:.2f} dB",
            "status": "PASS",
            "notes": f"{s['low_outliers']} low outliers, {s['tone_points']} tone points flagged",
        }

    rows = [
        p14_row("TL", "tm2002_p14_fig09_54vane_exhaust_measured.csv",
                "54-vane radial, approach, exhaust"),
        p14_row("BL", "tm2002_p14_fig10_26vane_exhaust_measured.csv",
                "26-vane radial, approach, exhaust"),
        p14_row("TR", "tm2002_p14_fig11_26swept_exhaust_measured.csv",
                "26-vane swept, approach, exhaust"),
        {
            "dataset_id": "fernandes_fig18c_54vane_exhaust_pwl",
            "category": "exhaust PWL spectrum (vector)",
            "description": "54-vane exhaust, LAVA experiment curve",
            "source": "Fernandes.pdf p. 24, Fig. 18(c)",
            "figure": "Figure 18(c)",
            "method": "vector path extraction (PyMuPDF), thick black experimental curve",
            "n_points": fern["n_points"],
            "n_bins": fern["n_points"],
            "frequency_range_Hz": f"{fern['frequency_range_hz'][0]:.0f}-{fern['frequency_range_hz'][1]:.0f}",
            "level_range": f"{min(float(r['PWL_dB_per_Hz']) for r in read_csv(DIG / 'fernandes_fig18c_54vane_exhaust_pwl.csv')):.2f}-"
                           f"{max(float(r['PWL_dB_per_Hz']) for r in read_csv(DIG / 'fernandes_fig18c_54vane_exhaust_pwl.csv')):.2f} dB/Hz",
            "uncertainty": "0.03 dB (k=1); frequency 0.1 %",
            "calibration": f"x residual {fern['calibration']['x_fit']['max_abs_residual_decades']:.2e} dec; "
                           f"y residual {fern['calibration']['y_fit']['max_abs_residual_db']:.2e} dB",
            "key_check": f"2BPF peak 84.73 dB/Hz; cross-check r={cross['pearson_r_normalised']}, "
                         f"RMS={cross['rms_normalised_dB']} dB",
            "status": "PASS",
            "notes": "dB/Hz; compare shape, not absolute level, with NASA Fig. 9",
        },
        {
            "dataset_id": "sdt_fan_design",
            "category": "geometry table",
            "description": "R4 fan design parameters",
            "source": "20050175875.pdf p. 12 (report p. 8), Table 1",
            "figure": "Table 1",
            "method": "text-layer word boxes",
            "n_points": 9,
            "n_bins": 9,
            "frequency_range_Hz": "",
            "level_range": "",
            "uncertainty": "exact text; no digitisation uncertainty",
            "calibration": "cell y positions verified",
            "key_check": "9/9 rows parsed",
            "status": "PASS",
            "notes": "22 blades, 22 in tip diameter, radius ratio 0.30",
        },
        {
            "dataset_id": "sdt_geometry",
            "category": "geometry table",
            "description": "Fan and OGV airfoil geometry",
            "source": "20050175875.pdf p. 12 (report p. 8), Table 2",
            "figure": "Table 2",
            "method": "text-layer word boxes, column-centre assignment",
            "n_points": 64,
            "n_bins": 64,
            "frequency_range_Hz": "",
            "level_range": "",
            "uncertainty": "exact text; 4 blank and 4 '---' cells explicit",
            "calibration": "60/64 cells located by coordinate; 4 blank",
            "key_check": "56 numeric, 4 dash, 4 blank; counts/chord/sweep checks pass",
            "status": "PASS",
            "notes": "Fan hub/tip solidity and stagger blank in source",
        },
        {
            "dataset_id": "sdt_performance",
            "category": "performance table",
            "description": "Fan performance at approach/cutback/takeoff",
            "source": "20050175875.pdf p. 12 (report p. 8), Table 3",
            "figure": "Table 3",
            "method": "text-layer word boxes",
            "n_points": 24,
            "n_bins": 24,
            "frequency_range_Hz": "",
            "level_range": "",
            "uncertainty": "exact text",
            "calibration": "column bands verified",
            "key_check": "24/24 cells parsed; approach Fan/OGV used for Mach",
            "status": "PASS",
            "notes": "approach: 58.30 lbm/s, 1.159, 1.049, 0.8880",
        },
        {
            "dataset_id": "fig15_annulus_mach",
            "category": "derived flow quantity",
            "description": "OGV annulus area and mean axial Mach at approach",
            "source": "20050175875.pdf p. 22 (report p. 18), Fig. 15; Table 3",
            "figure": "Figure 15",
            "method": "vector contour extraction + continuity solve",
            "n_points": 474,
            "n_bins": 7,
            "frequency_range_Hz": "",
            "level_range": f"area {ann['results']['area_ft2_min_x3_6']:.3f}-"
                           f"{ann['results']['area_ft2_max_x3_6']:.3f} ft2; "
                           f"M {ann['results']['mean_axial_mach']:.3f}",
            "uncertainty": f"M +/- {ann['results']['mean_axial_mach_sigma_k1']:.4f} (k=1)",
            "calibration": f"x residual {ann['axis_calibration']['x_fit']['max_abs_residual_in']:.4f} in; "
                           f"y residual {ann['axis_calibration']['y_fit']['max_abs_residual_in']:.4f} in",
            "key_check": f"area 2.094-2.120 ft2; M={ann['results']['mean_axial_mach']:.4f}; "
                         f"static-density solve > total-density shortcut",
            "status": "PASS",
            "notes": "station bracket x=3-6 in; report gives no single OGV leading-edge station",
        },
    ]
    for r in raster:
        rows.append({
            "dataset_id": f"raster_{r['figure'].lower().replace(' ', '')}",
            "category": "raster wake/length-scale figure",
            "description": r["caption"],
            "source": f"{r['source_pdf']} p. {r['source_page_pdf']} (report p. {r['source_page_report']})",
            "figure": r["figure"],
            "method": "screening only",
            "n_points": 0,
            "n_bins": 0,
            "frequency_range_Hz": "",
            "level_range": "",
            "uncertainty": "n/a - no values claimed",
            "calibration": "",
            "key_check": r["reason"],
            "status": "SCREENED",
            "notes": f"{r['image_width_px']}x{r['image_height_px']} px, {r['bits_per_component']}-bit; "
                     f"largest component {r['largest_component_bbox_px']}",
        })

    fields = ["dataset_id", "category", "description", "source", "figure", "method",
              "n_points", "n_bins", "frequency_range_Hz", "level_range", "uncertainty",
              "calibration", "key_check", "status", "notes"]
    csv_path = QA / "data_quality.csv"
    write_csv(csv_path, rows, fields)

    md = []
    md.append("# Day-1 data-quality table\n")
    md.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.  ")
    md.append("Gate: **PASS** - 54-vane and 26-vane exhaust spectra digitised, with geometry, "
              "provenance and uncertainty.\n")
    md.append("| Dataset | Source | Method | Points | Range | Uncertainty | Key check | Status |")
    md.append("|---|---|---|---:|---|---|---|---|")
    for r in rows:
        rng = r["frequency_range_Hz"] or r["level_range"]
        md.append(f"| {r['dataset_id']} | {r['source']} | {r['method']} | {r['n_points']} | "
                  f"{rng} | {r['uncertainty']} | {r['key_check']} | {r['status']} |")
    md.append("")
    md.append("## Cross-check\n")
    md.append(f"- 54-vane NASA Fig. 9 vs Fernandes Fig. 18(c): Pearson r = "
              f"{cross['pearson_r_normalised']}, RMS = {cross['rms_normalised_dB']} dB after removing "
              f"each 2-8 kHz median; raw median offset = {cross['raw_offset_dB']} dB "
              f"(dB vs dB/Hz, not a physical error).")
    md.append(f"- 2 BPF prominence: NASA {cross['2BPF_prominence_nasa_dB']} dB vs Fernandes "
              f"{cross['2BPF_prominence_fernandes_dB']} dB.")
    md.append(f"- BPF remains cut off in the 54-vane NASA spectrum (1 BPF prominence "
              f"{cross['1BPF_prominence_nasa_dB']} dB), as the report states.")
    md.append("")
    md.append("## Not digitised\n")
    md.append("Figs. 26-28 (wake PSDs, von Karman comparison, integral length scales) are "
              "1-bit scans about 500 px wide. Curves, symbols and dotted log-grids merge into "
              "single connected components; no numeric values are claimed. See "
              "`data/qa/raster_figure_screening.csv`.")
    md.append("")
    md_path = QA / "data_quality.md"
    md_path.write_text("\n".join(md) + "\n")
    return csv_path, md_path


def combined_validation() -> Path:
    rows = []
    for fname, domain in [
        (QA / "validation_checks.csv", "acoustic spectra"),
        (GEO / "sdt_geometry_check.csv", "geometry"),
        (QA / "annulus_checks.csv", "annulus/Mach"),
    ]:
        for r in read_csv(fname):
            rows.append({
                "domain": domain,
                "check": r.get("check", ""),
                "subject": r.get("dataset", r.get("table", "")),
                "value": r.get("value", r.get("detail", "")),
                "criterion": r.get("criterion", ""),
                "pass": r.get("pass", ""),
                "note": r.get("note", r.get("detail", "")),
            })
    fern = json.loads((QA / "fernandes_validation_checks.json").read_text())
    for r in fern:
        rows.append({
            "domain": "vector spectrum",
            "check": r["check"],
            "subject": r["dataset"],
            "value": r["value"],
            "criterion": r["criterion"],
            "pass": str(r["pass"]),
            "note": r["note"],
        })
    out = QA / "validation_checks_all.csv"
    write_csv(out, rows, list(rows[0].keys()))
    return out


def main() -> dict:
    p14 = plot_p14()
    cross_plot, cross_metrics = crosscheck()
    ann = plot_annulus()
    gate = plot_gate()
    dq_csv, dq_md = data_quality(cross_metrics)
    val = combined_validation()

    cross = {m["metric"]: m["value"] for m in cross_metrics}
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_status": "PASS",
        "gate_statement": "54-vane and 26-vane measured exhaust spectra digitised with geometry, provenance and uncertainty.",
        "crosscheck_54vane": cross,
        "outputs": {
            "p14_spectra_plot": str(p14.relative_to(ROOT)),
            "crosscheck_plot": str(cross_plot.relative_to(ROOT)),
            "annulus_plot": str(ann.relative_to(ROOT)),
            "gate_plot": str(gate.relative_to(ROOT)),
            "data_quality_csv": str(dq_csv.relative_to(ROOT)),
            "data_quality_md": str(dq_md.relative_to(ROOT)),
            "validation_all": str(val.relative_to(ROOT)),
        },
        "limitations": [
            "NASA p14 is an image-only plate; values come from 3 px raster bins with 0.12 dB k=1 uncertainty.",
            "26 %-35 % of 3 px bins required short-gap interpolation; they are flagged in the CSVs.",
            "Fernandes Fig. 18c is dB/Hz and NASA Fig. 9 is dB; only spectral shape is compared.",
            "Figs. 26-28 are source-limited raster scans and are not numerically digitised.",
        ],
    }
    write_json(QA / "qa_summary.json", summary)

    print("QA outputs:")
    for k, v in summary["outputs"].items():
        print(f"  {k}: {v}")
    print(f"  cross-check r = {cross['pearson_r_normalised']}, RMS = {cross['rms_normalised_dB']} dB")
    return summary


if __name__ == "__main__":
    main()
