#!/usr/bin/env python3
"""Standalone plots of digitized data (no source image overlays).

Generates clean publication-ready figures from the digitized CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DIG, GEO, QA, ROOT, read_csv, write_json

BPF_HZ = 7808.0 * 22.0 / 60.0  # 2862.933... Hz

OUT = QA / "standalone"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    rows = read_csv(path)
    return rows


# =====================================================================
# A1–A3: NASA TM-2002 p.14 exhaust spectra (measured + masked)
# =====================================================================
def plot_nallasamy_p14_spectra():
    """Standalone 3-panel plot of the three exhaust spectra."""
    datasets = [
        ("TL", "tm2002_p14_fig09_54vane_exhaust_measured.csv",
         "tm2002_p14_fig09_54vane_exhaust_masked.csv",
         "54-vane radial — Approach (BPF cut-off)"),
        ("BL", "tm2002_p14_fig10_26vane_exhaust_measured.csv",
         "tm2002_p14_fig10_26vane_exhaust_masked.csv",
         "26-vane radial — Approach (BPF cut-on)"),
        ("TR", "tm2002_p14_fig11_26swept_exhaust_measured.csv",
         "tm2002_p14_fig11_26swept_exhaust_masked.csv",
         "26-vane swept — Approach (BPF cut-on)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (key, meas_f, mask_f, title) in zip(axes, datasets):
        # Measured (raw with tones)
        meas = load_csv(DIG / meas_f)
        f_m = np.array([float(r["frequency_Hz"]) for r in meas])
        y_m = np.array([float(r["PWL_dB"]) if r["PWL_dB"] else np.nan for r in meas])
        interp = np.array([r["interpolated"] == "1" for r in meas])

        good = np.isfinite(y_m)
        ax.semilogx(f_m[good], y_m[good], "-", color="C0", lw=1.0, label="Measured (raw)")
        if interp.any():
            ax.semilogx(f_m[interp & good], y_m[interp & good], "o", ms=2.5, mfc="none",
                        mec="C1", mew=0.8, label="Interpolated gap")

        # Masked (broadband envelope)
        mask = load_csv(DIG / mask_f)
        f_k = np.array([float(r["frequency_Hz"]) for r in mask])
        y_k = np.array([float(r["PWL_dB"]) if r["PWL_dB"] else np.nan for r in mask])
        good_k = np.isfinite(y_k)
        ax.semilogx(f_k[good_k], y_k[good_k], "-", color="C2", lw=1.2, label="Masked envelope")

        # BPF markers
        for k in range(1, 9):
            fq = k * BPF_HZ
            if fq < f_m.max():
                ax.axvline(fq, color="k", ls=":", lw=0.5, alpha=0.4)
                if k <= 3:
                    ax.text(fq, ax.get_ylim()[1] - 2, f"{k}×BPF", ha="center", va="top",
                            fontsize=7, alpha=0.7)

        ax.set_ylim(70, 120)
        ax.set_ylabel("Exhaust PWL, dB")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.10)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Frequency, Hz")
    axes[-1].set_xlim(1e3, 1e5)
    fig.suptitle("NASA/TM-2002-211798 p.14 — Measured Exhaust Spectra (Approach, 7808 rpm)\n"
                 "Digitized from raster dash morphology; masked = prominence-gated tone removal",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "nallasamy_p14_spectra_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# A5: Fernandes Fig 18(c) 54-vane exhaust PWL
# =====================================================================
def plot_fernandes_fig18c():
    rows = load_csv(DIG / "fernandes_fig18c_54vane_exhaust_pwl.csv")
    f = np.array([float(r["frequency_Hz"]) for r in rows])
    y = np.array([float(r["PWL_dB_per_Hz"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(f, y, "-", color="C0", lw=1.2)
    for k in range(1, 9):
        fq = k * BPF_HZ
        if fq < f.max():
            ax.axvline(fq, color="k", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Exhaust PWL, dB/Hz")
    ax.set_xlim(800, 3e4)
    ax.set_ylim(55, 95)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.set_title("Fernandes Fig. 18(c) — 54-vane Exhaust PWL (LAVA Experiment)\n"
                 "Vector path extraction; 7808 rpm, 865–28 660 Hz, 433 points", fontsize=10)
    fig.tight_layout()
    out = OUT / "fernandes_fig18c_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# B1: Podboy Fig 28 — Integral Length Scales
# =====================================================================
def plot_podboy_fig28():
    rows = load_csv(DIG / "podboy_fig28_integral_length_scales.csv")
    # Group by component and station (HW1/HW2)
    components = ["streamwise", "upwash", "spanwise"]
    stations = ["HW1", "HW2"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for i, comp in enumerate(components):
        for j, st in enumerate(stations):
            ax = axes[i, j]
            subset = [r for r in rows if r["component"] == comp and r["station"] == st]
            if not subset:
                ax.set_visible(False)
                continue
            x = np.array([float(r["radius_in"]) for r in subset])
            y = np.array([float(r["length_scale_in"]) for r in subset])
            sort_idx = np.argsort(x)
            ax.plot(x[sort_idx], y[sort_idx], "-", color="C0" if st == "HW1" else "C1", lw=1.2)
            ax.set_title(f"{comp.capitalize()} — {st}", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(0, 12)
            if i == 2:
                ax.set_xlabel("Radius, in")
            if j == 0:
                ax.set_ylabel("Integral length scale, in")

    fig.suptitle("Podboy Fig. 28 — Integral Length Scales\n"
                 "3 turbulence components × HW1 (line) / HW2 (asterisk proxy)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "podboy_fig28_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# B2: Podboy Fig 26 — Wake PSDs (6 curves)
# =====================================================================
def plot_podboy_fig26_psd():
    rows = load_csv(DIG / "podboy_fig26_psd.csv")
    spans = ["97", "81", "25"]
    stations = ["HW1", "HW2"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for i, span in enumerate(spans):
        for j, st in enumerate(stations):
            ax = axes[i, j]
            subset = [r for r in rows if r["span_percent"] == span and r["station"] == st]
            if not subset:
                ax.set_visible(False)
                continue
            f = np.array([float(r["frequency_Hz"]) for r in subset])
            # log10_PSD is log10(PSD_linear); convert to dB: 10 * log10(PSD)
            psd_db = np.array([10.0 * float(r["log10_PSD"]) for r in subset])
            sort_idx = np.argsort(f)
            ax.semilogx(f[sort_idx], psd_db[sort_idx], "-", color="C0" if st == "HW1" else "C1", lw=1.0)
            ax.set_title(f"{span}% span — {st}", fontsize=9)
            ax.grid(True, which="major", alpha=0.25)
            ax.grid(True, which="minor", alpha=0.10)
            ax.set_xlim(100, 5e4)
            if i == 2:
                ax.set_xlabel("Frequency, Hz")
            if j == 0:
                ax.set_ylabel("PSD, dB (10×log₁₀)")

    fig.suptitle("Podboy Fig. 26 — Wake Upwash PSDs (6 curves)\n"
                 "3 span stations × HW1 (black) / HW2 (red)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "podboy_fig26_psd_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# B2: Podboy Fig 26 — Variance Integrals (bar chart)
# =====================================================================
def plot_podboy_fig26_variance():
    rows = load_csv(DIG / "podboy_fig26_variance_integrals.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_pos = np.arange(len(rows))
    colors = {"integral_measured_full": "C0", "integral_20k_truncated": "C1", "integral_extrapolated_inf": "C2"}
    labels = {"integral_measured_full": "Measured (full)", "integral_20k_truncated": "Truncated at 20 kHz",
              "integral_extrapolated_inf": "Extrapolated tail"}

    for idx, row in enumerate(rows):
        for var_type in ["integral_measured_full", "integral_20k_truncated", "integral_extrapolated_inf"]:
            val = float(row[var_type])
            ax.bar(x_pos[idx] + (list(colors).index(var_type) - 1) * 0.22,
                   val, width=0.2, color=colors[var_type],
                   label=labels[var_type] if idx == 0 else "", alpha=0.85)
            ax.text(x_pos[idx] + (list(colors).index(var_type) - 1) * 0.22,
                    val + 0.01, f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r['span_percent']}% {r['station']}" for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Variance integral, in$^2$")
    ax.set_title("Podboy Fig. 26 — Wake Upwash Variance Integrals\n"
                 "Three variants: measured full / 20 kHz truncated / extrapolated",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = OUT / "podboy_fig26_variance_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# B3: Podboy Fig 18 — Max Tangential Velocity
# =====================================================================
def plot_podboy_fig18():
    # Points (8 points: 4 speeds × 2 stations)
    points = load_csv(DIG / "podboy_fig18_points.csv")
    # Full curves
    curves = load_csv(DIG / "podboy_fig18_curve.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: 8 discrete points at 61.7% span
    speeds = sorted(set(float(r["speed_percent_calibrated"]) for r in points))
    stations = ["LDV1", "LDV2"]
    for st in stations:
        subset = [r for r in points if r["station"] == st]
        x = np.array([float(r["speed_percent_calibrated"]) for r in subset])
        y = np.array([float(r["max_tangential_velocity_ft_s"]) for r in subset])
        sort_idx = np.argsort(x)
        ax0.plot(x[sort_idx], y[sort_idx], "o-",
                 color="C0" if st == "LDV1" else "C1", lw=1.5, ms=6, label=st)
    ax0.set_xlabel("Fan speed, %")
    ax0.set_ylabel("Max tangential velocity, ft/s")
    ax0.set_title("B3 Primary: 61.7% span — Max tangential velocity", fontsize=10)
    ax0.grid(True, alpha=0.25)
    ax0.legend()

    # Right: full curves (v_tang vs speed) at each station
    # Note: the curves are v_tang vs radius at fixed speeds; we plot all points
    for st in stations:
        subset = [r for r in curves if r["station"] == st]
        if not subset:
            continue
        x_vals = np.array([float(r["speed_percent"]) for r in subset])
        y_vals = np.array([float(r["max_tangential_velocity_ft_s"]) for r in subset])
        sort_idx = np.argsort(x_vals)
        ax1.plot(x_vals[sort_idx], y_vals[sort_idx], "-", lw=1.0,
                 color="C0" if st == "LDV1" else "C1", label=f"{st} (curve)")

    ax1.set_xlabel("Fan speed, %")
    ax1.set_ylabel("Tangential velocity, ft/s")
    ax1.set_title("Digitized curves at 61.7% span (both stations)", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    fig.suptitle("Podboy Fig. 18 — Max Tangential Turbulent Velocity\n"
                 "4 speeds × 2 LDV stations (3.12\" & 6.49\" downstream)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "podboy_fig18_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# Geometry: SDT Fan Design (Table 1)
# =====================================================================
def plot_sdt_fan_design():
    rows = load_csv(GEO / "sdt_fan_design.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    cell_text = [[r["parameter"], r["value"]] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=["Parameter", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("NASA/TM-2005-211681 Table 1 — R4 Fan Design Parameters", fontsize=11, pad=20)
    fig.tight_layout()
    out = OUT / "sdt_fan_design_table.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# Geometry: Annulus Area + Mach
# =====================================================================
def plot_annulus_mach():
    am = load_csv(ROOT / "data/derived/annulus_area_mach.csv")
    station = [r for r in am if r["x_in"] and r["x_in"] not in ("3-6",)]
    xs = np.array([float(r["x_in"]) for r in station])
    mach = np.array([float(r["mean_axial_mach"]) for r in station])
    sig = np.array([float(r["mach_sigma"]) for r in station])
    nom = [r for r in am if r["station"] == "mean x=3-6 in"][0]
    fan = [r for r in am if r["station"].startswith("fan face")][0]
    m_nom, m_sig = float(nom["mean_axial_mach"]), float(nom["mach_sigma"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Annulus area
    audit = load_csv(ROOT / "data/derived/flowpath_audit.csv")
    x = np.array([float(r["x_in"]) for r in audit])
    area = np.array([float(r["annulus_area_ft2"]) for r in audit])
    ax0.plot(x, area, "o-", color="C0", lw=1.3, ms=4)
    ax0.axvspan(3.0, 6.0, color="C1", alpha=0.15, label="OGV bracket x = 3–6 in")
    ax0.axhline(float(fan["annulus_area_ft2"]), color="k", ls="--", lw=0.9,
                label=f"Fan face = {float(fan['annulus_area_ft2']):.3f} ft²")
    ax0.set_xlabel("Axial distance downstream of fan stacking axis, in")
    ax0.set_ylabel("Annulus area, ft²")
    ax0.set_title("(a) Fig. 15 Flowpath Area", fontsize=10)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    # Mean axial Mach
    ax1.plot(xs, mach, "o-", color="C0", lw=1.2, ms=4)
    ax1.fill_between(xs, mach - sig, mach + sig, color="C0", alpha=0.20,
                     label=f"±{m_sig:.4f} (k=1)")
    ax1.axhline(m_nom, color="C0", ls="--", lw=0.8)
    ax1.set_xlabel("Axial station x, in")
    ax1.set_ylabel("Mean axial Mach number")
    ax1.set_title(f"(b) Approach (61.7%): M = {m_nom:.3f} ± {m_sig:.3f}", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    fig.suptitle("OGV Annulus Area & Mean Axial Mach — NASA/TM-2005-211681 Fig. 15 + Table 3",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / "annulus_mach_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# =====================================================================
# Reference text tables
# =====================================================================
def plot_reference_tables():
    nallasamy = load_csv(GEO / "nallasamy_table1.csv")
    podboy = load_csv(GEO / "podboy_survey_locations.csv")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.5))

    # Nallasamy Table 1
    ax0.axis("off")
    cell_text = [[r["power_setting"], r["corrected_fan_rpm"], r["corrected_tip_speed_fps"]]
                 for r in nallasamy]
    table0 = ax0.table(cellText=cell_text,
                       colLabels=["Power Setting", "Corrected Fan RPM", "Tip Speed (fps)"],
                       loc="center", cellLoc="center")
    table0.auto_set_font_size(False)
    table0.set_fontsize(10)
    table0.scale(1, 1.8)
    ax0.set_title("Nallasamy TM-2002-211798 Table 1\nFan Operating Conditions (Approach/Cutback/Takeoff)",
                  fontsize=10, pad=10)

    # Podboy stations
    ax1.axis("off")
    cell_text = [[r["station"], r["technique"], r["axial_distance_in"], r["axial_distance_cm"]]
                 for r in podboy]
    table1 = ax1.table(cellText=cell_text,
                       colLabels=["Station", "Technique", "Axial Dist. (in)", "Axial Dist. (cm)"],
                       loc="center", cellLoc="center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.8)
    ax1.set_title("Podboy TM-2003-212329 pp. 6–7\nHW/LDV Survey Stations (downstream of tip TE)",
                  fontsize=10, pad=10)

    fig.suptitle("Reference Text Transcriptions (PDF text layer)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = OUT / "reference_tables_standalone.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    print("Generating standalone plots...")
    outputs = []

    outputs.append(("NASA p14 spectra", plot_nallasamy_p14_spectra()))
    outputs.append(("Fernandes Fig 18c", plot_fernandes_fig18c()))
    outputs.append(("Podboy Fig 28 length scales", plot_podboy_fig28()))
    outputs.append(("Podboy Fig 26 PSDs", plot_podboy_fig26_psd()))
    outputs.append(("Podboy Fig 26 variance integrals", plot_podboy_fig26_variance()))
    outputs.append(("Podboy Fig 18 max tangential", plot_podboy_fig18()))
    outputs.append(("SDT fan design table", plot_sdt_fan_design()))
    outputs.append(("Annulus + Mach", plot_annulus_mach()))
    outputs.append(("Reference tables", plot_reference_tables()))

    print("\nGenerated:")
    for name, path in outputs:
        print(f"  {name}: {path.relative_to(ROOT)}")

    summary = {
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(OUT.relative_to(ROOT)),
        "plots": [str(p.relative_to(ROOT)) for _, p in outputs],
    }
    write_json(OUT / "standalone_plots.json", summary)
    print(f"\nManifest: {OUT / 'standalone_plots.json'}")


if __name__ == "__main__":
    main()