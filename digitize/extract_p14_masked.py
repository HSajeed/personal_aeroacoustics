"""Build broadband-envelope spectra for the three Nallasamy exhaust panels.

The original rule proposed a +/-3 % mask around every k*BPF.  That is wrong:
the total mask width is 0.06*k*BPF while the tone spacing is 1*BPF, so at
k ~ 17 the masks tile the whole axis.  This script uses the replacement
decided with the user:

    absolute half-width 0.1*BPF, applied only where the local peak is
    prominent above the broadband floor (peak - floor >= 3 dB).

Peak:    maximum PWL_max_dB within +/-0.15*BPF of k*BPF.
Floor:   rolling 30th percentile of PWL_dB over +/-0.3 decade.
Mask:    all bins within +/-0.1*BPF of k*BPF when the peak is prominent.
         Above k ~ 10 this is narrower than one bin, so only the nearest bin
         is removed; that resolution limit is recorded in the summary.

The broadband envelope is a rolling 25th percentile of the unmasked bins
over +/-0.2 decade, linearly interpolated in log10(f) across the masks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import DIG, OVER, QA, ROOT, read_csv, write_csv, write_json

BPF_HZ = 22.0 * 7808.0 / 60.0  # 2862.933... Hz
PROMINENCE_DB = 3.0
PEAK_HALF_BPF = 0.15
MASK_HALF_BPF = 0.10
FLOOR_HALF_DEC = 0.30
ENVELOPE_HALF_DEC = 0.15

DATASETS = [
    {
        "dataset_id": "tm2002_p14_fig09_54vane_exhaust",
        "figure": "Figure 9",
        "panel": "TL",
        "description": "54 radial vanes, approach, exhaust",
        "infile": "tm2002_p14_fig09_54vane_exhaust_measured.csv",
        "outfile": "tm2002_p14_fig09_54vane_exhaust_masked.csv",
        "tonefile": "tm2002_p14_fig09_54vane_exhaust_tones.csv",
    },
    {
        "dataset_id": "tm2002_p14_fig10_26vane_exhaust",
        "figure": "Figure 10",
        "panel": "BL",
        "description": "26 radial vanes, approach, exhaust",
        "infile": "tm2002_p14_fig10_26vane_exhaust_measured.csv",
        "outfile": "tm2002_p14_fig10_26vane_exhaust_masked.csv",
        "tonefile": "tm2002_p14_fig10_26vane_exhaust_tones.csv",
    },
    {
        "dataset_id": "tm2002_p14_fig11_26swept_exhaust",
        "figure": "Figure 11",
        "panel": "TR",
        "description": "26 swept vanes, approach, exhaust",
        "infile": "tm2002_p14_fig11_26swept_exhaust_measured.csv",
        "outfile": "tm2002_p14_fig11_26swept_exhaust_masked.csv",
        "tonefile": "tm2002_p14_fig11_26swept_exhaust_tones.csv",
    },
]


def rolling_percentile(f: np.ndarray, v: np.ndarray, half_dec: float,
                       pct: float) -> np.ndarray:
    """Rolling percentile in log10(f), evaluated at each sample."""
    lf = np.log10(f)
    out = np.full_like(v, np.nan, dtype=float)
    for i, x in enumerate(lf):
        sel = np.abs(lf - x) <= half_dec
        vals = v[sel]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[i] = np.percentile(vals, pct)
    return out


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    f, p, pmax = [], [], []
    for r in rows:
        if not r["PWL_dB"]:
            continue
        f.append(float(r["frequency_Hz"]))
        p.append(float(r["PWL_dB"]))
        pmax.append(float(r["PWL_max_dB"]) if r["PWL_max_dB"] else float(r["PWL_dB"]))
    return np.asarray(f), np.asarray(p), np.asarray(pmax)


def detect_and_mask(f: np.ndarray, p: np.ndarray, pmax: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    floor = rolling_percentile(f, p, FLOOR_HALF_DEC, 30.0)
    masked = np.zeros(len(f), dtype=bool)
    tones = []
    kmax = int(np.floor(f.max() / BPF_HZ))
    for k in range(1, kmax + 1):
        f0 = k * BPF_HZ
        win = np.abs(f - f0) <= PEAK_HALF_BPF * BPF_HZ
        if not win.any():
            continue
        j = int(np.argmax(np.where(win, pmax, -np.inf)))
        peak = float(pmax[j])
        fl = float(np.median(floor[win]))
        prom = peak - fl
        tone = {
            "k": k,
            "f_harmonic_Hz": f0,
            "f_peak_Hz": float(f[j]),
            "peak_PWL_dB": peak,
            "floor_PWL_dB": fl,
            "prominence_dB": prom,
            "detected": int(prom >= PROMINENCE_DB),
            "mask_lo_Hz": f0 - MASK_HALF_BPF * BPF_HZ,
            "mask_hi_Hz": f0 + MASK_HALF_BPF * BPF_HZ,
            "n_bins_masked": 0,
            "note": "",
        }
        if prom >= PROMINENCE_DB:
            m = np.abs(f - f0) <= MASK_HALF_BPF * BPF_HZ
            if not m.any():
                # sub-bin mask: remove the nearest bin so the tone is not left in
                m[np.argmin(np.abs(f - f0))] = True
                tone["note"] = "sub-bin mask; nearest bin removed"
            tone["n_bins_masked"] = int(m.sum())
            masked |= m
        tones.append(tone)
    return tones, masked, floor


def envelope(f: np.ndarray, p: np.ndarray, masked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadband envelope from unmasked bins, interpolated across masks."""
    env = np.full(len(f), np.nan)
    good = ~masked & np.isfinite(p)
    if good.sum() < 5:
        return env, good
    env_good = rolling_percentile(f[good], p[good], ENVELOPE_HALF_DEC, 50.0)
    lf = np.log10(f)
    env = np.interp(lf, lf[good], env_good)
    return env, good


def make_plot(ds: dict, f: np.ndarray, p: np.ndarray, pmax: np.ndarray,
              envelope_db: np.ndarray, masked: np.ndarray, tones: list[dict],
              outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(f, p, color="0.55", lw=0.8, label="measured (3 px bins)")
    ax.plot(f[~masked], p[~masked], ".", color="0.25", ms=3, label="retained")
    ax.plot(f[masked], p[masked], "x", color="crimson", ms=5, label="masked tone bins")
    ax.plot(f, envelope_db, color="tab:blue", lw=2.0, label="broadband envelope")
    for t in tones:
        if t["detected"]:
            ax.axvspan(t["mask_lo_Hz"], t["mask_hi_Hz"], color="crimson", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlim(f.min() * 0.98, f.max() * 1.02)
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("PWL, dB")
    ax.set_title(f"{ds['dataset_id']} — {ds['figure']} ({ds['description']})")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> dict:
    summaries = []
    for ds in DATASETS:
        f, p, pmax = load_spectrum(DIG / ds["infile"])
        tones, masked, floor = detect_and_mask(f, p, pmax)
        env, good = envelope(f, p, masked)

        tone_rows = []
        for t in tones:
            if not t["detected"] and t["k"] > 6:
                continue
            row = {
                "dataset_id": ds["dataset_id"],
                "figure": ds["figure"],
                "panel": ds["panel"],
                **{k: (f"{v:.4f}" if isinstance(v, float) else v)
                   for k, v in t.items() if k != "note"},
                "note": t["note"],
                "source_pdf": "20030003692.pdf",
                "source_page_pdf": 14,
                "source_page_report": 10,
                "method": "prominence-gated absolute +/-0.1 BPF mask",
            }
            tone_rows.append(row)
        if tone_rows:
            write_csv(DIG / ds["tonefile"], tone_rows, list(tone_rows[0].keys()))

        rows = []
        for i in range(len(f)):
            rows.append({
                "dataset_id": ds["dataset_id"],
                "figure": ds["figure"],
                "panel": ds["panel"],
                "frequency_Hz": f"{f[i]:.3f}",
                "BPF_order": f"{f[i] / BPF_HZ:.5f}",
                "PWL_dB": f"{p[i]:.3f}",
                "PWL_max_dB": f"{pmax[i]:.3f}",
                "floor_dB": f"{floor[i]:.3f}",
                "tone_flag": int(masked[i]),
                "envelope_dB": f"{env[i]:.3f}",
                "envelope_valid": int(np.isfinite(env[i])),
                "method": "prominence-gated absolute +/-0.1 BPF mask; "
                          "25th-percentile rolling envelope",
            })
        write_csv(DIG / ds["outfile"], rows, list(rows[0].keys()))
        make_plot(ds, f, p, pmax, env, masked, tones,
                  OVER / f"{ds['dataset_id']}_masked.png")

        detected = [t for t in tones if t["detected"]]
        summaries.append({
            "dataset_id": ds["dataset_id"],
            "figure": ds["figure"],
            "panel": ds["panel"],
            "description": ds["description"],
            "n_bins": int(len(f)),
            "f_min_Hz": float(f.min()),
            "f_max_Hz": float(f.max()),
            "k_max_checked": int(np.floor(f.max() / BPF_HZ)),
            "n_tones_detected": len(detected),
            "k_detected": [t["k"] for t in detected],
            "masked_fraction": float(masked.mean()),
            "max_mask_width_Hz": float(2 * MASK_HALF_BPF * BPF_HZ),
            "max_mask_width_BPF": 2 * MASK_HALF_BPF,
            "bin_width_at_k10_Hz": float(np.interp(10 * BPF_HZ, f, np.gradient(f))),
            "prominence_threshold_dB": PROMINENCE_DB,
            "envelope_min_dB": float(np.nanmin(env)),
            "envelope_max_dB": float(np.nanmax(env)),
            "method": "absolute +/-0.1 BPF, prominence >= 3 dB",
        })
        print(f"{ds['dataset_id']}: {len(f)} bins, {len(detected)} tones "
              f"(k={[t['k'] for t in detected]}), {masked.mean():.1%} bins masked")

    write_json(QA / "p14_masked_summary.json", summaries)
    return {"datasets": summaries}


if __name__ == "__main__":
    main()
