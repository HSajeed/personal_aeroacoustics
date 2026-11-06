"""Derive the OGV annulus area and mean axial Mach number at approach.

Sources
-------
20050175875.pdf (NASA/TM-2005-211681)
  PDF p. 22 / report p. 18, Figure 15: vector flowpath (inner + outer contours)
  PDF p. 12 / report p.  8, Tables 1 and 3: fan annulus and approach performance

Method
------
The Figure 15 axes are vector tick marks, so both axes are fitted from the
actual tick coordinates.  The red flowpath path contains the inner contour
followed by the outer contour; the two are split at the x reversal.  Hub and
tip radii are interpolated at axial stations x = 3--6 in downstream of the
fan stacking axis.  That range brackets the OGV axial extent (pitchline
chords 1.57 in baseline, 3.26 in low-count/low-noise); the report does not
quote a single OGV leading-edge station.

Mean axial Mach follows from continuity with static properties:

    m_dot = rho_t A a_t M / (1 + (gamma-1)/2 M^2)^3,  gamma = 1.4

using the fan exit total pressure and temperature from Table 3 (approach,
Fan/OGV): m_dot = 58.30 lbm/s, p_t2/p_t1 = 1.159, T_t2/T_t1 = 1.049.
Inlet total conditions are standard sea level (101325 Pa, 288.15 K); the
sensitivity table covers +/-1 % inlet pressure and +/-2 K inlet temperature.

Using the total density directly (rho_t A V = m_dot) ignores the static-to-
total density rise and under-predicts M; that value is retained only as an
audit cross-check, not as the reported result.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pymupdf as fitz
from scipy.optimize import brentq

from common import (DER, OVER, PROV, QA, ROOT, render_page, write_csv,
                    write_json, sha256)

PDF = ROOT / "20050175875.pdf"
PDF_PAGE = 22
REPORT_PAGE = 18
FIGURE = "Figure 15"

# Figure 15 frame in PDF points (from the vector frame quad).
FRAME = (107.16, 252.90, 529.38, 420.84)
# Axis limits at the frame edges, from the printed tick labels.
X_LIMITS = (-15.0, 30.0)   # in, axial distance downstream of fan stacking axis
Y_LIMITS = (0.0, 18.0)     # in, radius from model centerline

GAMMA = 1.4
R_AIR = 287.05             # J/kg/K
LBM_TO_KG = 0.45359237
IN2_TO_M2 = 0.00064516
FT2_TO_M2 = 0.09290304

# Table 1 / Table 3 approach inputs (verified by parse_geometry.py).
FAN_TIP_DIAMETER_IN = 22.0
FAN_RADIUS_RATIO = 0.30
M_DOT_LBM_S = 58.30
TOTAL_PRESSURE_RATIO = 1.159
TOTAL_TEMPERATURE_RATIO = 1.049
ADIABATIC_EFFICIENCY = 0.8880
P_T1_PA = 101325.0
T_T1_K = 288.15

STATIONS_IN = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def fit_axes(page) -> dict:
    """Fit both axes from the vector tick marks in the frame drawing."""
    drawings = page.get_drawings()
    frame_dd = drawings[1]
    xticks, yticks = [], []
    for item in frame_dd["items"]:
        if item[0] != "l":
            continue
        p0, p1 = item[1], item[2]
        # x ticks: short vertical strokes just below the bottom frame edge
        if (abs(p0.x - p1.x) < 1e-6 and 2.0 < abs(p0.y - p1.y) < 4.0
                and abs(min(p0.y, p1.y) - FRAME[3]) < 0.5):
            xticks.append(p0.x)
        # y ticks: short horizontal strokes just left of the left frame edge
        if (abs(p0.y - p1.y) < 1e-6 and 2.0 < abs(p0.x - p1.x) < 4.0
                and abs(max(p0.x, p1.x) - FRAME[0]) < 0.5):
            yticks.append(p0.y)
    xticks = sorted(xticks)
    yticks = sorted(yticks)
    if len(xticks) != 10 or len(yticks) != 10:
        raise RuntimeError(f"expected 10 x and 10 y ticks, got {xticks}, {yticks}")

    xvals = np.linspace(X_LIMITS[0], X_LIMITS[1], len(xticks))
    # Pixel y increases downward: the top tick is 18 in, the bottom tick 0 in.
    yvals = np.linspace(Y_LIMITS[1], Y_LIMITS[0], len(yticks))
    xcoef = np.polyfit(xticks, xvals, 1)
    ycoef = np.polyfit(yticks, yvals, 1)
    xres = np.polyval(xcoef, xticks) - xvals
    yres = np.polyval(ycoef, yticks) - yvals
    if np.max(np.abs(xres)) > 0.02 or np.max(np.abs(yres)) > 0.02:
        raise RuntimeError(f"axis fits poor: x {xres}, y {yres}")
    return {
        "frame_pt": list(FRAME),
        "x_ticks_pt": [round(v, 3) for v in xticks],
        "x_tick_in": xvals.tolist(),
        "x_fit": {"slope_in_per_pt": float(xcoef[0]), "intercept_in": float(xcoef[1]),
                  "max_abs_residual_in": float(np.max(np.abs(xres)))},
        "y_ticks_pt": [round(v, 3) for v in yticks],
        "y_tick_in": yvals.tolist(),
        "y_fit": {"slope_in_per_pt": float(ycoef[0]), "intercept_in": float(ycoef[1]),
                  "max_abs_residual_in": float(np.max(np.abs(yres)))},
    }


def contours(page) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return inner and outer contour vertices in inches."""
    drawings = page.get_drawings()
    red = []
    for i, dd in enumerate(drawings):
        if (dd["type"] == "s" and dd.get("color") is not None
                and dd["color"] == (1.0, 0.0, 0.0)):
            nseg = sum(1 for it in dd["items"] if it[0] == "l")
            red.append((nseg, i, dd))
    if not red:
        raise RuntimeError("no red paths found on Figure 15")
    # The flowpath has both hub and tip contours (237 segments).  The nacelle
    # outer line has 144 and the legend sample has 1; select the longest path.
    red.sort(reverse=True)
    nseg, idx, dd = red[0]
    if nseg < 200:
        raise RuntimeError(f"flowpath path too short: {nseg} segments")
    other_red = [{"drawing_index": i, "segments": n, "note": "not used"}
                 for n, i, _ in red[1:]]
    pts = []
    for item in dd["items"]:
        if item[0] == "l":
            pts.extend([item[1], item[2]])
    a = np.array([[q.x, q.y] for q in pts], dtype=float)
    # Split at the large x reversal between the inner and outer contour.
    jumps = np.where(np.diff(a[:, 0]) < -50.0)[0]
    if len(jumps) != 1:
        raise RuntimeError(f"expected one contour reversal, got {jumps}")
    split = int(jumps[0]) + 1
    xf, yf = AXES["x_fit"], AXES["y_fit"]
    def to_in(q):
        return np.c_[np.polyval([xf["slope_in_per_pt"], xf["intercept_in"]], q[:, 0]),
                     np.polyval([yf["slope_in_per_pt"], yf["intercept_in"]], q[:, 1])]
    inner = to_in(a[:split])
    outer = to_in(a[split:])
    info = {
        "drawing_index": idx,
        "n_segments": nseg,
        "n_vertices_total": len(a),
        "split_index": split,
        "n_inner": len(inner),
        "n_outer": len(outer),
        "other_red_paths": other_red,
    }
    return inner, outer, info


def clean_branch(branch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort an x-monotonic branch and average duplicate x values."""
    order = np.argsort(branch[:, 0])
    xs, ys = branch[order, 0], branch[order, 1]
    ux, uy = [], []
    for x in np.unique(xs):
        sel = np.abs(xs - x) < 1e-6
        ux.append(x)
        uy.append(float(np.mean(ys[sel])))
    return np.array(ux), np.array(uy)


def mass_flow_mach(mach: float, area_m2: float, p_t2: float, t_t2: float,
                   gamma: float = GAMMA, r_air: float = R_AIR) -> float:
    """Continuity mass flow for a given axial Mach number (kg/s)."""
    rho_t = p_t2 / (r_air * t_t2)
    a_t = math.sqrt(gamma * r_air * t_t2)
    denom = (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** 3
    return rho_t * area_m2 * a_t * mach / denom


def solve_mach(area_m2: float, m_dot_kg_s: float, p_t2: float, t_t2: float,
               gamma: float = GAMMA) -> float:
    f = lambda m: mass_flow_mach(m, area_m2, p_t2, t_t2, gamma) - m_dot_kg_s
    return float(brentq(f, 1e-4, 0.95, xtol=1e-12, rtol=1e-12))


def mach_inputs(m_dot_lbm_s=M_DOT_LBM_S, pr=TOTAL_PRESSURE_RATIO,
                tr=TOTAL_TEMPERATURE_RATIO, p_t1=P_T1_PA, t_t1=T_T1_K,
                gamma=GAMMA) -> tuple[float, float, float]:
    m_dot = m_dot_lbm_s * LBM_TO_KG
    p_t2 = pr * p_t1
    t_t2 = tr * t_t1
    return m_dot, p_t2, t_t2


AXES: dict = {}


def main() -> dict:
    global AXES
    doc = fitz.open(PDF)
    page = doc[PDF_PAGE - 1]
    AXES = fit_axes(page)
    inner, outer, cinfo = contours(page)
    xi, ri = clean_branch(inner)
    xo, ro = clean_branch(outer)

    # flowpath audit table at 0.25 in spacing over the useful range
    audit = []
    for x in np.arange(2.0, 7.001, 0.25):
        r_hub = float(np.interp(x, xi, ri))
        r_tip = float(np.interp(x, xo, ro))
        area_ft2 = math.pi * (r_tip ** 2 - r_hub ** 2) / 144.0
        audit.append({
            "source_pdf": PDF.name,
            "source_page_pdf": PDF_PAGE,
            "source_page_report": REPORT_PAGE,
            "figure": FIGURE,
            "x_in": f"{x:.2f}",
            "hub_radius_in": f"{r_hub:.4f}",
            "tip_radius_in": f"{r_tip:.4f}",
            "annulus_area_ft2": f"{area_ft2:.4f}",
            "annulus_area_m2": f"{area_ft2 * FT2_TO_M2:.6f}",
        })
    write_csv(DER / "flowpath_audit.csv", audit, list(audit[0].keys()))

    # stations and nominal area
    station_rows = []
    areas = []
    for x in STATIONS_IN:
        r_hub = float(np.interp(x, xi, ri))
        r_tip = float(np.interp(x, xo, ro))
        area_ft2 = math.pi * (r_tip ** 2 - r_hub ** 2) / 144.0
        areas.append((x, r_hub, r_tip, area_ft2))
    area_mean = float(np.mean([a[3] for a in areas]))
    area_min = float(min(a[3] for a in areas))
    area_max = float(max(a[3] for a in areas))

    m_dot, p_t2, t_t2 = mach_inputs()
    mach_nom = solve_mach(area_mean * FT2_TO_M2, m_dot, p_t2, t_t2)
    mach_min_area = solve_mach(area_max * FT2_TO_M2, m_dot, p_t2, t_t2)
    mach_max_area = solve_mach(area_min * FT2_TO_M2, m_dot, p_t2, t_t2)

    # sensitivity cases; each is (label, M)
    sens = [("nominal (mean area x=3-6 in)", mach_nom)]
    def add(label, **kw):
        m, p, t = mach_inputs(**kw)
        sens.append((label, solve_mach(area_mean * FT2_TO_M2, m, p, t, kw.get("gamma", GAMMA))))

    add("mass flow +0.5 %", m_dot_lbm_s=M_DOT_LBM_S * 1.005)
    add("mass flow -0.5 %", m_dot_lbm_s=M_DOT_LBM_S * 0.995)
    add("pressure ratio +0.001", pr=TOTAL_PRESSURE_RATIO + 0.001)
    add("pressure ratio -0.001", pr=TOTAL_PRESSURE_RATIO - 0.001)
    add("temperature ratio +0.001", tr=TOTAL_TEMPERATURE_RATIO + 0.001)
    add("temperature ratio -0.001", tr=TOTAL_TEMPERATURE_RATIO - 0.001)
    add("inlet total pressure +1 %", p_t1=P_T1_PA * 1.01)
    add("inlet total pressure -1 %", p_t1=P_T1_PA * 0.99)
    add("inlet total temperature +2 K", t_t1=T_T1_K + 2.0)
    add("inlet total temperature -2 K", t_t1=T_T1_K - 2.0)
    add("gamma 1.39", gamma=1.39)
    add("gamma 1.41", gamma=1.41)
    sens.append(("area at x=6 in (minimum)", mach_max_area))
    sens.append(("area at x=3 in (maximum)", mach_min_area))

    sens_rows = []
    for label, mach in sens:
        area_ft2 = area_mean if not label.startswith("area at") else (
            area_min if "minimum" in label else area_max)
        area_m2 = area_ft2 * FT2_TO_M2
        m, p, t = mach_inputs()
        v = mach * math.sqrt(GAMMA * R_AIR * t_t2 / (1 + 0.5 * (GAMMA - 1) * mach ** 2))
        sens_rows.append({
            "case": label,
            "area_ft2": f"{area_ft2:.4f}",
            "mean_axial_mach": f"{mach:.5f}",
            "delta_mach_vs_nominal": f"{mach - mach_nom:+.5f}",
            "mean_axial_velocity_m_s": f"{v:.3f}",
        })
    write_csv(DER / "mach_sensitivity.csv", sens_rows, list(sens_rows[0].keys()))

    # root-sum-square uncertainty from the independent perturbations
    def spread(prefix: str) -> float:
        vals = [abs(m - mach_nom) for label, m in sens if label.startswith(prefix)]
        return max(vals) if vals else 0.0
    contributions = {
        "mass_flow_0.5pct": spread("mass flow"),
        "pressure_ratio_0.001": spread("pressure ratio"),
        "temperature_ratio_0.001": spread("temperature ratio"),
        "inlet_pressure_1pct": spread("inlet total pressure"),
        "inlet_temperature_2K": spread("inlet total temperature"),
        "gamma_0.01": spread("gamma"),
        "area_station_range": abs(mach_max_area - mach_nom),
    }
    mach_sigma = math.sqrt(sum(v ** 2 for v in contributions.values()))
    v_nom = mach_nom * math.sqrt(GAMMA * R_AIR * t_t2 / (1 + 0.5 * (GAMMA - 1) * mach_nom ** 2))

    # total-density cross-check (deliberately ignores static density rise)
    rho_t = p_t2 / (R_AIR * t_t2)
    a_t = math.sqrt(GAMMA * R_AIR * t_t2)
    mach_total_density = m_dot / (rho_t * area_mean * FT2_TO_M2 * a_t)

    fan_tip_r = FAN_TIP_DIAMETER_IN / 2.0
    fan_hub_r = fan_tip_r * FAN_RADIUS_RATIO
    fan_area_ft2 = math.pi * (fan_tip_r ** 2 - fan_hub_r ** 2) / 144.0

    rows = []
    for i, (x, r_hub, r_tip, area_ft2) in enumerate(areas):
        mach = solve_mach(area_ft2 * FT2_TO_M2, m_dot, p_t2, t_t2)
        v = mach * math.sqrt(GAMMA * R_AIR * t_t2 / (1 + 0.5 * (GAMMA - 1) * mach ** 2))
        rows.append({
            "station": f"x={x:.1f} in",
            "x_in": f"{x:.1f}",
            "hub_radius_in": f"{r_hub:.4f}",
            "tip_radius_in": f"{r_tip:.4f}",
            "annulus_area_ft2": f"{area_ft2:.4f}",
            "annulus_area_m2": f"{area_ft2 * FT2_TO_M2:.6f}",
            "mass_flow_lbm_s": f"{M_DOT_LBM_S:.2f}",
            "mass_flow_kg_s": f"{m_dot:.4f}",
            "mean_axial_mach": f"{mach:.4f}",
            "mean_axial_velocity_m_s": f"{v:.3f}",
            "mach_sigma": f"{mach_sigma:.4f}",
            "note": "annulus at axial station",
        })
    rows.append({
        "station": "mean x=3-6 in",
        "x_in": "3-6",
        "hub_radius_in": f"{np.mean([a[1] for a in areas]):.4f}",
        "tip_radius_in": f"{np.mean([a[2] for a in areas]):.4f}",
        "annulus_area_ft2": f"{area_mean:.4f}",
        "annulus_area_m2": f"{area_mean * FT2_TO_M2:.6f}",
        "mass_flow_lbm_s": f"{M_DOT_LBM_S:.2f}",
        "mass_flow_kg_s": f"{m_dot:.4f}",
        "mean_axial_mach": f"{mach_nom:.4f}",
        "mean_axial_velocity_m_s": f"{v_nom:.3f}",
        "mach_sigma": f"{mach_sigma:.4f}",
        "note": "nominal reported value; area range 2.094-2.119 ft2",
    })
    rows.append({
        "station": "fan face (Table 1)",
        "x_in": "",
        "hub_radius_in": f"{fan_hub_r:.3f}",
        "tip_radius_in": f"{fan_tip_r:.3f}",
        "annulus_area_ft2": f"{fan_area_ft2:.4f}",
        "annulus_area_m2": f"{fan_area_ft2 * FT2_TO_M2:.6f}",
        "mass_flow_lbm_s": "",
        "mass_flow_kg_s": "",
        "mean_axial_mach": "",
        "mean_axial_velocity_m_s": "",
        "mach_sigma": "",
        "note": "reference area only; Table 1 tip diameter 22 in and radius ratio 0.30",
    })
    write_csv(DER / "annulus_area_mach.csv", rows, list(rows[0].keys()))

    # efficiency consistency check (Table 3 pi, tau vs eta)
    eta_from_ratios = (TOTAL_PRESSURE_RATIO ** ((GAMMA - 1) / GAMMA) - 1) / (TOTAL_TEMPERATURE_RATIO - 1)

    checks = [
        {"check": "Figure 15 axis fits", "detail": f"x residual {AXES['x_fit']['max_abs_residual_in']:.4f} in; "
                                                     f"y residual {AXES['y_fit']['max_abs_residual_in']:.4f} in",
         "pass": AXES["x_fit"]["max_abs_residual_in"] < 0.02 and AXES["y_fit"]["max_abs_residual_in"] < 0.02},
        {"check": "flowpath contour split", "detail": f"{cinfo['n_inner']} inner + {cinfo['n_outer']} outer vertices; "
                                                       f"split at x reversal index {cinfo['split_index']}",
         "pass": cinfo["n_inner"] > 50 and cinfo["n_outer"] > 50},
        {"check": "annulus area plausible", "detail": f"{area_min:.3f}-{area_max:.3f} ft2 over x=3-6 in",
         "pass": 1.8 < area_min and area_max < 2.3},
        {"check": "mean axial Mach plausible", "detail": f"M = {mach_nom:.3f} +/- {mach_sigma:.3f} (k=1)",
         "pass": 0.20 < mach_nom < 0.45},
        {"check": "static-vs-total density audit", "detail": f"static solve M={mach_nom:.4f}; total-density shortcut "
                                                             f"M={mach_total_density:.4f}; difference {mach_nom - mach_total_density:+.4f}",
         "pass": mach_nom > mach_total_density},
        {"check": "eta consistency (pi, tau vs tabulated eta)",
         "detail": f"pi-tau implies eta={eta_from_ratios:.4f}; Table 3 reports {ADIABATIC_EFFICIENCY:.4f}",
         "pass": abs(eta_from_ratios - ADIABATIC_EFFICIENCY) < 0.015},
    ]
    write_csv(QA / "annulus_checks.csv", checks, list(checks[0].keys()))

    meta = {
        "pdf": PDF.name,
        "pdf_sha256": sha256(PDF),
        "pdf_page": PDF_PAGE,
        "report_page": REPORT_PAGE,
        "figure": FIGURE,
        "axis_calibration": AXES,
        "contour_extraction": cinfo,
        "inputs": {
            "mass_flow_lbm_s": M_DOT_LBM_S,
            "total_pressure_ratio": TOTAL_PRESSURE_RATIO,
            "total_temperature_ratio": TOTAL_TEMPERATURE_RATIO,
            "adiabatic_efficiency_reported": ADIABATIC_EFFICIENCY,
            "inlet_total_pressure_pa": P_T1_PA,
            "inlet_total_temperature_k": T_T1_K,
            "gamma": GAMMA,
            "gas_constant_j_kg_k": R_AIR,
            "source": "NASA/TM-2005-211681 Tables 1 and 3, approach (61.7 %) Fan/OGV",
        },
        "results": {
            "area_ft2_mean_x3_6": area_mean,
            "area_ft2_min_x3_6": area_min,
            "area_ft2_max_x3_6": area_max,
            "fan_face_area_ft2": fan_area_ft2,
            "mean_axial_mach": mach_nom,
            "mean_axial_mach_sigma_k1": mach_sigma,
            "mean_axial_velocity_m_s": v_nom,
            "mach_total_density_shortcut": mach_total_density,
            "mach_sensitivity_min": min(m for _, m in sens),
            "mach_sensitivity_max": max(m for _, m in sens),
            "uncertainty_contributions": contributions,
        },
        "station_choice": (
            "The report does not quote a single OGV leading-edge station.  Figure 15 "
            "is evaluated at x = 3-6 in, which brackets the OGV axial extent "
            "(pitchline chord 1.57 in baseline, 3.26 in low-count/low-noise).  "
            "The area varies by only 1.2 % over that range."
        ),
        "inlet_assumption": (
            "Corrected flow 58.30 lbm/s is referenced to standard sea-level total "
            "conditions.  The test-section total pressure/temperature are not quoted "
            "at this operating point, so p_t1 = 101325 Pa and T_t1 = 288.15 K are used; "
            "sensitivity to +/-1 % p_t1 and +/-2 K T_t1 is tabulated."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "annulus_mach.json", meta)

    # QA overlay: crop Figure 15, draw the digitised contours and the x=3-6 band
    gray = render_page(PDF, PDF_PAGE, dpi=400)
    make_overlay(gray, xi, ri, xo, ro, OVER / "fig15_flowpath_overlay.png")

    print(f"Figure 15: {cinfo['n_inner']} inner + {cinfo['n_outer']} outer vertices")
    print(f"area x=3-6 in: {area_min:.4f}-{area_max:.4f} ft2; mean {area_mean:.4f} ft2")
    print(f"fan face area: {fan_area_ft2:.4f} ft2")
    print(f"mean axial Mach: {mach_nom:.4f} +/- {mach_sigma:.4f} (k=1); V = {v_nom:.2f} m/s")
    print(f"total-density shortcut: {mach_total_density:.4f} (under-predicts, as expected)")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return meta


def make_overlay(gray: np.ndarray, xi, ri, xo, ro, outpath) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = 400.0 / 72.0
    L, T, R, B = [int(round(v * s)) for v in FRAME]
    m = 30
    crop = gray[max(0, T - m):B + m, max(0, L - m):R + m]
    xf, yf = AXES["x_fit"], AXES["y_fit"]
    def to_px(x, y):
        xpt = (x - xf["intercept_in"]) / xf["slope_in_per_pt"]
        ypt = (y - yf["intercept_in"]) / yf["slope_in_per_pt"]
        return xpt * s - (L - m), ypt * s - (T - m)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.plot([to_px(x, y)[0] for x, y in zip(xi, ri)],
            [to_px(x, y)[1] for x, y in zip(xi, ri)], "-", color="red", lw=1.2,
            label="inner contour (digitised)")
    ax.plot([to_px(x, y)[0] for x, y in zip(xo, ro)],
            [to_px(x, y)[1] for x, y in zip(xo, ro)], "-", color="blue", lw=1.2,
            label="outer contour (digitised)")
    for x in (3.0, 6.0):
        for y in (0.0, 18.0):
            px, py = to_px(x, y)
            ax.axvline(px, color="lime", lw=0.8, alpha=0.7)
    ax.set_xlim(0, crop.shape[1]); ax.set_ylim(crop.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TM-2005-211681 Fig. 15 - digitised flowpath; green lines x=3 and 6 in")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
