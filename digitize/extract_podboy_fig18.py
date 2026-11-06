"""Digitise Podboy Fig. 18 (B3): maximum tangential turbulent velocity.

Native image: 327 x 319 px, indexed image with an exact black / red / white
palette.  The legend inside the plot reads "Axial station LDV1" (black line)
and "Axial station LDV2" (red line), so the two curves are separated by
palette, not by guesswork.

Axes (read from the rendered figure; the tick labels are raster, they are not
in the PDF text layer):

    x : rotor speed, % speed, 50 at the left frame to 100 at the right frame,
        major ticks at 60/70/80/90 %
    y : maximum tangential turbulent velocity, ft/s, 80 at the top frame to
        30 at the bottom frame, major ticks every 5 ft/s

Each curve carries four asterisk markers (50, 61.7, 87.5 and 100 % speed).
The markers at 50 % and 100 % are centred on the plot frame and are therefore
clipped; the 61.7 % and 87.5 % markers are fully visible.  The figure list
asks for the 61.7 % pair (LDV1 + LDV2), which is published as the primary
result; the other speeds are published as flagged, lower-confidence context.

No vision model is used: colours are separated by exact palette masks, the
frame and ticks are found by run-length analysis, the legend box is found as a
hollow rectangle, the asterisk markers are found from their vertical run
length, and the curves are traced column by column.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

from common import DIG, OVER, PROV, QA, ROOT, write_csv, write_json
from raster_common import (colour_masks, detect_panels, detect_ticks,
                           fit_axis, load_embedded, remove_ticks)

PDF = ROOT / "20030065891.pdf"
PAGE = 28
REPORT_PAGE = 24
IMAGE_INDEX = 0
FIGURE = "Figure 18"

# major tick positions in the native raster and their axis values
X_TICKS_PX = [84.0, 140.0, 195.0, 251.0]
X_TICKS_VAL = [60.0, 70.0, 80.0, 90.0]
Y_TICKS_PX = [42.0, 70.0, 98.0, 126.0, 153.0, 181.0, 209.0, 237.0, 265.0]
Y_TICKS_VAL = [75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0]

STATIONS = {"black": "LDV1", "red": "LDV2"}
PRIMARY_SPEED_PCT = 61.7
NOMINAL_SPEEDS = [50.0, 61.7, 87.5, 100.0]

MARKER_RUN_PX = 5        # a full asterisk has a vertical run of >= 5 px
EDGE_STRIP_PX = 3        # columns searched for frame-clipped markers
MIN_EDGE_PX_PER_COL = 3  # pixels per column that indicate a clipped asterisk
TICK_MATCH_TOL_PX = 2.0


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def column_runs(mask: np.ndarray, x0: int, x1: int) -> dict[int, list[tuple[int, int]]]:
    runs: dict[int, list[tuple[int, int]]] = {}
    for x in range(x0, x1 + 1):
        ys = np.where(mask[:, x])[0]
        rr: list[tuple[int, int]] = []
        if len(ys):
            start = prev = int(ys[0])
            for y in ys[1:]:
                y = int(y)
                if y - prev > 1:
                    rr.append((start, prev))
                    start = y
                prev = y
            rr.append((start, prev))
        runs[x] = rr
    return runs


def legend_box(mask: np.ndarray, panel: dict) -> dict | None:
    """The legend rectangle: a hollow box drawn inside the plot frame.

    The search window is the plot interior, so the frame itself (which is also
    a hollow rectangle) can never be mistaken for the legend.
    """
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    x0, x1 = L + 2, R - 2
    y0, y1 = T + 2, B - 2
    sub = mask[y0:y1 + 1, x0:x1 + 1]
    lab, _ = ndimage.label(sub, np.ones((3, 3)))
    best = None
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
        box = {"x0": int(sl[1].start) + x0, "x1": int(sl[1].stop - 1) + x0,
               "y0": int(sl[0].start) + y0, "y1": int(sl[0].stop - 1) + y0,
               "area": area, "hole_fraction": round(hole, 3)}
        if best is None or area > best["area"]:
            best = box
    return best


def erase_ticks_at(mask: np.ndarray, panel: dict,
                   x_ticks: list[float], y_ticks: list[float],
                   depth: int = 8) -> np.ndarray:
    """Erase the inward tick stubs at verified tick positions.

    ``raster_common.remove_ticks`` rediscovers the ticks from the image and
    therefore also erases the two frame-clipped asterisks, whose half-visible
    strokes look like tick stubs.  Here the tick positions are the ones that
    were matched against the calibration, so only real ticks are removed.
    """
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    out = mask.copy()
    for x in x_ticks:
        xi = int(round(x))
        out[max(T, B - depth - 1):B + 1, max(0, xi - 1):xi + 2] = False
        out[T:min(B, T + depth + 1), max(0, xi - 1):xi + 2] = False
    for y in y_ticks:
        yi = int(round(y))
        out[max(0, yi - 1):yi + 2, L:min(R, L + depth + 1)] = False
        out[max(0, yi - 1):yi + 2, max(L, R - depth - 1):R + 1] = False
    return out


def clean_mask(mask: np.ndarray, panel: dict, box: dict | None,
               x_ticks: list[float], y_ticks: list[float]) -> np.ndarray:
    """Plot interior without frame, ticks or legend."""
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    out = mask.copy()
    out[:T + 1, :] = False
    out[B:, :] = False
    out[:, :L + 1] = False
    out[:, R:] = False
    if box:
        out[box["y0"]:box["y1"] + 1, box["x0"]:box["x1"] + 1] = False
    return erase_ticks_at(out, panel, x_ticks, y_ticks, depth=8)


# --------------------------------------------------------------------------- #
# marker detection
# --------------------------------------------------------------------------- #
def find_markers(mask: np.ndarray, panel: dict) -> list[dict]:
    """Asterisk markers: full ones by vertical run length, clipped by the frame.

    A full asterisk is 7 x 9 px, so its vertical run is >= 5 px while the
    1--2 px curve never produces a longer contiguous run.  The markers at the
    two frame edges are cut in half, so they are found instead by looking for
    columns next to the frame that carry several separated pixels.
    """
    L, R = panel["L"], panel["R"]
    runs = column_runs(mask, L + 1, R - 1)
    thick: dict[int, list[tuple[int, int]]] = {}
    for x, rr in runs.items():
        heavy = [r for r in rr if (r[1] - r[0] + 1) >= MARKER_RUN_PX]
        if heavy:
            thick[x] = heavy

    markers: list[dict] = []
    if thick:
        xs = sorted(thick)
        groups: list[list[int]] = [[xs[0]]]
        for x in xs[1:]:
            if x - groups[-1][-1] <= 2:
                groups[-1].append(x)
            else:
                groups.append([x])
        for grp in groups:
            mids = []
            for x in grp:
                a, b = max(thick[x], key=lambda r: r[1] - r[0])
                mids.append((a + b) / 2.0)
            cx = float(np.median(grp))
            cy = float(np.median(mids))
            spread = float(np.max(mids) - np.min(mids)) / 2.0 if len(mids) > 1 else 0.5
            markers.append({
                "kind": "full",
                "x_px": cx,
                "y_px": cy,
                "columns": [int(grp[0]), int(grp[-1])],
                "midpoint_spread_px": spread,
                "n_pixels": int(sum(mask[:, x].sum() for x in grp)),
            })

    # frame-clipped markers
    for side, (xa, xb) in (("left", (L + 1, L + EDGE_STRIP_PX)),
                           ("right", (R - EDGE_STRIP_PX, R - 1))):
        if xb < xa:
            continue
        strip = mask[:, xa:xb + 1]
        per_col = strip.sum(axis=0)
        if per_col.max() < MIN_EDGE_PX_PER_COL:
            continue
        ys = np.where(strip)[0]
        if len(ys) == 0:
            continue
        markers.append({
            "kind": "clipped",
            "side": side,
            "x_px": float(L if side == "left" else R),
            "y_px": float(np.median(ys)),
            "columns": [int(xa), int(xb)],
            "midpoint_spread_px": float((ys.max() - ys.min()) / 2.0),
            "n_pixels": int(len(ys)),
        })
    markers.sort(key=lambda m: m["x_px"])
    return markers


def trace_curve(mask: np.ndarray, panel: dict,
                markers: list[dict]) -> list[dict]:
    """Column-by-column curve trace with the marker blobs removed.

    The asterisk pixels bias the column median, so the marker columns are
    erased first; the gap they leave is closed by straight interpolation and
    the marker centre itself is imposed as the data point.
    """
    L, R = panel["L"], panel["R"]
    erased: set[int] = set()
    for m in markers:
        if m["kind"] == "full":
            lo, hi = m["columns"]
            erased.update(range(lo - 3, hi + 4))
        else:
            lo, hi = m["columns"]
            erased.update(range(lo, hi + 1))
    runs = column_runs(mask, L + 1, R - 1)
    xs: list[int] = []
    ys: list[float] = []
    for x in range(L + 1, R):
        if x in erased:
            continue
        rr = runs.get(x, [])
        if not rr:
            continue
        mids = [0.5 * (a + b) for a, b in rr]
        xs.append(x)
        ys.append(float(np.median(mids)))
    if not xs:
        return []
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    grid = np.arange(L + 1, R, dtype=float)
    values = np.interp(grid, xa, ya)
    # straight extrapolation beyond the first/last traced column, using a
    # least-squares fit to the nearest clean columns
    n_fit = min(6, len(xa))
    if grid.min() < xa.min():
        coef = np.polyfit(xa[:n_fit], ya[:n_fit], 1)
        left = grid < xa.min()
        values[left] = np.polyval(coef, grid[left])
    if grid.max() > xa.max():
        coef = np.polyfit(xa[-n_fit:], ya[-n_fit:], 1)
        right = grid > xa.max()
        values[right] = np.polyval(coef, grid[right])
    for m in markers:
        if m["kind"] == "full":
            cx = int(round(m["x_px"]))
            if L + 1 <= cx <= R - 1:
                values[cx - (L + 1)] = m["y_px"]
    return [{"x": int(x), "y": float(y)}
            for x, y in zip(grid, values)]


# --------------------------------------------------------------------------- #
# validation against the report text
# --------------------------------------------------------------------------- #
def cross_check(points: list[dict]) -> dict:
    """Check the digitised points against the statements in the report text.

    Report p. 10 says (Fig. 18 discussion):

        "The values ... show the same trend - the max measured tangential
         turbulent velocity increases almost linearly with rotor speed.  The
         main deviation from the linear trend occurs at 61.7 % speed, with
         these values slightly higher than the almost exact linear
         relationship suggested by the data of the other 3 speeds.  The
         dissipation of the max measured value with downstream distance is
         rather consistent across the 4 speeds, with the downstream values
         ranging between 24 to 31 % lower than the upstream values."

    LDV1 is the upstream station (3.12 in) and LDV2 the downstream one
    (6.49 in), so the expected dissipation is (LDV1 - LDV2) / LDV1.
    """
    by_station: dict[str, dict[float, dict]] = {}
    for p in points:
        by_station.setdefault(p["station"], {})[float(p["nominal_speed_percent"])] = p

    out: dict = {"statement_source": "20030065891.pdf, report p. 10 (Fig. 18 discussion)",
                 "stations": {}}
    for station, series in by_station.items():
        others = [s for s in (50.0, 87.5, 100.0) if s in series]
        if len(others) >= 2:
            sx = np.array(others, dtype=float)
            sy = np.array([float(series[s]["max_tangential_velocity_ft_s"]) for s in others])
            coef = np.polyfit(sx, sy, 1)
            pred = {float(s): float(np.polyval(coef, s)) for s in others}
            resid = {float(s): float(float(series[s]["max_tangential_velocity_ft_s"]) - pred[s])
                     for s in others}
            primary = PRIMARY_SPEED_PCT
            entry = {
                "linear_fit_other_speeds": {
                    "speeds_percent": others,
                    "slope_ft_s_per_percent": float(coef[0]),
                    "intercept_ft_s": float(coef[1]),
                    "residuals_ft_s": {f"{s:g}": round(v, 3) for s, v in resid.items()},
                    "max_abs_residual_ft_s": float(np.max(np.abs(list(resid.values())))),
                },
            }
            if primary in series:
                p = float(series[primary]["max_tangential_velocity_ft_s"])
                lin = float(np.polyval(coef, primary))
                entry["primary_61_7_percent"] = {
                    "digitised_ft_s": round(p, 3),
                    "linear_prediction_ft_s": round(lin, 3),
                    "excess_ft_s": round(p - lin, 3),
                    "excess_percent": round(100.0 * (p - lin) / lin, 2),
                    "report_says": "higher than the linear trend",
                    "consistent": bool(p - lin > 0.0),
                }
            out["stations"][station] = entry

    # downstream dissipation, LDV1 upstream and LDV2 downstream
    if "LDV1" in by_station and "LDV2" in by_station:
        diss = {}
        for speed in sorted(set(by_station["LDV1"]) & set(by_station["LDV2"])):
            up = float(by_station["LDV1"][speed]["max_tangential_velocity_ft_s"])
            down = float(by_station["LDV2"][speed]["max_tangential_velocity_ft_s"])
            pct = 100.0 * (up - down) / up
            diss[f"{speed:g}"] = {
                "upstream_LDV1_ft_s": round(up, 3),
                "downstream_LDV2_ft_s": round(down, 3),
                "dissipation_percent": round(pct, 2),
                "within_report_band_24_to_31_percent": bool(23.0 <= pct <= 32.0),
            }
        out["downstream_dissipation"] = diss
        out["downstream_dissipation"]["report_band_percent"] = [24.0, 31.0]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    rgb, image_path, info, rect = load_embedded(PDF, PAGE, IMAGE_INDEX)
    masks = colour_masks(rgb)
    panels = detect_panels(masks["black"])
    if len(panels) != 1:
        raise RuntimeError(f"expected one panel, found {len(panels)}")
    panel = panels[0]

    # ------------------------------------------------------------------ axes
    xfit = fit_axis(X_TICKS_PX, X_TICKS_VAL)
    yfit = fit_axis(Y_TICKS_PX, Y_TICKS_VAL)
    det_x = detect_ticks(masks["black"], panel, "bottom", depth=8)
    det_y = detect_ticks(masks["black"], panel, "left", depth=8)
    det_y_right = detect_ticks(masks["black"], panel, "right", depth=8)

    def match_ticks(detected: list[float], expected_px: list[float],
                    values: list[float]) -> list[dict]:
        checks = []
        for px, val in zip(expected_px, values):
            near = [d for d in detected if abs(d - px) <= TICK_MATCH_TOL_PX]
            checks.append({"tick_px": px, "value": val,
                           "detected_px": [round(d, 2) for d in near],
                           "matched": bool(near)})
        return checks

    x_checks = match_ticks(det_x, X_TICKS_PX, X_TICKS_VAL)
    y_checks = match_ticks(det_y, Y_TICKS_PX, Y_TICKS_VAL)
    y_checks_right = match_ticks(det_y_right, Y_TICKS_PX, Y_TICKS_VAL)
    if not all(c["matched"] for c in x_checks + y_checks + y_checks_right):
        raise RuntimeError("tick detection does not match the calibrated ticks")
    # detections that do not sit on a calibrated tick are rejected: they are
    # the half-visible asterisks on the frame, not ticks
    x_rejected = [round(d, 2) for d in det_x
                  if min(abs(d - t) for t in X_TICKS_PX) > TICK_MATCH_TOL_PX]
    y_rejected = [round(d, 2) for d in det_y
                  if min(abs(d - t) for t in Y_TICKS_PX) > TICK_MATCH_TOL_PX]

    def to_speed(x: float) -> float:
        return float(np.polyval([xfit["slope"], xfit["intercept"]], x))

    def to_velocity(y: float) -> float:
        return float(np.polyval([yfit["slope"], yfit["intercept"]], y))

    frame_speed = [to_speed(panel["L"]), to_speed(panel["R"])]
    frame_velocity = [to_velocity(panel["B"]), to_velocity(panel["T"])]

    # --------------------------------------------------------------- legend
    box = legend_box(masks["black"], panel)
    if box is None:
        raise RuntimeError("legend box not found")
    legend = {k: box[k] for k in ("x0", "x1", "y0", "y1", "area", "hole_fraction")}

    # -------------------------------------------------------------- extract
    point_rows: list[dict] = []
    curve_rows: list[dict] = []
    summary: dict = {"stations": {}}

    u_pixel_v = 0.5 * abs(yfit["slope"])                  # half a native pixel
    u_pixel_s = 0.5 * abs(xfit["slope"])
    u_cal_v = float(yfit["max_abs_residual"])
    u_cal_s = float(xfit["max_abs_residual"])

    for colour, station in STATIONS.items():
        mask = clean_mask(masks[colour], panel, box, X_TICKS_PX, Y_TICKS_PX)
        markers = find_markers(mask, panel)
        curve = trace_curve(mask, panel, markers)
        summary["stations"][station] = {
            "colour": colour,
            "n_markers": len(markers),
            "n_curve_columns": len(curve),
        }
        for r in curve:
            curve_rows.append({
                "source_pdf": PDF.name,
                "source_page_pdf": PAGE,
                "source_page_report": REPORT_PAGE,
                "figure": FIGURE,
                "station": station,
                "colour": colour,
                "x_px": r["x"],
                "y_px": f"{r['y']:.2f}",
                "speed_percent": f"{to_speed(r['x']):.3f}",
                "max_tangential_velocity_ft_s": f"{to_velocity(r['y']):.3f}",
                "method": "column median of the exact-palette curve pixels; "
                          "marker columns closed by interpolation and the "
                          "frame-clipped ends by straight extrapolation",
                "note": "",
            })
        for m in markers:
            speed = to_speed(m["x_px"])
            speed_nominal = min(NOMINAL_SPEEDS, key=lambda s: abs(s - speed))
            primary = abs(speed_nominal - PRIMARY_SPEED_PCT) < 1e-6
            if m["kind"] == "full":
                quality = "measured"
                u_marker = max(0.5, m["midpoint_spread_px"])
            else:
                quality = "clipped_by_frame"
                u_marker = max(0.8, m["midpoint_spread_px"])
            u_v = float(np.sqrt(u_cal_v ** 2 + u_pixel_v ** 2 + u_marker ** 2))
            u_s = float(np.sqrt(u_cal_s ** 2 + u_pixel_s ** 2))
            point_rows.append({
                "source_pdf": PDF.name,
                "source_page_pdf": PAGE,
                "source_page_report": REPORT_PAGE,
                "figure": FIGURE,
                "station": station,
                "colour": colour,
                "nominal_speed_percent": f"{speed_nominal:.1f}",
                "speed_percent_calibrated": f"{speed:.3f}",
                "speed_sigma_percent": f"{u_s:.3f}",
                "x_px": f"{m['x_px']:.2f}",
                "y_px": f"{m['y_px']:.2f}",
                "max_tangential_velocity_ft_s": f"{to_velocity(m['y_px']):.3f}",
                "velocity_sigma_ft_s": f"{u_v:.3f}",
                "u_calibration_ft_s": f"{u_cal_v:.3f}",
                "u_pixel_ft_s": f"{u_pixel_v:.3f}",
                "u_marker_ft_s": f"{u_marker:.3f}",
                "primary_b3_point": int(primary),
                "quality": quality,
                "method": ("asterisk marker: vertical run >= 5 px"
                           if m["kind"] == "full" else
                           "asterisk marker clipped by the plot frame; "
                           "centre taken as the median of the visible pixels"),
                "note": ("B3 deliverable: 61.7 % speed"
                         if primary else
                         "context only; marker clipped by the frame"
                         if m["kind"] == "clipped" else
                         "context only; fully visible marker"),
            })

    point_rows.sort(key=lambda r: (float(r["nominal_speed_percent"]),
                                    r["station"]))

    write_csv(DIG / "podboy_fig18_points.csv", point_rows,
              list(point_rows[0].keys()))
    write_csv(DIG / "podboy_fig18_curve.csv", curve_rows,
              list(curve_rows[0].keys()))

    # -------------------------------------------------------- report cross-check
    cross = cross_check(point_rows)

    # ----------------------------------------------------------------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    ax.imshow(rgb)
    colours = {"black": "k", "red": "crimson"}
    for colour, station in STATIONS.items():
        sel = [r for r in curve_rows if r["station"] == station]
        ax.plot([r["x_px"] for r in sel], [float(r["y_px"]) for r in sel],
                color=colours[colour], lw=1.2, label=f"{station} traced curve")
    for row in point_rows:
        primary = row["primary_b3_point"] == "1"
        ax.errorbar(float(row["x_px"]), float(row["y_px"]),
                    yerr=float(row["velocity_sigma_ft_s"]),
                    fmt="o" if primary else "s", ms=9 if primary else 6,
                    mfc="none" if primary else colours[row["colour"]],
                    mec=colours[row["colour"]], mew=1.6,
                    capsize=3, zorder=5,
                    label=(f"{row['station']} {row['nominal_speed_percent']} %"
                           + (" (B3 point)" if primary else " (context)")))
    # calibrated axes
    ax.set_xticks([panel["L"]] + X_TICKS_PX + [panel["R"]])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"])
    ax.set_yticks(Y_TICKS_PX)
    ax.set_yticklabels([f"{v:g}" for v in Y_TICKS_VAL])
    ax.set_xlim(18, 318)
    ax.set_ylim(302, 6)
    ax.set_xlabel("Rotor speed, %")
    ax.set_ylabel("Max tangential turbulent velocity, ft/s")
    ax.set_title("Podboy Fig. 18 — maximum tangential turbulent velocity\n"
                 "(open circles: B3 61.7 % points; filled squares: flagged context)")
    ax.legend(fontsize=7.0, loc="lower left", bbox_to_anchor=(0.0, -0.20),
              ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OVER / "podboy_fig18_digitised.png", dpi=180)
    plt.close(fig)

    meta = {
        "figure": FIGURE,
        "pdf": PDF.name,
        "pdf_page": PAGE,
        "source_page_report": REPORT_PAGE,
        "image": info,
        "image_file": str(image_path),
        "palette": "black/red/white",
        "station_mapping": "black = LDV1, red = LDV2 (legend inside the plot)",
        "axes": {
            "x": "rotor speed 50-100 %",
            "y": "maximum tangential turbulent velocity 30-80 ft/s",
            "tick_labels_source": "read from the rendered raster; the tick "
                                  "labels are not present in the PDF text layer",
        },
        "calibration": {
            "x": xfit, "y": yfit,
            "tick_checks_x": x_checks, "tick_checks_y": y_checks,
            "tick_checks_y_right": y_checks_right,
            "rejected_tick_detections_x": x_rejected,
            "rejected_tick_detections_y": y_rejected,
            "frame_speed_percent": [round(v, 3) for v in frame_speed],
            "frame_velocity_ft_s": [round(v, 3) for v in frame_velocity],
        },
        "legend_box_px": legend,
        "method": "exact palette masks; hollow-rectangle legend removal; "
                  "asterisk detection by vertical run length; column-median "
                  "curve trace",
        "b3_points": [r for r in point_rows if r["primary_b3_point"] == "1"],
        "report_cross_check": cross,
        "uncertainty": {
            "pixel_ft_s": u_pixel_v,
            "calibration_ft_s": u_cal_v,
            "marker_rule": "half the spread of the marker run midpoints "
                           "(full markers); half the visible vertical extent "
                           "(frame-clipped markers), floor 0.8 ft/s",
        },
        "summary": summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "podboy_fig18.json", meta)
    write_json(QA / "podboy_fig18_summary.json", {
        "n_points": len(point_rows),
        "n_curve_rows": len(curve_rows),
        "primary_points": [r for r in point_rows if r["primary_b3_point"] == "1"],
        "tick_checks_x": x_checks,
        "tick_checks_y": y_checks,
        "tick_checks_y_right": y_checks_right,
        "rejected_tick_detections_x": x_rejected,
        "rejected_tick_detections_y": y_rejected,
        "frame_speed_percent": [round(v, 3) for v in frame_speed],
        "frame_velocity_ft_s": [round(v, 3) for v in frame_velocity],
        "summary": summary,
        "report_cross_check": cross,
    })
    for r in point_rows:
        print(f"Fig18 {r['station']} {r['nominal_speed_percent']:>5}% "
              f"x={r['x_px']:>6} y={r['y_px']:>6} -> "
              f"{r['max_tangential_velocity_ft_s']:>7} ft/s "
              f"+- {r['velocity_sigma_ft_s']} ({r['quality']})"
              + ("  <-- B3" if r["primary_b3_point"] == "1" else ""))
    return meta


if __name__ == "__main__":
    main()
