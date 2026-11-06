"""Digitise Podboy Fig. 28 (B1): turbulence integral length scales.

Native image: 505 x 232 px, indexed CalRGB with exact black/red/blue/white
palette.  Report p. 29 gives the mapping:

    black = Streamwise, red = Upwash, blue = Spanwise
    line without symbols = HW1, line with symbols = HW2

Each colour therefore carries two curves: a thin HW1 line and a sequence of
HW2 asterisks.  The asterisks are found as pixels whose vertical and
horizontal dark runs are both at least five pixels; the HW1 line is traced
after erasing a 9 x 9 box around every asterisk.  No vision model is used.

Axes: radius 5--11 in, integral length scale 0--0.4 in.  The red and blue
curves are flagged where they lie within 0.006 in of one another near 10 in,
as noted on the figure list.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

from common import DIG, OVER, PROV, QA, ROOT, write_csv, write_json
from raster_common import (colour_masks, detect_panels, detect_ticks,
                           fit_axis, load_embedded, remove_ticks)

PDF = ROOT / "20030065891.pdf"
PAGE = 34
REPORT_PAGE = 30
FIGURE = "Figure 28"
IMAGE_INDEX = 0

COMPONENTS = {"black": "streamwise", "red": "upwash", "blue": "spanwise"}
START_Y = {"black": 158.0, "red": 193.0, "blue": 185.0}

X_TICKS_PX = [105.0, 183.0, 260.0, 337.0, 415.0]
X_TICKS_VAL = [6.0, 7.0, 8.0, 9.0, 10.0]
Y_TICKS_PX = [35.0, 60.0, 86.0, 112.0, 137.0, 163.0, 188.0]
Y_TICKS_VAL = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

X_SIGMA_IN = 0.013          # one pixel, 6 in over 464 px
Y_SIGMA_IN = 0.0020         # one pixel, 0.4 in over 205 px
MARKER_RAD_PX = 4           # erase radius around an asterisk
MAX_INTERP_GAP_PX = 12
MARKER_EXTRAP_PX = 2        # small extrapolation to reach a measured line endpoint
OVERLAP_PX = 3.0            # red/blue separation regarded as overlap
X_MIN, X_MAX = 44, 486


def run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Length of the dark run through every pixel, along rows and columns."""
    h, w = mask.shape
    vr = np.zeros_like(mask, dtype=int)
    hr = np.zeros_like(mask, dtype=int)
    for x in range(w):
        col = mask[:, x]
        i = 0
        while i < h:
            if col[i]:
                j = i
                while j < h and col[j]:
                    j += 1
                vr[i:j, x] = j - i
                i = j
            else:
                i += 1
    for y in range(h):
        row = mask[y, :]
        i = 0
        while i < w:
            if row[i]:
                j = i
                while j < w and row[j]:
                    j += 1
                hr[y, i:j] = j - i
                i = j
            else:
                i += 1
    return vr, hr


def interior(colour: str, masks: dict, panel: dict) -> np.ndarray:
    """Plot interior for one colour, without frame, ticks or legend."""
    mask = masks[colour].copy()
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    mask[:T + 1, :] = False
    mask[B:, :] = False
    mask[:, :L + 1] = False
    mask[:, R:] = False
    # legend box + text, top-left
    mask[T + 2:66, L + 14:152] = False
    mask = remove_ticks(mask, panel, depth=8)
    return mask


def find_markers(mask: np.ndarray) -> list[dict]:
    """Asterisks: both the horizontal and vertical run through the centre >=5."""
    vr, hr = run_lengths(mask)
    cand = (vr >= 5) & (hr >= 5)
    lab, _ = ndimage.label(cand, np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        yy, xx = np.where(lab[sl] == i)
        cx = float(xx.mean() + sl[1].start)
        cy = float(yy.mean() + sl[0].start)
        if not (X_MIN <= cx <= X_MAX):
            continue
        y0 = max(0, int(round(cy)) - 3)
        y1 = min(mask.shape[0], int(round(cy)) + 4)
        x0 = max(0, int(round(cx)) - 3)
        x1 = min(mask.shape[1], int(round(cx)) + 4)
        score = int(mask[y0:y1, x0:x1].sum())
        if score < 18:
            continue
        out.append({"x": cx, "y": cy, "score": score})
    return sorted(out, key=lambda p: (p["x"], p["y"]))


def robust_markers(points: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reject isolated components and resolve duplicate x positions."""
    if len(points) < 6:
        return points, []
    x = np.array([p["x"] for p in points])
    y = np.array([p["y"] for p in points])
    keep = np.ones(len(x), dtype=bool)
    residual = np.zeros(len(x))
    for _ in range(5):
        if keep.sum() < 6:
            break
        residual = np.zeros(len(x))
        for i in range(len(x)):
            sel = keep & (np.abs(x - x[i]) <= 30) & (np.abs(x - x[i]) > 0)
            if sel.sum() < 4:
                sel = keep & (np.abs(x - x[i]) <= 50) & (np.abs(x - x[i]) > 0)
            if sel.sum() >= 3:
                coef = np.polyfit(x[sel] - x[i], y[sel], 2)
                residual[i] = y[i] - np.polyval(coef, 0)
        scale = np.median(np.abs(residual[keep])) * 1.4826
        limit = max(2.0, 3.0 * scale)
        new = keep & (np.abs(residual) <= limit)
        if new.sum() == keep.sum():
            break
        keep = new

    kept = [p for p, k in zip(points, keep) if k]
    rejected = [p for p, k in zip(points, keep) if not k]
    kept.sort(key=lambda p: (p["x"], p["y"]))
    merged = []
    i = 0
    while i < len(kept):
        j = i + 1
        while j < len(kept) and kept[j]["x"] - kept[i]["x"] <= 2.0:
            j += 1
        if j - i == 1:
            merged.append(kept[i])
        else:
            kx = np.array([p["x"] for p in kept])
            ky = np.array([p["y"] for p in kept])
            coef = np.polyfit(kx, ky, 3) if len(kx) > 4 else None
            pred = np.polyval(coef, kept[i]["x"]) if coef is not None else np.median(ky[i:j])
            best = min(kept[i:j], key=lambda p: abs(p["y"] - pred))
            merged.append(best)
        i = j
    return merged, rejected


def strip_markers(mask: np.ndarray, points: list[dict], radius: int = MARKER_RAD_PX) -> np.ndarray:
    out = mask.copy()
    for p in points:
        xi = int(round(p["x"]))
        yi = int(round(p["y"]))
        out[max(0, yi - radius):yi + radius + 1,
            max(0, xi - radius):xi + radius + 1] = False
    return out


def column_runs(mask: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    runs = {}
    for x in range(X_MIN, X_MAX + 1):
        ys = np.where(mask[:, x])[0]
        rr = []
        if len(ys):
            start = prev = int(ys[0])
            for y in ys[1:]:
                y = int(y)
                if y - prev > 1:
                    rr.append((start, prev))
                    start = y
                prev = y
            rr.append((start, prev))
        runs[x] = [(a, b) for a, b in rr if 1 <= b - a + 1 <= 20]
    return runs


def track_line(runs: dict[int, list[tuple[int, int]]], start_y: float) -> dict[int, dict]:
    """Slope-prior tracking of the thin HW1 line through marker gaps."""
    y = start_y
    slope = 0.0
    gap = 0
    out = {}
    for x in range(X_MIN, X_MAX + 1):
        pred = y + slope
        rr = runs.get(x, [])
        if rr:
            centres = np.array([(a + b) / 2.0 for a, b in rr])
            d = np.abs(centres - pred)
            k = int(np.argmin(d))
            limit = 4.0 + min(gap, 6) * 1.2
            chosen = None
            if d[k] <= limit:
                chosen = float(centres[k])
            elif gap >= 3 and d[k] <= 15.0:
                # re-acquire after the line has been hidden under an asterisk
                chosen = float(centres[k])
            if chosen is not None:
                slope = float(np.clip(0.6 * slope + 0.4 * (chosen - y), -4.0, 4.0))
                y = chosen
                gap = 0
                a, b = rr[k]
                out[x] = {"y": y, "y_min": float(a), "y_max": float(b),
                          "run_px": int(b - a + 1), "gap_before": 0}
                continue
        gap += 1
        y = pred
        slope *= 0.85
    # annotate the gap length before each retained column
    last = None
    for x in sorted(out):
        if last is not None and x - last > 1:
            out[x]["gap_before"] = x - last - 1
        last = x
    return out


def build_line_curve(trace: dict[int, dict],
                     markers: list[dict]) -> tuple[dict[int, float], dict[int, str]]:
    """Measured HW1 line, short-gap interpolation, long-gap HW2 proxy.

    Where the thin line is hidden under the HW2 asterisks for more than
    ``MAX_INTERP_GAP_PX`` columns, the HW2 asterisk curve is used as a proxy.
    The vertical offset between the visible HW1 line and the asterisk curve is
    interpolated between the two measured endpoints of the hidden span, so the
    proxy joins the measured line continuously at both ends.  No value is
    invented when neither source exists.
    """
    grid = np.arange(X_MIN, X_MAX + 1)
    values = np.full(len(grid), np.nan)
    source = np.full(len(grid), "", dtype=object)
    xs = np.array(sorted(trace), dtype=float)
    ys = np.array([trace[int(x)]["y"] for x in xs], dtype=float)
    for x in trace:
        values[x - X_MIN] = trace[x]["y"]
        source[x - X_MIN] = "measured"

    marker_at = np.full(len(grid), np.nan)
    if len(markers) >= 2:
        mx = np.array([p["x"] for p in markers], dtype=float)
        my = np.array([p["y"] for p in markers], dtype=float)
        order = np.argsort(mx)
        mx, my = mx[order], my[order]
        marker_at = np.interp(grid, mx, my)
        # np.interp holds the end values constant; replace that with a short
        # linear extrapolation so a hidden span can close on a measured point
        # lying one or two pixels beyond the last asterisk.
        left = grid < mx.min()
        right = grid > mx.max()
        marker_at[left] = np.nan
        marker_at[right] = np.nan
        near_left = left & (grid >= mx.min() - MARKER_EXTRAP_PX)
        near_right = right & (grid <= mx.max() + MARKER_EXTRAP_PX)
        if near_left.any():
            slope = (my[1] - my[0]) / (mx[1] - mx[0])
            marker_at[near_left] = my[0] + slope * (grid[near_left] - mx[0])
        if near_right.any():
            slope = (my[-1] - my[-2]) / (mx[-1] - mx[-2])
            marker_at[near_right] = my[-1] + slope * (grid[near_right] - mx[-1])

    residual_at_measured = {
        int(x): float(y - marker_at[int(x) - X_MIN])
        for x, y in zip(xs, ys)
        if np.isfinite(marker_at[int(x) - X_MIN])
    }

    for i in range(len(grid)):
        x = int(grid[i])
        if source[i]:
            continue
        j = np.searchsorted(xs, x)
        if j == 0 or j >= len(xs):
            continue
        x0, x1 = int(xs[j - 1]), int(xs[j])
        if x1 - x0 <= MAX_INTERP_GAP_PX + 1:
            values[i] = np.interp(x, xs, ys)
            source[i] = "interpolated"
        elif (x0 in residual_at_measured and x1 in residual_at_measured
              and np.isfinite(marker_at[i])):
            fraction = (x - x0) / float(x1 - x0)
            offset = ((1.0 - fraction) * residual_at_measured[x0]
                      + fraction * residual_at_measured[x1])
            values[i] = marker_at[i] + offset
            source[i] = "HW2_offset_proxy"
    return ({int(x): float(v) for x, v in zip(grid, values) if np.isfinite(v)},
            {int(x): str(s) for x, s in zip(grid, source) if s})


def contiguous_runs(xs: list[int]) -> list[tuple[int, int]]:
    if not xs:
        return []
    runs = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x - prev <= 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs


def overlap_interval(line_traces: dict[str, dict[int, float]]) -> dict:
    """Longest contiguous red/blue coincidence; the figure notes one near 10 in."""
    red, blue = line_traces["red"], line_traces["blue"]
    common = sorted(set(red) & set(blue))
    flagged = [x for x in common if abs(red[x] - blue[x]) <= OVERLAP_PX]
    runs = contiguous_runs(flagged)
    if not runs:
        return {"x_px_min": None, "x_px_max": None,
                "radius_in_min": None, "radius_in_max": None,
                "n_columns": 0, "all_runs": []}

    def radius(px: int) -> float:
        return float((px + 360.0) / 77.5)

    # choose the run nearest 10 in (x ~ 415), preferring longer runs
    def score(run: tuple[int, int]) -> tuple[float, int]:
        centre = (run[0] + run[1]) / 2.0
        return (abs(radius(centre) - 10.0), -(run[1] - run[0]))
    best = min(runs, key=score)
    return {"x_px_min": best[0], "x_px_max": best[1],
            "radius_in_min": radius(best[0]),
            "radius_in_max": radius(best[1]),
            "n_columns": best[1] - best[0] + 1,
            "all_runs": [{"x_px_min": a, "x_px_max": b,
                          "radius_in_min": radius(a),
                          "radius_in_max": radius(b)} for a, b in runs]}


def main() -> dict:
    rgb, image_path, info, rect = load_embedded(PDF, PAGE, IMAGE_INDEX)
    masks = colour_masks(rgb)
    panels = detect_panels(masks["black"])
    if len(panels) != 1:
        raise RuntimeError(f"expected one panel, found {len(panels)}")
    panel = panels[0]

    xfit = fit_axis(X_TICKS_PX, X_TICKS_VAL)
    yfit = fit_axis(Y_TICKS_PX, Y_TICKS_VAL)

    def to_radius(x: float) -> float:
        return float(np.polyval([xfit["slope"], xfit["intercept"]], x))

    def to_value(y: float) -> float:
        return float(np.polyval([yfit["slope"], yfit["intercept"]], y))

    # verify tick calibration against the frame
    x_at_frame = [to_radius(panel["L"]), to_radius(panel["R"])]
    y_at_frame = [to_value(panel["B"]), to_value(panel["T"])]

    rows = []
    summary = {"components": {}, "markers": {}, "line_columns": {},
               "rejected_markers": {}}
    line_traces = {}
    marker_rows = []
    for colour, component in COMPONENTS.items():
        mask = interior(colour, masks, panel)
        points = find_markers(mask)
        kept, rejected = robust_markers(points)
        summary["markers"][component] = len(kept)
        summary["rejected_markers"][component] = len(rejected)
        line_mask = strip_markers(mask, points)
        runs = column_runs(line_mask)
        trace = track_line(runs, START_Y[colour])
        line_values, line_source = build_line_curve(trace, kept)
        line_traces[colour] = line_values
        summary["line_columns"][component] = {
            "measured": sum(1 for s in line_source.values() if s == "measured"),
            "interpolated": sum(1 for s in line_source.values() if s == "interpolated"),
            "hw2_proxy": sum(1 for s in line_source.values() if s == "HW2_offset_proxy"),
            "total": len(line_values),
        }

        for x, y in sorted(line_values.items()):
            radius = to_radius(x)
            value = to_value(y)
            src = line_source[x]
            rows.append({
                "source_pdf": PDF.name,
                "source_page_pdf": PAGE,
                "source_page_report": REPORT_PAGE,
                "figure": FIGURE,
                "component": component,
                "station": "HW1",
                "series_type": "line",
                "radius_in": f"{radius:.4f}",
                "length_scale_in": f"{value:.4f}",
                "radius_sigma_in": f"{X_SIGMA_IN:.4f}",
                "length_scale_sigma_in": f"{Y_SIGMA_IN:.4f}",
                "x_px": x,
                "y_px": f"{y:.2f}",
                "method": ("thin-line trace after erasing HW2 asterisks"
                           if src == "measured" else
                           "linear interpolation across short gap"
                           if src == "interpolated" else
                           "HW2 asterisk curve plus smoothed HW1-HW2 offset"),
                "hw1_source": src,
                "interpolated": int(src != "measured"),
                "overlap_flag": "",
                "note": ("line obscured by asterisks; HW2 offset proxy"
                         if src == "HW2_offset_proxy" else ""),
            })

        for p in kept:
            radius = to_radius(p["x"])
            value = to_value(p["y"])
            row = {
                "source_pdf": PDF.name,
                "source_page_pdf": PAGE,
                "source_page_report": REPORT_PAGE,
                "figure": FIGURE,
                "component": component,
                "station": "HW2",
                "series_type": "asterisk",
                "radius_in": f"{radius:.4f}",
                "length_scale_in": f"{value:.4f}",
                "radius_sigma_in": f"{X_SIGMA_IN:.4f}",
                "length_scale_sigma_in": f"{Y_SIGMA_IN:.4f}",
                "x_px": f"{p['x']:.2f}",
                "y_px": f"{p['y']:.2f}",
                "method": "asterisk centre: horizontal and vertical runs >=5 px",
                "hw1_source": "",
                "interpolated": 0,
                "overlap_flag": "",
                "note": "",
            }
            rows.append(row)
            marker_rows.append(row)

    overlap = overlap_interval(line_traces)
    # mark rows inside the overlap interval (both stations of red and blue)
    for row in rows:
        if row["component"] not in ("upwash", "spanwise"):
            continue
        if overlap["x_px_min"] is None:
            continue
        x = float(row["x_px"])
        if overlap["x_px_min"] <= x <= overlap["x_px_max"]:
            row["overlap_flag"] = 1
            row["note"] = "red/blue curves within 0.006 in"

    outfile = DIG / "podboy_fig28_integral_length_scales.csv"
    write_csv(outfile, rows, list(rows[0].keys()))

    # QA overlay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(rgb)
    colours = {"black": "k", "red": "crimson", "blue": "royalblue"}
    for colour, component in COMPONENTS.items():
        tr = sorted(line_traces[colour].items())
        ax.plot([p[0] for p in tr], [p[1] for p in tr],
                color=colours[colour], lw=1.4,
                label=f"{component} HW1")
        pts = [r for r in marker_rows if r["component"] == component]
        ax.scatter([float(p["x_px"]) for p in pts],
                   [float(p["y_px"]) for p in pts],
                   s=50, facecolors="none", edgecolors=colours[colour],
                   linewidths=1.1, label=f"{component} HW2")
    ax.set_xlim(24, 498)
    ax.set_ylim(218, 4)
    ax.set_title("Podboy Fig. 28 — digitised HW1 lines and HW2 asterisks")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OVER / "podboy_fig28_digitised.png", dpi=180)
    plt.close(fig)

    meta = {
        "figure": FIGURE,
        "pdf": PDF.name,
        "pdf_page": PAGE,
        "report_page": REPORT_PAGE,
        "image": info,
        "image_file": info.get("file", str(image_path)),
        "palette": "black/red/blue/white",
        "component_mapping": COMPONENTS,
        "station_mapping": "line = HW1, asterisks = HW2 (report p. 29)",
        "axes": {"x": "radius 5-11 in", "y": "integral length scale 0-0.4 in"},
        "calibration": {
            "x": xfit, "y": yfit,
            "frame_radius_in": [round(v, 4) for v in x_at_frame],
            "frame_length_scale_in": [round(v, 4) for v in y_at_frame],
        },
        "method": "exact palette masks; asterisk run-length detection; "
                  "slope-prior thin-line trace after 9x9 marker erasure",
        "markers_kept": summary["markers"],
        "markers_rejected": summary["rejected_markers"],
        "line_columns": summary["line_columns"],
        "overlap_red_blue": overlap,
        "uncertainty": {"radius_in": X_SIGMA_IN, "length_scale_in": Y_SIGMA_IN,
                        "basis": "one native pixel"},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "podboy_fig28.json", meta)
    write_json(QA / "podboy_fig28_summary.json", {
        "n_rows": len(rows), "n_markers": len(marker_rows),
        "overlap_red_blue": overlap,
        "markers_kept": summary["markers"],
        "line_columns": summary["line_columns"],
    })
    print(f"Fig28: {len(rows)} rows; markers "
          f"{summary['markers']}; line columns {summary['line_columns']}")
    print(f"  overlap red/blue: {overlap['radius_in_min']:.2f}-"
          f"{overlap['radius_in_max']:.2f} in ({overlap['n_columns']} columns)")
    return meta


if __name__ == "__main__":
    main()
