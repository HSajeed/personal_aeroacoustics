"""Deterministic helpers for the embedded palette raster figures.

The Podboy figures are indexed CalRGB images with 3--4 exact palette entries
(black / red / blue / white).  The Nallasamy Figs. 8 and 17 are grayscale page
scans.  Nothing here uses a vision model: colours are separated by exact RGB
value, frames and ticks are found by run-length analysis, and curves are
extracted column by column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image
from scipy import ndimage

from common import RAW, write_json


# --------------------------------------------------------------------------- #
# embedded image access
# --------------------------------------------------------------------------- #
def load_embedded(pdf: Path, page: int, index: int = 0,
                  cache_dir: Path | None = None) -> tuple[np.ndarray, Path, dict, fitz.Rect]:
    """Extract an embedded image as RGB, cache the native PNG, return placement.

    Returns (rgb, path, info, placement_rect).  ``placement_rect`` is in PDF
    points and is used only to map text-layer tick labels onto image pixels.
    """
    doc = fitz.open(pdf)
    page_obj = doc[page - 1]
    images = page_obj.get_images(full=True)
    if index >= len(images):
        raise IndexError(f"page {page}: only {len(images)} images")
    xref = images[index][0]
    info = doc.extract_image(xref)
    cache_dir = cache_dir or (RAW / "figures")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"p{page}_img{index}_{xref}.{info['ext']}"
    if not path.exists():
        path.write_bytes(info["image"])
    rgb = np.array(Image.open(path).convert("RGB"))
    rects = page_obj.get_image_rects(xref)
    if not rects:
        raise RuntimeError(f"page {page} xref {xref}: no placement rect")
    meta = {
        "xref": xref,
        "width": int(info["width"]),
        "height": int(info["height"]),
        "bpc": int(info["bpc"]),
        "colorspace": int(info["colorspace"]),
        "ext": info["ext"],
        "bytes": len(info["image"]),
        "placement_pt": [round(rects[0].x0, 3), round(rects[0].y0, 3),
                         round(rects[0].x1, 3), round(rects[0].y1, 3)],
    }
    return rgb, path, meta, rects[0]


def page_to_image(x_pt: float, y_pt: float, rect: fitz.Rect,
                  width: int, height: int) -> tuple[float, float]:
    x = (x_pt - rect.x0) / rect.width * width
    y = (y_pt - rect.y0) / rect.height * height
    return float(x), float(y)


# --------------------------------------------------------------------------- #
# exact palette masks
# --------------------------------------------------------------------------- #
def colour_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Exact palette masks for the Podboy figures.

    The native images contain only (0,0,0), (255,0,0), (0,0,255) and white,
    so thresholds are wide safety margins rather than tunable parameters.
    """
    r = rgb[:, :, 0].astype(int)
    g = rgb[:, :, 1].astype(int)
    b = rgb[:, :, 2].astype(int)
    return {
        "black": (r < 40) & (g < 40) & (b < 40),
        "red": (r > 180) & (g < 80) & (b < 80),
        "blue": (b > 180) & (r < 80) & (g < 80),
        "grey": (r > 40) & (r < 160) & (np.abs(r - g) < 12) & (np.abs(r - b) < 12),
        "ink": (r < 200) | (g < 200) | (b < 200),
    }


def palette_census(rgb: np.ndarray) -> list[dict]:
    cols, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    rows = []
    for c, n in sorted(zip(cols, counts), key=lambda t: -t[1]):
        rows.append({"rgb": [int(v) for v in c], "count": int(n),
                     "fraction": round(float(n) / rgb.shape[0] / rgb.shape[1], 6)})
    return rows


# --------------------------------------------------------------------------- #
# frames and ticks
# --------------------------------------------------------------------------- #
def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    out = []
    start = None
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def detect_panels(binary: np.ndarray, row_frac: float = 0.45,
                  col_frac: float = 0.45, min_panel_width: int = 60) -> list[dict]:
    """Detect panel frames from long horizontal/vertical black runs.

    Returns one dict per panel with L,T,R,B pixel bounds (inclusive).
    """
    h, w = binary.shape
    rows = [y for y in range(h) if binary[y].sum() > row_frac * w]
    cols = [x for x in range(w) if binary[:, x].sum() > col_frac * h]
    if len(rows) < 2 or len(cols) < 2:
        raise RuntimeError(f"frame not found: rows={rows}, cols={cols}")
    top, bottom = min(rows), max(rows)
    # group columns that belong to the same vertical frame line
    groups: list[list[int]] = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    centres = [int(round(np.mean(g))) for g in groups]
    # consecutive pairs are the left/right edges of each panel
    panels = []
    for i in range(0, len(centres) - 1, 2):
        L, R = centres[i], centres[i + 1]
        if R - L >= min_panel_width:
            panels.append({"L": L, "T": top, "R": R, "B": bottom,
                           "width": R - L + 1, "height": bottom - top + 1})
    if not panels:
        panels = [{"L": centres[0], "T": top, "R": centres[-1], "B": bottom,
                   "width": centres[-1] - centres[0] + 1,
                   "height": bottom - top + 1}]
    return panels


def detect_ticks(binary: np.ndarray, panel: dict, side: str,
                 depth: int = 7, min_hits: int = 3) -> list[float]:
    """Detect major tick centres on one side of a panel frame.

    Ticks point inward.  A tick is accepted when at least ``min_hits`` dark
    pixels lie in the inward strip; contiguous accepted positions are grouped.
    """
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    if side == "bottom":
        span = range(L + 2, R - 1)
        hits = [x for x in span
                if binary[max(T, B - depth):B, x].sum() >= min_hits]
    elif side == "top":
        span = range(L + 2, R - 1)
        hits = [x for x in span
                if binary[T + 1:min(B, T + depth), x].sum() >= min_hits]
    elif side == "left":
        span = range(T + 2, B - 1)
        hits = [y for y in span
                if binary[y, L + 1:min(R, L + depth)].sum() >= min_hits]
    elif side == "right":
        span = range(T + 2, B - 1)
        hits = [y for y in span
                if binary[y, max(L, R - depth):R].sum() >= min_hits]
    else:
        raise ValueError(side)
    return group_tick_positions([float(v) for v in hits], gap=2.0)


def group_tick_positions(positions: list[float], gap: float = 2.0) -> list[float]:
    if not positions:
        return []
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [float(np.mean(g)) for g in groups]


def remove_ticks(mask: np.ndarray, panel: dict, depth: int = 8) -> np.ndarray:
    """Erase inward tick marks on all four sides of a panel.

    Ticks are short runs attached to the frame.  Removing them before tracing
    prevents a column near the frame from reporting a tick as a curve value.
    """
    out = mask.copy()
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    for x in detect_ticks(mask, panel, "bottom", depth=depth):
        xi = int(round(x))
        out[max(T, B - depth - 1):B + 1, max(0, xi - 1):xi + 2] = False
    for x in detect_ticks(mask, panel, "top", depth=depth):
        xi = int(round(x))
        out[T:min(B, T + depth + 1), max(0, xi - 1):xi + 2] = False
    for y in detect_ticks(mask, panel, "left", depth=depth):
        yi = int(round(y))
        out[max(0, yi - 1):yi + 2, L:min(R, L + depth + 1)] = False
    for y in detect_ticks(mask, panel, "right", depth=depth):
        yi = int(round(y))
        out[max(0, yi - 1):yi + 2, max(L, R - depth - 1):R + 1] = False
    return out


def fit_axis(px: list[float], values: list[float], log: bool = False) -> dict:
    """Least-squares axis fit; returns slope/intercept and residuals."""
    x = np.asarray(px, dtype=float)
    y = np.asarray(values, dtype=float)
    if log:
        y = np.log10(y)
    coef = np.polyfit(x, y, 1)
    resid = np.polyval(coef, x) - y
    return {
        "px": [float(v) for v in x],
        "values": [float(v) for v in values],
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "max_abs_residual": float(np.max(np.abs(resid))),
        "residual": [float(v) for v in resid],
        "log10": bool(log),
    }


# --------------------------------------------------------------------------- #
# connected components
# --------------------------------------------------------------------------- #
def components(mask: np.ndarray, min_area: int = 1, max_area: int = 10 ** 9,
               structure: np.ndarray | None = None) -> list[dict]:
    lab, n = ndimage.label(mask, structure=structure or np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if not (min_area <= area <= max_area):
            continue
        yy, xx = np.where(m)
        out.append({
            "label": i,
            "area": area,
            "x0": int(sl[1].start), "x1": int(sl[1].stop - 1),
            "y0": int(sl[0].start), "y1": int(sl[0].stop - 1),
            "width": int(sl[1].stop - sl[1].start),
            "height": int(sl[0].stop - sl[0].start),
            "x": float(xx.mean() + sl[1].start),
            "y": float(yy.mean() + sl[0].start),
        })
    return out


# --------------------------------------------------------------------------- #
# calibration record
# --------------------------------------------------------------------------- #
def make_calibration(figure: str, pdf: str, page: int, report_page: int,
                     image_meta: dict, panels: list[dict],
                     x_axis: dict, y_axis: dict, notes: list[str]) -> dict:
    return {
        "figure": figure,
        "pdf": pdf,
        "pdf_page": page,
        "report_page": report_page,
        "image": image_meta,
        "panels": panels,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "notes": notes,
    }


def save_calibration(path: Path, obj: dict) -> Path:
    return write_json(path, obj)
