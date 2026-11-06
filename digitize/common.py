"""Shared helpers for the UHBR Day-1 digitisation pipeline.

All methods are deterministic pixel/vector operations.  No vision model is used.
Classical OCR was used once, off-line, only to locate the in-plot "Data" and
"Theory" labels; those boxes are stored in the calibration JSON and verified
here by pixel occupancy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pymupdf as fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CAL = DATA / "calibration"
DIG = DATA / "digitized"
GEO = DATA / "geometry"
DER = DATA / "derived"
PROV = DATA / "provenance"
QA = DATA / "qa"
OVER = QA / "overlays"

for _d in (RAW, CAL, DIG, GEO, DER, PROV, QA, OVER):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# source integrity
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest() -> list[dict]:
    """Checksums of every PDF used by the pipeline."""
    out = []
    for name in (
        "20030003692.pdf",   # NASA/TM-2002-211798
        "Fernandes.pdf",      # LAVA Part II
        "20050175875.pdf",    # NASA/TM-2005-211681
        "20030065891.pdf",    # NASA/TM-2003-212329
    ):
        p = ROOT / name
        if p.exists():
            out.append({"file": name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    return out


def write_checksums() -> Path:
    path = RAW / "checksums.sha256"
    lines = [f"{d['sha256']}  {d['file']}" for d in source_manifest()]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_page(pdf: Path, page: int, dpi: int = 400, gray: bool = True) -> np.ndarray:
    """Render a 1-based PDF page to a uint8 numpy array, with a PNG cache.

    The cache name includes the PDF stem, page and dpi so a change of source
    cannot silently reuse a stale image.
    """
    cache = RAW / "render" / f"{pdf.stem}_p{page:03d}_{dpi}dpi_{'gray' if gray else 'rgb'}.png"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        im = Image.open(cache)
        return np.array(im.convert("L" if gray else "RGB"))

    doc = fitz.open(pdf)
    page_obj = doc[page - 1]
    pix = page_obj.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY if gray else fitz.csRGB)
    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = buf.reshape(pix.h, pix.stride)[:, : pix.w].copy()
    Image.fromarray(arr).save(cache)
    return arr


# --------------------------------------------------------------------------- #
# binary helpers
# --------------------------------------------------------------------------- #
def runs(v: Sequence[bool]) -> list[tuple[int, int]]:
    """Inclusive runs of True in a 1-D boolean sequence."""
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(v):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(v) - 1))
    return out


def group_positions(pos: Iterable[int], gap: int = 2) -> list[tuple[float, int]]:
    """Group adjacent integer positions; return (mean, count)."""
    vals = list(pos)
    if not vals:
        return []
    groups = [[vals[0]]]
    for q in vals[1:]:
        if q - groups[-1][-1] <= gap:
            groups[-1].append(q)
        else:
            groups.append([q])
    return [(float(np.mean(g)), len(g)) for g in groups]


def detect_major_left_ticks(binary: np.ndarray, L: int, T: int, B: int,
                            min_run: int = 8) -> list[float]:
    ys = [y for y in range(T - 20, B + 21)
          if binary[y, max(0, L - 18):L - 2].sum() >= 3]
    return [y for y, n in group_positions(ys) if n >= min_run]


def detect_major_bottom_ticks(binary: np.ndarray, L: int, R: int, B: int) -> list[float]:
    xs = [x for x in range(L - 20, R + 21)
          if binary[B + 2:B + 18, x].sum() >= 3]
    groups = group_positions(xs)
    return [x for x, _ in groups]


def fit_y_axis(ticks_px: Sequence[float], top_db: float = 120.0,
               step_db: float = 10.0) -> dict:
    ticks = np.asarray(sorted(ticks_px), dtype=float)
    values = top_db - step_db * np.arange(len(ticks))
    coef = np.polyfit(ticks, values, 1)
    resid = np.polyval(coef, ticks) - values
    return {
        "tick_px": ticks.tolist(),
        "tick_db": values.tolist(),
        "slope_db_per_px": float(coef[0]),
        "intercept_db": float(coef[1]),
        "max_abs_residual_db": float(np.max(np.abs(resid))),
        "residual_db": resid.tolist(),
    }


def fit_x_axis(ticks_px: Sequence[float], exponents: Sequence[float]) -> dict:
    ticks = np.asarray(ticks_px, dtype=float)
    exps = np.asarray(exponents, dtype=float)
    coef = np.polyfit(ticks, exps, 1)
    resid = np.polyval(coef, ticks) - exps
    return {
        "tick_px": ticks.tolist(),
        "tick_log10_hz": exps.tolist(),
        "slope_decades_per_px": float(coef[0]),
        "intercept_decades": float(coef[1]),
        "max_abs_residual_decades": float(np.max(np.abs(resid))),
        "residual_decades": resid.tolist(),
    }


# --------------------------------------------------------------------------- #
# connected-component filter for the raster dash trace
# --------------------------------------------------------------------------- #
def extract_dashes(gray: np.ndarray, panel: dict,
                   threshold: int = 170,
                   margin_left: int = 14, margin_top: int = 14,
                   margin_right: int = 8, margin_bottom: int = 8,
                   area_range: tuple[int, int] = (4, 120),
                   max_width: int = 24, max_height: int = 12,
                   max_hole_fraction: float = 0.05) -> list[dict]:
    """Extract small, solid, dash-like components from a panel interior.

    Measured data are the dashed curve: isolated 2--3 px thick dashes.  The
    theory curve is a continuous solid line with open circles; the solid line
    forms long components and the circles have a high hole fraction, so both
    are rejected by these morphology gates.
    """
    from scipy import ndimage

    L, T, R, B = panel["frame_px"]
    binary = gray < threshold

    y0 = T + margin_top
    y1 = B - margin_bottom
    x0 = L + margin_left
    x1 = R - margin_right
    sub = binary[y0:y1, x0:x1].copy()

    # mask the in-plot words "Data" and "Theory", expanded by 4 px
    for bx0, bx1, by0, by1 in panel.get("label_boxes_px", []):
        sub[max(0, by0 - 4 - y0):max(0, by1 + 4 - y0),
            max(0, bx0 - 4 - x0):max(0, bx1 + 4 - x0)] = False

    # mask inward bottom major-tick columns (they extend ~11 px above the frame)
    for xtick in panel.get("x_major_ticks_px", []):
        ix = int(round(xtick)) - x0
        if 0 <= ix < sub.shape[1]:
            sub[max(0, sub.shape[0] - 14):, max(0, ix - 2):ix + 3] = False

    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    objects = ndimage.find_objects(lab)
    out: list[dict] = []
    for idx, sl in enumerate(objects, 1):
        if sl is None:
            continue
        mask = lab[sl] == idx
        area = int(mask.sum())
        if not (area_range[0] <= area <= area_range[1]):
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if w > max_width or h > max_height:
            continue
        filled = int(ndimage.binary_fill_holes(mask).sum())
        hole = (filled - area) / filled if filled else 0.0
        if hole > max_hole_fraction:
            continue
        yy, xx = np.where(mask)
        gxx = xx + sl[1].start
        gyy = yy + sl[0].start
        # reject components touching the crop edge (frame / tick residue)
        if gxx.min() == 0 or gyy.min() == 0 or gxx.max() == sub.shape[1] - 1 or gyy.max() == sub.shape[0] - 1:
            continue
        out.append({
            "x_px": float(gxx.mean() + x0),
            "y_px": float(gyy.mean() + y0),
            "area_px": area,
            "width_px": int(w),
            "height_px": int(h),
            "hole_fraction": float(hole),
        })
    out.sort(key=lambda d: d["x_px"])
    return out


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    return path


def fnum(value: float, digits: int = 4) -> str:
    """Stable numeric formatting for CSV/JSON."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{value:.{digits}f}"
