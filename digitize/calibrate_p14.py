"""Calibrate the four panels of NASA/TM-2002-211798 page 14 (report p. 10).

The page is an image-only figure plate.  Frames and major ticks are detected
deterministically at 400 dpi; the three x tick labels (10^3, 10^4, 10^5) were
read once by glyph inspection and are recorded here as a one-time calibration.
The y-axis range (70--120 dB) and the in-plot word boxes come from the same
one-time inspection.  No plot values are read by OCR or by a vision model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from common import (CAL, ROOT, fit_x_axis, fit_y_axis, render_page,
                    detect_major_bottom_ticks, detect_major_left_ticks,
                    write_json, sha256)

PDF = ROOT / "20030003692.pdf"
PAGE = 14  # 1-based PDF page; report page 10

# Frame corners measured from long dark runs at 400 dpi (x0, y0, x1, y1).
PANELS = {
    "TL": {
        "figure": "Figure 9",
        "description": "Baseline stator (54 radial vanes), approach, exhaust",
        "frame_px": [528, 825, 1638, 1626],
        "label_boxes_px": [[673, 741, 1053, 1078], [714, 819, 1280, 1314]],
        "label_text": ["Data", "Theory"],
        "label_ocr_confidence": [None, None],  # filled below if hOCR box was used
    },
    "BL": {
        "figure": "Figure 10",
        "description": "Low-count stator (26 radial vanes), approach, exhaust",
        "frame_px": [528, 2299, 1638, 3100],
        "label_boxes_px": [[616, 687, 2559, 2587], [706, 812, 2750, 2784]],
        "label_text": ["Data", "Theory"],
        "label_ocr_confidence": [96, 96],
    },
    "TR": {
        "figure": "Figure 11",
        "description": "Low-noise stator (26 swept vanes), approach, exhaust",
        "frame_px": [1868, 818, 2978, 1620],
        "label_boxes_px": [[1973, 2043, 1038, 1066], [2071, 2177, 1278, 1313]],
        "label_text": ["Data", "Theory"],
        "label_ocr_confidence": [95, 94],
    },
}

X_EXPONENTS = [3.0, 4.0, 5.0]
Y_TOP_DB = 120.0


def frame_occupancy(binary: np.ndarray, frame: list[int]) -> dict:
    L, T, R, B = frame
    top = binary[T, L:R + 1].mean()
    bottom = binary[B, L:R + 1].mean()
    left = binary[T:B + 1, L].mean()
    right = binary[T:B + 1, R].mean()
    return {"top": float(top), "bottom": float(bottom),
            "left": float(left), "right": float(right)}


def main() -> dict:
    gray = render_page(PDF, PAGE, dpi=400)
    binary = gray < 170
    out = {
        "pdf": PDF.name,
        "pdf_sha256": sha256(PDF),
        "pdf_page": PAGE,
        "report_page": 10,
        "dpi": 400,
        "threshold_gray": 170,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axis_labels": {
            "x": "Frequency, Hz",
            "y": "Exhaust PWL Spectrum, dB",
            "x_method": "OCR of rotated/axis titles at 400 dpi plus glyph inspection",
            "y_method": "OCR of rotated y-axis title at 400 dpi plus glyph inspection",
        },
        "x_range_hz": [1000.0, 100000.0],
        "y_range_db": [70.0, 120.0],
        "panels": {},
    }

    for name, p in PANELS.items():
        L, T, R, B = p["frame_px"]
        occ = frame_occupancy(binary, p["frame_px"])
        if min(occ.values()) < 0.95:
            raise RuntimeError(f"{name}: frame not found at {p['frame_px']}: {occ}")

        y_ticks = detect_major_left_ticks(binary, L, T, B, min_run=8)
        if len(y_ticks) != 6:
            raise RuntimeError(f"{name}: expected 6 y major ticks, got {y_ticks}")
        yfit = fit_y_axis(y_ticks, top_db=Y_TOP_DB, step_db=10.0)
        if yfit["max_abs_residual_db"] > 0.05:
            raise RuntimeError(f"{name}: y calibration residual too large: {yfit}")

        all_x = detect_major_bottom_ticks(binary, L, R, B)
        wanted = [L + 0.0, (L + R) / 2.0, R + 0.0]
        x_ticks = []
        for w in wanted:
            cand = [x for x in all_x if abs(x - w) <= 4.0]
            if not cand:
                raise RuntimeError(f"{name}: no x tick near {w}: {all_x}")
            x_ticks.append(min(cand, key=lambda x: abs(x - w)))
        xfit = fit_x_axis(x_ticks, X_EXPONENTS)
        if xfit["max_abs_residual_decades"] > 5e-4:
            raise RuntimeError(f"{name}: x calibration residual too large: {xfit}")

        # verify the stored label boxes actually contain dark glyphs
        lab_check = []
        for (bx0, bx1, by0, by1) in p["label_boxes_px"]:
            frac = binary[by0:by1 + 1, bx0:bx1 + 1].mean()
            lab_check.append(round(float(frac), 4))
            if frac < 0.02:
                raise RuntimeError(f"{name}: label box appears empty: {(bx0, bx1, by0, by1)}")

        panel = {
            "figure": p["figure"],
            "description": p["description"],
            "frame_px": p["frame_px"],
            "frame_occupancy": {k: round(v, 4) for k, v in occ.items()},
            "x_major_ticks_px": [round(x, 2) for x in x_ticks],
            "x_tick_log10_hz": X_EXPONENTS,
            "x_tick_labels": ["10^3", "10^4", "10^5"],
            "x_fit": xfit,
            "y_major_ticks_px": [round(y, 2) for y in y_ticks],
            "y_tick_db": yfit["tick_db"],
            "y_fit": yfit,
            "label_boxes_px": p["label_boxes_px"],
            "label_text": p["label_text"],
            "label_ocr_confidence": p["label_ocr_confidence"],
            "label_box_dark_fraction": lab_check,
            "notes": (
                "Measured data = dashed curve; theory = solid line through "
                "open-circle calculation symbols.  This assignment is fixed "
                "by morphology and by the report text (symbols mark computed "
                "points; lines through symbols are curve fits)."
            ),
        }
        out["panels"][name] = panel
        print(f"{name}: frame ok; y ticks {y_ticks}; y residual {yfit['max_abs_residual_db']:.4f} dB; "
              f"x ticks {x_ticks}; x residual {xfit['max_abs_residual_decades']:.2e} decades")

    path = CAL / "p14_panels.json"
    write_json(path, out)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
