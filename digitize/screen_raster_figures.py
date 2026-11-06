"""Palette-aware screening of the Podboy raster figures (Figs. 26--28).

This replaces the earlier grayscale screen, which converted the indexed
CalRGB images to luminance and therefore collapsed the red and blue curves
into one dark mask.  The embedded images are not 1-bit scans: Figs. 26/27
contain black + red on white, and Fig. 28 contains black + red + blue on
white.  Colour separation is exact and deterministic.

The screen records, for every figure and panel, the palette, the frame
bounds, and the connected-component census per colour.  It does not claim
numeric values; digitisation is done by the dedicated extraction scripts.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from common import PROV, QA, ROOT, sha256, write_csv, write_json
from raster_common import (colour_masks, components, detect_panels,
                           load_embedded, palette_census)

PDF = ROOT / "20030065891.pdf"

FIGURES = [
    {
        "figure": "Figure 26",
        "pdf_page": 33,
        "report_page": 29,
        "image_index": 0,
        "caption": "PSDs computed from upwash velocities in the rotor wake at 61.7 % speed",
        "axis_x": "frequency, 10^2 to 10^4 Hz",
        "axis_y": "PSD, 10^1 to 10^-4 ft^2/sec^2/Hz",
        "panels": "97 %, 81 %, 25 % span (left to right)",
        "legend": "black = HW1, red = HW2 (report p. 8 text)",
        "status": "palette-separated; digitised by extract_podboy_fig26.py",
    },
    {
        "figure": "Figure 27",
        "pdf_page": 33,
        "report_page": 29,
        "image_index": 1,
        "caption": "Comparison of experimental and von Karman model PSDs",
        "axis_x": "frequency, 10^2 to 10^4 Hz",
        "axis_y": "PSD, 10^1 to 10^-4 ft^2/sec^2/Hz",
        "panels": "97 %, 81 %, 25 % span (left to right)",
        "legend": "experimental + von Karman model; screening only",
        "status": "screened only (not in extraction scope)",
    },
    {
        "figure": "Figure 28",
        "pdf_page": 34,
        "report_page": 30,
        "image_index": 0,
        "caption": "Radial distributions of turbulence integral length scale at 61.7 % speed",
        "axis_x": "radius, 5 to 11 in",
        "axis_y": "integral length scale, 0 to 0.4 in",
        "panels": "single panel",
        "legend": "black Streamwise, red Upwash, blue Transverse; line = HW1, symbols = HW2",
        "status": "palette-separated; digitised by extract_podboy_fig28.py",
    },
]


def panel_census(rgb: np.ndarray, panel: dict) -> dict:
    """Component census for each palette colour inside a panel interior."""
    masks = colour_masks(rgb)
    L, T, R, B = panel["L"], panel["T"], panel["R"], panel["B"]
    out = {}
    for name in ("black", "red", "blue", "grey"):
        m = masks[name][T + 2:B - 1, L + 2:R - 1]
        if not m.any():
            out[name] = {"n": 0, "largest_area": 0, "largest_width": 0,
                         "largest_height": 0, "small_le4": 0}
            continue
        comps = components(m, min_area=1)
        comps.sort(key=lambda c: -c["area"])
        big = comps[0]
        out[name] = {
            "n": len(comps),
            "largest_area": big["area"],
            "largest_width": big["width"],
            "largest_height": big["height"],
            "small_le4": sum(1 for c in comps if c["area"] <= 4),
        }
    return out


def main() -> dict:
    rows = []
    meta = {
        "pdf": PDF.name,
        "pdf_sha256": sha256(PDF),
        "method": "exact palette separation + connected-component census",
        "supersedes": "raster_figure_screening_grayscale_invalid.csv",
        "conclusion": (
            "The embedded figures are 8-bpc indexed CalRGB images, not 1-bit "
            "scans.  Figs. 26/27 use black + red; Fig. 28 uses black + red + "
            "blue.  The earlier grayscale screen collapsed those colours and "
            "wrongly concluded the curves could not be separated.  Colour "
            "separation is exact; Figs. 26 and 28 are digitised, Fig. 27 is "
            "outside scope."
        ),
        "figures": [],
    }
    for fig in FIGURES:
        rgb, path, info, rect = load_embedded(PDF, fig["pdf_page"], fig["image_index"])
        census = palette_census(rgb)
        masks = colour_masks(rgb)
        panels = detect_panels(masks["black"])
        panel_rows = []
        for pi, panel in enumerate(panels, 1):
            pc = panel_census(rgb, panel)
            panel_rows.append({"panel": pi, **{k: panel[k] for k in ("L", "T", "R", "B")},
                               "width": panel["width"], "height": panel["height"],
                               "colours": pc})
            print(f"  {fig['figure']} panel {pi}: frame "
                  f"({panel['L']},{panel['T']})-({panel['R']},{panel['B']}) "
                  f"{panel['width']}x{panel['height']} px; "
                  f"black {pc['black']['n']} comps (largest {pc['black']['largest_area']}), "
                  f"red {pc['red']['n']} (largest {pc['red']['largest_area']}), "
                  f"blue {pc['blue']['n']} (largest {pc['blue']['largest_area']})")

        row = {
            "figure": fig["figure"],
            "source_pdf": PDF.name,
            "source_page_pdf": fig["pdf_page"],
            "source_page_report": fig["report_page"],
            "caption": fig["caption"],
            "image_file": str(path.relative_to(ROOT)),
            "native_width_px": info["width"],
            "native_height_px": info["height"],
            "bits_per_component": info["bpc"],
            "colorspace": info["colorspace"],
            "n_palette_colours": len(census),
            "palette": "; ".join(
                f"rgb({c['rgb'][0]},{c['rgb'][1]},{c['rgb'][2]})={c['count']}"
                for c in census),
            "axis_x": fig["axis_x"],
            "axis_y": fig["axis_y"],
            "panels": fig["panels"],
            "legend": fig["legend"],
            "n_panels_detected": len(panels),
            "panel_frames_px": "; ".join(
                f"({p['L']},{p['T']})-({p['R']},{p['B']})" for p in panels),
            "black_components": "; ".join(
                f"{p['colours']['black']['n']} (largest {p['colours']['black']['largest_area']})"
                for p in panel_rows),
            "red_components": "; ".join(
                f"{p['colours']['red']['n']} (largest {p['colours']['red']['largest_area']})"
                for p in panel_rows),
            "blue_components": "; ".join(
                f"{p['colours']['blue']['n']} (largest {p['colours']['blue']['largest_area']})"
                for p in panel_rows),
            "grid_dots_le4px": "; ".join(
                str(p["colours"]["black"]["small_le4"]) for p in panel_rows),
            "status": fig["status"],
            "numeric_values_claimed": 0,
        }
        rows.append(row)
        meta["figures"].append({**fig, "image": info,
                                "placement_pt": info["placement_pt"],
                                "palette": census, "panels": panel_rows})

    write_csv(QA / "raster_figure_screening.csv", rows, list(rows[0].keys()))
    meta["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(QA / "raster_screening.json", meta)
    return meta


if __name__ == "__main__":
    main()
