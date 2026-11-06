"""Build the source inventory and figure/table map for the Day-1 data set.

Everything here is descriptive metadata: checksums, titles, page numbers and
which extraction method was used.  No plot values are read by this script.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pymupdf as fitz

from common import PROV, ROOT, write_csv, write_json, sha256

INVENTORY = ROOT / "data" / "inventory"

SOURCES = [
    {
        "file": "20030003692.pdf",
        "title": "Fan Noise Source Diagnostic Test - Computation of Rotor Wake Turbulence Noise",
        "report": "NASA/TM-2002-211798; AIAA-2002-2489",
        "authors": "Nallasamy, Envia, Thorp, Shabbir",
        "date": "August 2002",
        "role": "measured exhaust power spectra (Figs. 9-12) and prediction context",
        "used_pages": "PDF p. 14 = report p. 10 (Figs. 9-12); report pp. 3-6 for text checks",
    },
    {
        "file": "20050175875.pdf",
        "title": "Fan Noise Source Diagnostic Test - Rotor Alone Aerodynamic Performance Results",
        "report": "NASA/TM-2005-211681; AIAA-2002-2426",
        "authors": "Hughes, Jeracki, Woodward, Miller",
        "date": "April 2005",
        "role": "geometry, fan/OGV performance and flowpath",
        "used_pages": "PDF p. 12 = report p. 8 (Tables 1-3); PDF p. 22 = report p. 18 (Fig. 15)",
    },
    {
        "file": "20030065891.pdf",
        "title": "Steady and Unsteady Flow Field Measurements Within a NASA 22-Inch Fan Model",
        "report": "NASA/TM-2003-212329; AIAA-2002-1033",
        "authors": "Podboy, Krupar, Helland, Hughes",
        "date": "July 2003",
        "role": "wake spectra and integral length scales (Figs. 26-28); source screening only",
        "used_pages": "PDF p. 33 = report p. 29 (Figs. 26-27); PDF p. 34 = report p. 30 (Fig. 28)",
    },
    {
        "file": "Fernandes.pdf",
        "title": "Fan Noise Predictions of the NASA Source Diagnostic Test using Unsteady Simulations with LAVA Part II - Tonal and Broadband Noise Assessment",
        "report": "NASA Ames Research Center manuscript",
        "authors": "Fernandes, Housman, Stich, Duensing",
        "date": "preprint",
        "role": "54-vane exhaust PWL vector cross-check (Fig. 18c)",
        "used_pages": "PDF p. 24 (Fig. 18c)",
    },
    {
        "file": "Part2_acoustics_856953.pdf",
        "title": "ACAT1 Benchmark of RANS-informed Analytical Methods for Fan Broadband Noise Prediction: Part II - Influence of the Acoustic Models",
        "report": "ACAT1 Part II article",
        "authors": "Guerin et al.",
        "date": "preprint",
        "role": "method reference for the strip-theory model (Days 2-3); no Day-1 data",
        "used_pages": "not digitised",
    },
]

FIGURES = [
    # figure, pdf, pdf_page, report_page, content, method, dataset, status
    ("Figure 9", "20030003692.pdf", 14, 10, "54-vane radial, approach, exhaust PWL spectrum (TL)",
     "raster dash morphology, 400 dpi", "tm2002_p14_fig09_54vane_exhaust", "digitised"),
    ("Figure 10", "20030003692.pdf", 14, 10, "26-vane radial, approach, exhaust PWL spectrum (BL)",
     "raster dash morphology, 400 dpi", "tm2002_p14_fig10_26vane_exhaust", "digitised"),
    ("Figure 11", "20030003692.pdf", 14, 10, "26-vane swept, approach, exhaust PWL spectrum (TR)",
     "raster dash morphology, 400 dpi", "tm2002_p14_fig11_26swept_exhaust", "digitised"),
    ("Figure 12", "20030003692.pdf", 14, 10, "tip-speed comparison (BR)",
     "not used", "", "not used"),
    ("Figure 18(c)", "Fernandes.pdf", 24, 24, "54-vane exhaust PWL spectrum, experimental curve",
     "vector path extraction (PyMuPDF)", "fernandes_fig18c_54vane_exhaust_pwl", "digitised"),
    ("Figure 15", "20050175875.pdf", 22, 18, "model inner/outer flowpath with OGVs and RAN",
     "vector path extraction (PyMuPDF)", "fig15_annulus_mach", "digitised"),
    ("Table 1", "20050175875.pdf", 12, 8, "R4 fan design parameters",
     "text-layer word boxes", "sdt_fan_design", "digitised"),
    ("Table 2", "20050175875.pdf", 12, 8, "fan and OGV airfoil geometries",
     "text-layer word boxes", "sdt_geometry", "digitised"),
    ("Table 3", "20050175875.pdf", 12, 8, "fan performance at approach/cutback/takeoff",
     "text-layer word boxes", "sdt_performance", "digitised"),
    ("Figure 26", "20030065891.pdf", 33, 29, "upwash PSDs at 25/81/97 % span, 61.7 % speed",
     "raster screening only", "", "screened, not digitised"),
    ("Figure 27", "20030065891.pdf", 33, 29, "experimental vs von Karman PSDs",
     "raster screening only", "", "screened, not digitised"),
    ("Figure 28", "20030065891.pdf", 34, 30, "radial integral length scales, 61.7 % speed",
     "raster screening only", "", "screened, not digitised"),
]


def main() -> dict:
    rows = []
    for s in SOURCES:
        p = ROOT / s["file"]
        doc = fitz.open(p)
        rows.append({
            **s,
            "pages": doc.page_count,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    write_csv(INVENTORY / "pdf_inventory.csv", rows, list(rows[0].keys()))

    fig_rows = []
    for fig, pdf, pdf_page, report_page, content, method, dataset, status in FIGURES:
        fig_rows.append({
            "figure": fig,
            "source_pdf": pdf,
            "source_page_pdf": pdf_page,
            "source_page_report": report_page,
            "content": content,
            "method": method,
            "dataset_id": dataset,
            "status": status,
        })
    write_csv(INVENTORY / "figure_map.csv", fig_rows, list(fig_rows[0].keys()))

    # checksums for every PDF in the project root
    checks = [(sha256(p), p.name) for p in sorted(ROOT.glob("*.pdf"))]
    (ROOT / "data" / "raw" / "checksums.sha256").write_text(
        "\n".join(f"{h}  {n}" for h, n in checks) + "\n")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": rows,
        "figures": fig_rows,
        "checksums_file": "data/raw/checksums.sha256",
    }
    write_json(PROV / "source_inventory.json", meta)

    print(f"inventory: {len(rows)} PDFs, {len(fig_rows)} figures/tables")
    for r in rows:
        print(f"  {r['file']}: {r['pages']} pp, {r['sha256'][:12]}...")
    return meta


if __name__ == "__main__":
    main()
