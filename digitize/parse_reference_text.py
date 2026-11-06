"""Transcribe the two text-layer reference tables required by the figure list.

  NASA/TM-2002-211798 Table 1  fan operating conditions (report p. 2)
  NASA/TM-2003-212329          HW1/HW2 and LDV1/LDV2 axial stations (pp. 2--3)

These are transcriptions, not digitisation.  Every value is read from the PDF
text layer and carries its page and word-box coordinates.  The Podboy values
are also cross-checked against the inch-to-centimetre conversion printed in
the same sentence.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pymupdf as fitz

from common import GEO, PROV, ROOT, sha256, write_csv, write_json

NALLASAMY = ROOT / "20030003692.pdf"
PODBOY = ROOT / "20030065891.pdf"


def numeric(text: str) -> float:
    return float(text.replace(",", ""))


def word_at(page, text: str, occurrence: int = 0) -> dict:
    hits = [w for w in page.get_text("words") if w[4] == text]
    if len(hits) <= occurrence:
        raise RuntimeError(f"word {text!r} occurrence {occurrence} not found")
    w = hits[occurrence]
    return {"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4]}


def nallasamy_table1() -> tuple[list[dict], list[dict]]:
    doc = fitz.open(NALLASAMY)
    page = doc[5]  # PDF p6 = report p2
    rows = []
    for setting, y in (("Approach", 608.9), ("Cutback", 622.3), ("Takeoff", 635.7)):
        words = [w for w in page.get_text("words") if abs(w[1] - y) <= 1.5]
        label = [w for w in words if 80 <= w[0] <= 140]
        rpm = [w for w in words if 155 <= w[0] <= 200 and re.fullmatch(r"[\d,]+", w[4])]
        tip = [w for w in words if 220 <= w[0] <= 270 and re.fullmatch(r"[\d,]+", w[4])]
        if len(label) != 1 or len(rpm) != 1 or len(tip) != 1:
            raise RuntimeError(f"Table 1 {setting}: label={label}, rpm={rpm}, tip={tip}")
        rows.append({
            "table": "Table 1",
            "power_setting": setting,
            "corrected_fan_rpm": f"{numeric(rpm[0][4]):g}",
            "corrected_tip_speed_fps": f"{numeric(tip[0][4]):g}",
            "source_pdf": NALLASAMY.name,
            "source_page_pdf": 6,
            "source_page_report": 2,
            "source_y_pt": f"{y:.2f}",
            "source_rpm_x_pt": f"{rpm[0][0]:.2f}",
            "source_tip_x_pt": f"{tip[0][0]:.2f}",
            "method": "text-layer word boxes; comma removed",
        })
    checks = [
        {"check": "Table 1 rows", "detail": f"{len(rows)}/3 operating points",
         "pass": len(rows) == 3},
        {"check": "Table 1 values",
         "detail": "; ".join(f"{r['power_setting']} {r['corrected_fan_rpm']} rpm / "
                             f"{r['corrected_tip_speed_fps']} fps" for r in rows),
         "pass": [(r["power_setting"], r["corrected_fan_rpm"], r["corrected_tip_speed_fps"])
                  for r in rows] == [("Approach", "7808", "750"),
                                     ("Cutback", "11074", "1060"),
                                     ("Takeoff", "12656", "1215")]},
    ]
    return rows, checks


def podboy_locations() -> tuple[list[dict], list[dict]]:
    doc = fitz.open(PODBOY)
    p6 = " ".join(doc[5].get_text().split())
    p7 = " ".join(doc[6].get_text().split())

    # LDV stations: 3.12 / 6.49 in (7.93 / 16.48 cm), report p. 2
    m_ldv = re.search(
        r"corresponding to ([\d.]+) and ([\d.]+) inches \(([\d.]+) and ([\d.]+) cm\) "
        r"downstream of the static position of the tip trailing edge", p6)
    # HW stations: 2.96 / 5.89 in (7.52 / 15.0 cm), report p. 3
    m_hw = re.search(
        r"The two axial locations are ([\d.]+) and ([\d.]+) inches \(([\d.]+) and ([\d.]+) cm\) "
        r"downstream of the static position of the rotor tip trailing edge", p7)
    if not m_ldv or not m_hw:
        raise RuntimeError("Podboy station sentences not found")

    rows = []
    for station, inch, cm, page_pdf, page_report, technique in (
        ("LDV1", m_ldv.group(1), m_ldv.group(3), 6, 2, "LDV"),
        ("LDV2", m_ldv.group(2), m_ldv.group(4), 6, 2, "LDV"),
        ("HW1", m_hw.group(1), m_hw.group(3), 7, 3, "hot wire"),
        ("HW2", m_hw.group(2), m_hw.group(4), 7, 3, "hot wire"),
    ):
        rows.append({
            "station": station,
            "technique": technique,
            "axial_distance_in": inch,
            "axial_distance_cm": cm,
            "reference": "static position of the rotor tip trailing edge",
            "source_pdf": PODBOY.name,
            "source_page_pdf": page_pdf,
            "source_page_report": page_report,
            "method": "regex transcription from PDF text layer",
        })

    checks = [
        {"check": "Podboy station rows", "detail": f"{len(rows)}/4 stations",
         "pass": len(rows) == 4},
        {"check": "Podboy inch/cm consistency",
         "detail": "; ".join(f"{r['station']} {r['axial_distance_in']} in = "
                             f"{float(r['axial_distance_in'])*2.54:.3f} cm "
                             f"(printed {r['axial_distance_cm']})" for r in rows),
         "pass": all(abs(float(r["axial_distance_in"]) * 2.54 -
                         float(r["axial_distance_cm"])) <= 0.05 for r in rows)},
        {"check": "HW upstream of LDV",
         "detail": "HW1 2.96 < LDV1 3.12; HW2 5.89 < LDV2 6.49 in",
         "pass": (float(rows[2]["axial_distance_in"]) < float(rows[0]["axial_distance_in"])
                  and float(rows[3]["axial_distance_in"]) < float(rows[1]["axial_distance_in"]))},
    ]
    return rows, checks


def main() -> dict:
    t1, c1 = nallasamy_table1()
    loc, c2 = podboy_locations()
    checks = c1 + c2
    write_csv(GEO / "nallasamy_table1.csv", t1, list(t1[0].keys()))
    write_csv(GEO / "podboy_survey_locations.csv", loc, list(loc[0].keys()))
    write_csv(GEO / "reference_text_checks.csv", checks, list(checks[0].keys()))

    meta = {
        "sources": [
            {"pdf": NALLASAMY.name, "sha256": sha256(NALLASAMY),
             "content": "Table 1 fan operating conditions", "page_pdf": 6, "page_report": 2},
            {"pdf": PODBOY.name, "sha256": sha256(PODBOY),
             "content": "HW1/HW2 and LDV1/LDV2 axial stations", "page_pdf": "6-7",
             "page_report": "2-3"},
        ],
        "method": "PDF text layer; word boxes for Table 1; regex for station sentences",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": {"nallasamy_table1": len(t1), "podboy_survey_locations": len(loc)},
        "checks": checks,
    }
    write_json(PROV / "reference_text.json", meta)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return meta


if __name__ == "__main__":
    main()
