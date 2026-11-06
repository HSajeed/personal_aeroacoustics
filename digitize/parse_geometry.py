"""Parse the SDT geometry and performance tables from NASA/TM-2005-211681.

Source: 20050175875.pdf, PDF page 12 = report page 8.
  Table 1  Fan design parameters
  Table 2  Fan and OGV airfoil geometries
  Table 3  Fan performance at the three acoustic rating points

The PDF has a text layer, so every cell is read from its actual word box
(x/y coordinates), not from the linear text order.  Table 2 has blank cells
(Fan hub/tip solidity and stagger) and cells reported as "---" (Fan sweep and
camber); those are emitted explicitly so the table shape is auditable.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pymupdf as fitz

from common import GEO, PROV, ROOT, write_csv, write_json, sha256

PDF = ROOT / "20050175875.pdf"
PDF_PAGE = 12          # 1-based PDF page
REPORT_PAGE = 8
TABLE_1_Y = [85.0, 95.0, 107.5, 117.5, 130.0, 140.0, 152.5, 162.5, 175.0]
TABLE_3_Y = [575.0, 587.5, 602.5]

# Table 2 column bands, from the word-box centres.  The Fan column is centred
# near x=278, Baseline near 346, Low Count near 411, Low Noise near 479.
TABLE2_COLUMNS = [
    ("Fan", 250.0, 305.0),
    ("Baseline OGV (54)", 320.0, 375.0),
    ("Low Count OGV (26)", 385.0, 440.0),
    ("Low Noise OGV (26)", 455.0, 510.0),
]

TABLE2_ROWS = [
    # y_pt, group, parameter, location, unit, expected status
    (252.5, "global",   "blade_or_vane_count", "",          "count", "4 numeric"),
    (265.0, "global",   "aft_sweep",           "",          "deg",   "3 numeric, 1 dash"),
    (275.0, "global",   "aspect_ratio",        "pitchline", "-",     "4 numeric"),
    (287.5, "global",   "chord",               "pitchline", "in",    "4 numeric"),
    (310.0, "solidity", "solidity",            "hub",       "-",     "3 numeric, 1 blank"),
    (320.0, "solidity", "solidity",            "pitchline", "-",     "4 numeric"),
    (332.5, "solidity", "solidity",            "tip",       "-",     "3 numeric, 1 blank"),
    (355.0, "stagger",  "stagger",             "hub",       "deg",   "3 numeric, 1 blank"),
    (367.5, "stagger",  "stagger",             "pitchline", "deg",   "4 numeric"),
    (377.5, "stagger",  "stagger",             "tip",       "deg",   "3 numeric, 1 blank"),
    (400.0, "camber",   "vane_camber",         "hub",       "deg",   "3 numeric, 1 dash"),
    (412.5, "camber",   "vane_camber",         "pitchline", "deg",   "3 numeric, 1 dash"),
    (422.5, "camber",   "vane_camber",         "tip",       "deg",   "3 numeric, 1 dash"),
    (445.0, "tmax_c",   "tmax_over_c",         "hub",       "-",     "4 numeric"),
    (457.5, "tmax_c",   "tmax_over_c",         "pitchline", "-",     "4 numeric"),
    (467.5, "tmax_c",   "tmax_over_c",         "tip",       "-",     "4 numeric"),
]

TABLE1_ROWS = [
    # y_pt, parameter, unit
    (85.0,  "number_of_fan_blades",        "count"),
    (95.0,  "fan_tip_diameter",            "in"),
    (107.5, "radius_ratio",                "-"),
    (117.5, "corrected_tip_speed",         "ft/sec"),
    (130.0, "corrected_rpm",               "rpm"),
    (140.0, "corrected_fan_weight_flow",   "lbm/sec"),
    (152.5, "specific_flow",               "lbm/sec-ft2"),
    (162.5, "stage_pressure_ratio",        "-"),
    (175.0, "design_bypass_ratio",         "-"),
]

TABLE3_METRICS = [
    # metric, unit, RAN x-band, Fan/OGV x-band
    ("corrected_weight_flow",     "lbm/sec", (165.0, 205.0), (210.0, 250.0)),
    ("total_pressure_ratio",      "-",       (260.0, 300.0), (305.0, 345.0)),
    ("total_temperature_ratio",   "-",       (355.0, 395.0), (400.0, 440.0)),
    ("adiabatic_efficiency",      "-",       (445.0, 485.0), (488.0, 530.0)),
]

CONDITIONS = [
    (575.0, "Approach", 61.7),
    (587.5, "Cutback", 87.5),
    (602.5, "Takeoff", 100.0),
]


def words_on_line(page, y: float, tol: float = 1.6) -> list[dict]:
    out = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text, block, line, n = w
        if abs(y0 - y) <= tol:
            out.append({"x0": x0, "x1": x1, "xc": (x0 + x1) / 2.0,
                        "y0": y0, "text": text, "block": block, "line": line})
    return sorted(out, key=lambda d: d["x0"])


def numeric_value(text: str) -> float | None:
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def table2(tokens_by_y: dict[float, list[dict]]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    checks: list[dict] = []
    n_numeric = n_dash = n_blank = 0
    seen: set[tuple[float, str]] = set()

    for y, group, parameter, location, unit, expected in TABLE2_ROWS:
        tokens = tokens_by_y[y]
        for component, lo, hi in TABLE2_COLUMNS:
            cell = [t for t in tokens if lo <= t["xc"] <= hi]
            if len(cell) > 1:
                raise RuntimeError(f"Table 2 y={y} {component}: multiple cells {cell}")
            if cell:
                tok = cell[0]
                seen.add((y, component))
                text = tok["text"]
                value = numeric_value(text)
                if text == "---":
                    status = "dash"
                    note = "reported as --- (not applicable)"
                    n_dash += 1
                elif value is not None:
                    status = "numeric"
                    note = ""
                    n_numeric += 1
                else:
                    raise RuntimeError(f"Table 2 y={y} {component}: unparsable {text!r}")
                xpt, ypt = tok["x0"], tok["y0"]
            else:
                status = "blank"
                text = ""
                value = None
                note = "blank in source table"
                n_blank += 1
                xpt = ypt = None

            rows.append({
                "table": "Table 2",
                "component": component,
                "row_group": group,
                "parameter": parameter,
                "location": location,
                "value": "" if value is None else f"{value:g}",
                "value_text": text,
                "unit": unit,
                "status": status,
                "note": note,
                "source_pdf": PDF.name,
                "source_page_pdf": PDF_PAGE,
                "source_page_report": REPORT_PAGE,
                "source_y_pt": "" if ypt is None else f"{ypt:.2f}",
                "source_x_pt": "" if xpt is None else f"{xpt:.2f}",
            })

    # every numeric/--- token in the table region must fall inside a column band
    unclaimed = []
    for y, tokens in tokens_by_y.items():
        for t in tokens:
            if t["xc"] < 255.0:
                continue
            if numeric_value(t["text"]) is None and t["text"] != "---":
                continue
            if not any(lo <= t["xc"] <= hi for _, lo, hi in TABLE2_COLUMNS):
                unclaimed.append((y, t["text"], t["xc"]))
    if unclaimed:
        raise RuntimeError(f"unclaimed numeric tokens in Table 2 region: {unclaimed}")

    checks.append({
        "check": "Table 2 cell census",
        "table": "Table 2",
        "detail": f"64 cells: {n_numeric} numeric, {n_dash} '---', {n_blank} blank; "
                  f"{len(seen)} cells located by coordinate",
        "pass": (n_numeric, n_dash, n_blank) == (56, 4, 4),
    })
    return rows, checks


def table1(tokens_by_y: dict[float, list[dict]]) -> tuple[list[dict], list[dict]]:
    rows = []
    for y, parameter, unit in TABLE1_ROWS:
        cand = [t for t in tokens_by_y[y] if t["x0"] > 350.0 and numeric_value(t["text"]) is not None]
        if len(cand) != 1:
            raise RuntimeError(f"Table 1 y={y}: expected one numeric token, got {cand}")
        tok = cand[0]
        rows.append({
            "table": "Table 1",
            "component": "Fan",
            "parameter": parameter,
            "value": f"{numeric_value(tok['text']):g}",
            "value_text": tok["text"],
            "unit": unit,
            "source_pdf": PDF.name,
            "source_page_pdf": PDF_PAGE,
            "source_page_report": REPORT_PAGE,
            "source_y_pt": f"{tok['y0']:.2f}",
            "source_x_pt": f"{tok['x0']:.2f}",
        })
    checks = [{
        "check": "Table 1 completeness",
        "table": "Table 1",
        "detail": f"{len(rows)}/9 design parameters parsed",
        "pass": len(rows) == 9,
    }]
    return rows, checks


def table3(tokens_by_y: dict[float, list[dict]]) -> tuple[list[dict], list[dict]]:
    rows = []
    for y, condition, speed in CONDITIONS:
        tokens = tokens_by_y[y]
        # verify condition label from the words on the same line
        label = "".join(t["text"] for t in tokens if t["x0"] < 165.0)
        if not label.startswith(condition):
            raise RuntimeError(f"Table 3 y={y}: condition label mismatch: {label!r}")
        if f"{speed:g}" not in label:
            raise RuntimeError(f"Table 3 y={y}: speed label mismatch: {label!r}")
        for metric, unit, ran_band, fogv_band in TABLE3_METRICS:
            for config, (lo, hi) in (("RAN", ran_band), ("Fan/OGV", fogv_band)):
                cell = [t for t in tokens if lo <= t["xc"] <= hi and numeric_value(t["text"]) is not None]
                if len(cell) != 1:
                    raise RuntimeError(f"Table 3 y={y} {metric} {config}: cells {cell}")
                tok = cell[0]
                rows.append({
                    "table": "Table 3",
                    "condition": condition,
                    "speed_percent": f"{speed:g}",
                    "configuration": config,
                    "metric": metric,
                    "value": f"{numeric_value(tok['text']):g}",
                    "value_text": tok["text"],
                    "unit": unit,
                    "source_pdf": PDF.name,
                    "source_page_pdf": PDF_PAGE,
                    "source_page_report": REPORT_PAGE,
                    "source_y_pt": f"{tok['y0']:.2f}",
                    "source_x_pt": f"{tok['x0']:.2f}",
                })
    checks = [{
        "check": "Table 3 completeness",
        "table": "Table 3",
        "detail": f"{len(rows)}/24 performance cells parsed (3 conditions x 2 configurations x 4 metrics)",
        "pass": len(rows) == 24,
    }]
    return rows, checks


def main() -> dict:
    doc = fitz.open(PDF)
    page = doc[PDF_PAGE - 1]
    tokens_by_y = {y: words_on_line(page, y) for y in
                   list(TABLE_1_Y) + [r[0] for r in TABLE2_ROWS] + list(TABLE_3_Y)}

    g1, c1 = table1(tokens_by_y)
    g2, c2 = table2({y: tokens_by_y[y] for y, *_ in TABLE2_ROWS})
    g3, c3 = table3({y: tokens_by_y[y] for y in TABLE_3_Y})
    checks = c1 + c2 + c3

    # semantic cross-checks against the report text
    by_key = {(r["component"], r["parameter"], r.get("location", "")): r for r in g2}
    counts = {r["component"]: r["value"] for r in g2 if r["parameter"] == "blade_or_vane_count"}
    chords = {r["component"]: float(r["value"]) for r in g2
              if r["parameter"] == "chord" and r["location"] == "pitchline"}
    sweep = {r["component"]: r["value_text"] for r in g2 if r["parameter"] == "aft_sweep"}
    checks += [
        {"check": "blade/vane counts", "table": "Table 2",
         "detail": f"{counts}", "pass": counts == {"Fan": "22", "Baseline OGV (54)": "54",
                                                   "Low Count OGV (26)": "26", "Low Noise OGV (26)": "26"}},
        {"check": "narrow vs wide chord", "table": "Table 2",
         "detail": f"baseline {chords.get('Baseline OGV (54)')} in < low-count {chords.get('Low Count OGV (26)')} in",
         "pass": chords.get("Baseline OGV (54)", 9e9) < chords.get("Low Count OGV (26)", -1)},
        {"check": "sweep angle", "table": "Table 2",
         "detail": f"Low Noise = {sweep.get('Low Noise OGV (26)')} deg; others 0",
         "pass": sweep.get("Low Noise OGV (26)") == "30" and sweep.get("Baseline OGV (54)") == "0"
                 and sweep.get("Low Count OGV (26)") == "0"},
    ]

    # Table 3 approach Fan/OGV values used by the Mach derivation
    perf = {(r["condition"], r["configuration"], r["metric"]): r for r in g3}
    checks.append({
        "check": "approach Fan/OGV performance available",
        "table": "Table 3",
        "detail": "flow 58.30 lbm/s, pressure ratio 1.159, temperature ratio 1.049, efficiency 0.8880",
        "pass": (perf[("Approach", "Fan/OGV", "corrected_weight_flow")]["value"] == "58.3"
                 and perf[("Approach", "Fan/OGV", "total_pressure_ratio")]["value"] == "1.159"
                 and perf[("Approach", "Fan/OGV", "total_temperature_ratio")]["value"] == "1.049"
                 and perf[("Approach", "Fan/OGV", "adiabatic_efficiency")]["value"] == "0.888"),
    })

    write_csv(GEO / "sdt_fan_design.csv", g1, list(g1[0].keys()))
    write_csv(GEO / "sdt_geometry.csv", g2, list(g2[0].keys()))
    write_csv(GEO / "sdt_performance.csv", g3, list(g3[0].keys()))
    write_csv(GEO / "sdt_geometry_check.csv", checks, list(checks[0].keys()))

    meta = {
        "pdf": PDF.name,
        "pdf_sha256": sha256(PDF),
        "pdf_page": PDF_PAGE,
        "report_page": REPORT_PAGE,
        "tables": ["Table 1", "Table 2", "Table 3"],
        "n_rows": {"fan_design": len(g1), "geometry": len(g2), "performance": len(g3)},
        "method": "text-layer word boxes, assigned to table cells by x/y coordinates",
        "notes": [
            "Table 2 blank Fan hub/tip solidity and stagger cells are emitted as status=blank.",
            "Table 2 Fan sweep and camber are reported as '---' and emitted as status=dash.",
            "Column centres, not reading order, determine cell ownership.",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(PROV / "sdt_geometry_tables.json", meta)

    print(f"Table 1: {len(g1)} rows")
    print(f"Table 2: {len(g2)} rows (16 span rows x 4 configurations)")
    print(f"Table 3: {len(g3)} rows")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return meta


if __name__ == "__main__":
    main()
