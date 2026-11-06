"""Consolidate all per-source provenance JSON files into a single provenance.json
and data-quality markdown table for the Day-1 digitisation output.

Creates:
  data/provenance/provenance.json   -- merged metadata from all sources
  data/provenance/provenance.md     -- human-readable data-quality summary
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json

from common import ROOT, PROV, write_csv, write_json, sha256

# ---------------------------------------------------------------------------
# Locate all per-source provenance files
# ---------------------------------------------------------------------------
PROV_DIR = PROV
OUTPUT_JSON = PROV_DIR / "provenance.json"
OUTPUT_MD = PROV_DIR / "provenance.md"


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"  warning: {label} not found at {path}; skipping")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"  warning: cannot parse {label} {path}: {exc}; skipping")
        return {}


# ---------------------------------------------------------------------------
# Gather per-source metadata
# ---------------------------------------------------------------------------
sources: list[dict] = []

# PDF inventory
src_inv = _load(PROV_DIR / "source_inventory.json", "source_inventory")
if src_inv:
    sources.append({
        "source_type": "pdf_inventory",
        "data": src_inv,
        "checksum": sha256(ROOT / "20030003692.pdf")[:12]
        + "...",
    })

# Per-source JSON files
for label, pname in [
    ("annulus Mach derivation", "annulus_mach.json"),
    ("Fernandes Fig 18(c)", "fernandes_fig18c.json"),
    ("SDT geometry tables", "sdt_geometry_tables.json"),
]:
    d = _load(PROV_DIR / pname, label)
    if d:
        sources.append({"source_type": pname.replace(".json", ""), "data": d, "checksum": ""})

# QA summary (cross-checks, gate, outputs, limitations)
qa_sum = _load(PROV_DIR / "qa_summary.json", "qa_summary")
if qa_sum:
    sources.append({"source_type": "qa_summary", "data": qa_sum, "checksum": ""})

# ---------------------------------------------------------------------------
# Build consolidated metadata
# ---------------------------------------------------------------------------
def _safe_get(meta: dict, *keys, default: str = "") -> str:
    """Walk nested dict keys; return '' if any key is missing."""
    cur = meta
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return str(cur)


meta: dict = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "software_versions": {
        "pymupdf": ">=1.23.6",
        "numpy": ">=1.24.0",
        "scipy": ">=1.10.0",
        "matplotlib": ">=3.8.0",
        "tesseract": "/tmp/opencode/ocr/bin/tesseract",
    },
    "sources": {},
}

# Merge each source, tracking checksums where available
overall_checksums: list[str] = []
for src in sources:
    stype = src["source_type"]
    sdata = src["data"]
    scheck = src.get("checksum", "")
    meta["sources"][stype] = sdata
    if scheck:
        overall_checksums.append(scheck)

# Consolidated checksums of all project PDFs
pdf_files = sorted(ROOT.glob("*.pdf"))
pdf_checksums: list[str] = []
for pf in pdf_files:
    h = sha256(pf)
    meta["sources"][f"{pf.name}_checksum"] = h
    overall_checksums.append(h)
    pdf_checksums.append(f"{h}  {pf.name}")

meta["project_checksums"] = {
    "pdf_sha256": pdf_checksums,
    "generated_utc": meta["generated_utc"],
}

# ---------------------------------------------------------------------------
# Write provenance.json
# ---------------------------------------------------------------------------
write_json(OUTPUT_JSON, meta)
print(f"Wrote {OUTPUT_JSON}  ({len(meta['sources'])} source entries)")


# ---------------------------------------------------------------------------
# Write provenance.md  (human-readable data-quality summary)
# ---------------------------------------------------------------------------
md_lines: list[str] = []


def md(text: str) -> None:
    md_lines.append(text)


md("# Day-1 SDT Digitisation — Provenance and Data-Quality Summary\n")
md(f"Generated: {meta['generated_utc']}")
qa = meta["sources"].get("qa_summary", {})
md(f"Gate status: **{qa.get('gate_status', 'UNKNOWN')}**")
md("")

# Cross-check metrics
cross = qa.get("crosscheck_54vane", {})
md("## 54-vane vs Fernandes cross-check\n")
md(f"- Pearson r (shape, median-removed): **{cross.get('pearson_r_normalised', '?')}**")
md(f"- RMS normalised difference: **{cross.get('rms_normalised_dB', '?')} dB**")
md(f"- Mean normalised difference: **{cross.get('mean_normalised_dB', '?')} dB**")
md(f"- 2 BPF peak NASA: **{cross.get('2BPF_peak_nasa_dB', '?')} dB**")
md(f"- 2 BPF peak Fernandes: **{cross.get('2BPF_peak_fernandes_dB_per_Hz', '?')} dB/Hz**")
md(f"- 1 BPF prominence NASA: **{cross.get('1BPF_prominence_nasa_dB', '?')} dB**)  (cut-off expected for 54 vanes)")
md(f"- Harmonic correlation (1–8 BPF): **{cross.get('harmonic_correlation', '?')}**")
md(f"- Harmonic RMS: **{cross.get('harmonic_rms_dB', '?')} dB**")
md("")

# Limitations
md("## Limitations\n")
for lim in qa.get("limitations", []):
    md(f"- {lim}")
md("")

# Per-source summary
md("## Source Summary\n")
for src in sources:
    stype = src["source_type"]
    sdata = src["data"]
    md(f"### {stype}\n")
    if stype == "pdf_inventory":
        n_pdfs = sdata.get("n_rows", 0)
        md(f"- {n_pdfs} PDFs inventoried\n")
    elif stype == "qa_summary":
        md(f"- Gate: {sdata.get('gate_status', '?')}\n")
        md(f"- Outputs: {', '.join(sdata.get('outputs', {}).keys())})\n")
    elif stype == "annulus_mach_json":
        res = sdata.get("results", {})
        md(f"- Annulus area x=3–6 in: {res.get('area_ft2_mean_x3_6', '?')} ft²\n")
        md(f"- Mean axial Mach: {res.get('mean_axial_mach', '?')} ± {res.get('mach_sigma_k1', '?')} (k=1)\n")
    elif stype == "fernandes_fig18c_json":
        md(f"- Vector points: {sdata.get('results', {}).get('n_points', '?')} points\n")
        md(f"- Frequency range: {sdata.get('results', {}).get('frequency_range_hz', ['?', '?'])}\n")
    elif stype == "sdt_geometry_tables_json":
        tables = sdata.get("tables", [])
        md(f"- Tables parsed: {', '.join(tables)}\n")

md_lines.append("\n")
OUTPUT_MD.write_text("\n".join(md_lines))
print(f"Wrote {OUTPUT_MD}")