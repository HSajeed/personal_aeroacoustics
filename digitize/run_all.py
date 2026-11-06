#!/usr/bin/env python3
"""Full Day-1 digitisation pipeline, run_all.py

Runs every extraction/derivation script in the correct order, producing
all CSVs, plots, provenance and the data-quality table.  Designed so that
a single command regenerates the entire auditable data package from a
clean state.

Order matters: inventory first, then calibration → p14 → Fernandes →
geometry → annulus → raster screening → QA.  Each script writes its own
outputs; the pipeline does not assume intermediate files exist.
"""
from __future__ import annotations

import subprocess
import sys

from common import ROOT

SCRIPTS = [
    # 1. Inventory and checksums (needed by downstream scripts that reference PDFs)
    "digitize/build_inventory.py",

    # 2. Panel/tick calibration for NASA p14
    "digitize/calibrate_p14.py",

    # 3. Measured exhaust spectra extraction from p14
    "digitize/extract_p14_spectra.py",

    # 4. Fernandes Fig. 18(c) vector extraction
    "digitize/extract_fernandes.py",

    # 5. Geometry tables from TM-2005-211681 (Tables 1–3)
    "digitize/parse_geometry.py",

    # 6. Annulus area + mean axial Mach from Fig. 15
    "digitize/derive_annulus.py",

    # 7. Reference text transcription (Nallasamy Table 1, Podboy station distances)
    "digitize/parse_reference_text.py",

    # 8. A1–A3 masked envelopes (prominence-gated tone masking)
    "digitize/extract_p14_masked.py",

    # 9. B1: Podboy Fig 28 — integral length scales
    "digitize/extract_podboy_fig28.py",

    # 10. B2: Podboy Fig 26 — wake PSDs + variance integrals
    "digitize/extract_podboy_fig26.py",

    # 11. B3: Podboy Fig 18 — max tangential turbulent velocity
    "digitize/extract_podboy_fig18.py",

    # 12. Raster figure screening (Figs. 26–28; done after digitisation for evidence)
    "digitize/screen_raster_figures.py",

    # 13. QA plots, cross-checks and data-quality table
    "digitize/make_qa.py",
]


def main() -> None:
    any_fail = False
    for script in SCRIPTS:
        path = ROOT / script
        if not path.exists():
            print(f"  SKIP {script}: not found")
            continue
        print(f"\n{'='*70}")
        print(f"RUNNING {script}  ...{'='*70}")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(line)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                print("ERR: " + line, file=sys.stderr)
        if result.returncode != 0:
            print(f"\n❌ {script} FAILED (exit {result.returncode})")
            any_fail = True
        else:
            print(f"\n✅ {script} completed successfully")
    print(f"\n{'='*70}")
    if any_fail:
        print("Pipeline finished with errors. Check output above.")
        sys.exit(1)
    else:
        print("All scripts completed successfully. Pipeline finished.")


if __name__ == "__main__":
    main()