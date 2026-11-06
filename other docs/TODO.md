# Day 1 digitisation progress

Goal: digitise the SDT plots/data needed for Day 1 and leave a reproducible, auditable data set.

## Gate-critical (PASS)

- [x] Inventory PDFs, checksums, page/figure map
- [x] Confirm p14 panel/figure mapping (Figs 9/10 = 54/26 exhaust)
- [x] Geometry tables -> `data/geometry/sdt_geometry.csv`
- [x] Annulus area / mean axial Mach check
- [x] 54-vane measured exhaust, vector fallback (Fernandes Fig 18c)
- [x] 54-vane measured exhaust, raster (TM-2002 p14 TL)
- [x] 26-vane measured exhaust, raster (TM-2002 p14 TR)
- [x] Cross-check 54 raster vs Fernandes vector (Pearson r = 0.9785)
- [x] Gate decision + provenance recorded

## Rest of Day 1 (complete)

- [x] Wake PSDs, Fig 26 (digitised: 6 curves × 3 spans × 2 stations, tones flagged, floor extracted)
- [x] von Karman comparison, Fig 27 (screened, not digitised — documented)
- [x] Integral length scales, Fig 28 (digitised: 3 components × HW1 line / HW2 asterisks)
- [x] Max tangential velocity, Fig 18 (digitised: LDV1/LDV2 at 50/61.7/87.5/100 % speed; B3 deliverable 61.7 %)
- [x] Data-quality table (data/qa/data_quality.csv / data_quality.md)
- [x] One-command regeneration (digitize/run_all.py)

## Method constraints

- [x] No vision model reads the plots.
- [x] Vector paths extracted with PyMuPDF where available.
- [x] Raster curves traced by deterministic pixel processing.
- [x] Classical OCR only for axis/caption text, or a one-time calibration JSON.
- [x] Every number carries source page/panel, method and uncertainty.

## Exported data package

- `data/provenance/provenance.json` — consolidated provenance from all sources
- `data/provenance/provenance.md` — human-readable data-quality summary
- `data/README.md` — tree, data dictionary, regeneration command, caveats
- `data/qa/data_quality.md` — one-page data-quality table
- `data/qa/validation_checks_all.csv` — combined validation across all domains
- All CSVs in `data/digitized/`, `data/geometry/`, `data/derived/`, `data/inventory/`
- All plots in `data/qa/overlays/`

## Not digitised (documented, not invented)

- Fig. 27 (von Karman comparison) is a 1-bit raster scan; curves merge with the dotted log-grid; no numeric values are claimed. See `data/qa/raster_figure_screening.csv`.

## Key cross-check results

- 54-vane NASA Fig. 9 vs Fernandes Fig. 18(c): Pearson r = 0.9785, RMS = 0.993 dB (shape, median-removed, 2–8 kHz band)
- 2 BPF peak: NASA 105.16 dB at ~2.98 kHz vs Fernandes 84.73 dB/Hz at ~5.73 kHz (different units; shape trend matches)
- 1 BPF prominence NASA = 1.45 dB (cut off for 54 vanes, as the report states)
- Annulus area x=3–6 in: 2.094–2.120 ft², mean 2.1040 ft²
- Mean axial Mach = 0.3029 ± 0.0044 (k=1), V = 104.60 m/s
- Podboy Fig 18 B3: LDV1 61.7 % = 53.1 ± 0.5 ft/s, LDV2 61.7 % = 40.0 ± 0.5 ft/s; report cross-check: 61.7 % points 8.7 % above linear trend (report: "higher"), downstream dissipation 24–31 % (report: 24–31 %) — all consistent