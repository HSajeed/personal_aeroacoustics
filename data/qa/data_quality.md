# Day-1 data-quality table

Generated: 2026-11-05 10:06 UTC.  
Gate: **PASS** - 54-vane and 26-vane exhaust spectra digitised, with geometry, provenance and uncertainty.

| Dataset | Source | Method | Points | Range | Uncertainty | Key check | Status |
|---|---|---|---:|---|---|---|---|
| tm2002_p14_fig09_54vane_exhaust | 20030003692.pdf p. 14 (report p. 10) | raster dash morphology, 400 dpi, 3 px bins | 201 | 1081-49659 | 0.12 dB (k=1); frequency 0.25 % | BPF-band max 99.72 dB; 2BPF max 106.39 dB | PASS |
| tm2002_p14_fig10_26vane_exhaust | 20030003692.pdf p. 14 (report p. 10) | raster dash morphology, 400 dpi, 3 px bins | 457 | 1081-45369 | 0.12 dB (k=1); frequency 0.25 % | BPF-band max 118.87 dB; 2BPF max 111.95 dB | PASS |
| tm2002_p14_fig11_26swept_exhaust | 20030003692.pdf p. 14 (report p. 10) | raster dash morphology, 400 dpi, 3 px bins | 410 | 1466-48239 | 0.12 dB (k=1); frequency 0.25 % | BPF-band max 113.11 dB; 2BPF max 102.94 dB | PASS |
| fernandes_fig18c_54vane_exhaust_pwl | Fernandes.pdf p. 24, Fig. 18(c) | vector path extraction (PyMuPDF), thick black experimental curve | 433 | 865-28660 | 0.03 dB (k=1); frequency 0.1 % | 2BPF peak 84.73 dB/Hz; cross-check r=0.9785, RMS=0.993 dB | PASS |
| sdt_fan_design | 20050175875.pdf p. 12 (report p. 8), Table 1 | text-layer word boxes | 9 |  | exact text; no digitisation uncertainty | 9/9 rows parsed | PASS |
| sdt_geometry | 20050175875.pdf p. 12 (report p. 8), Table 2 | text-layer word boxes, column-centre assignment | 64 |  | exact text; 4 blank and 4 '---' cells explicit | 56 numeric, 4 dash, 4 blank; counts/chord/sweep checks pass | PASS |
| sdt_performance | 20050175875.pdf p. 12 (report p. 8), Table 3 | text-layer word boxes | 24 |  | exact text | 24/24 cells parsed; approach Fan/OGV used for Mach | PASS |
| fig15_annulus_mach | 20050175875.pdf p. 22 (report p. 18), Fig. 15; Table 3 | vector contour extraction + continuity solve | 474 | area 2.094-2.120 ft2; M 0.303 | M +/- 0.0044 (k=1) | area 2.094-2.120 ft2; M=0.3029; static-density solve > total-density shortcut | PASS |
| raster_figure26 | 20030065891.pdf p. 33 (report p. 29) | screening only | 0 |  | n/a - no values claimed | curves merge with dotted log-grid and symbols at ~500 px source resolution; no isolated curve component | SCREENED |
| raster_figure27 | 20030065891.pdf p. 33 (report p. 29) | screening only | 0 |  | n/a - no values claimed | curves merge with dotted log-grid and symbols at ~500 px source resolution; no isolated curve component | SCREENED |
| raster_figure28 | 20030065891.pdf p. 34 (report p. 30) | screening only | 0 |  | n/a - no values claimed | curves merge with dotted log-grid and symbols at ~500 px source resolution; no isolated curve component | SCREENED |

## Cross-check

- 54-vane NASA Fig. 9 vs Fernandes Fig. 18(c): Pearson r = 0.9785, RMS = 0.993 dB after removing each 2-8 kHz median; raw median offset = +21.368 dB (dB vs dB/Hz, not a physical error).
- 2 BPF prominence: NASA 4.07 dB vs Fernandes 5.00 dB.
- BPF remains cut off in the 54-vane NASA spectrum (1 BPF prominence 1.45 dB), as the report states.

## Not digitised

Figs. 26-28 (wake PSDs, von Karman comparison, integral length scales) are 1-bit scans about 500 px wide. Curves, symbols and dotted log-grids merge into single connected components; no numeric values are claimed. See `data/qa/raster_figure_screening.csv`.

