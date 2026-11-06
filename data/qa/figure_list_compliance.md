# Figure-list compliance

## Match by figure-list item

| # | Required | What we did | Status |
|---|----------|-------------|--------|
| **A1** | Nallasamy Fig 9: 54-vane exhaust, Data curve + Theory circles | Digitised the dashed Data curve only (TL panel, 253 valid bins); Theory not extracted | **Partial** |
| **A2** | Nallasamy Fig 10: 26 radial, Data + Theory | Digitised dashed Data curve only (BL, 272 bins) | **Partial** |
| **A3** | Nallasamy Fig 11: 26 swept, Data + Theory | Digitised dashed Data curve only (TR, 230 bins) | **Partial** |
| **A4** | Nallasamy Fig 8c: both curves, OGV and Fan (Fan = subtraction floor) | Not touched; not in figure map | **Not done** |
| **A5** | Fernandes Fig 18c: Experimental Data, 7808 RPM, x BPF 0.3–10, y PWL dB/Hz | Vector-extracted, 433 points, 865–28 660 Hz | **Done** |
| **B1** | Podboy Fig 28: 6 curves, streamwise/upwash/spanwise × HW1 line/HW2 asterisks | Digitised: 1446 rows (HW1 line continuous via short-gap interp + HW2 offset proxy; HW2 asterisk markers); red/blue overlap flagged at 9.20–9.68 in | **Done** |
| **B2** | Podboy Fig 26: 6 upwash PSD curves (97/81/25 % × HW1/HW2) | Digitised: 6 curves, tones detected, floor extracted for 81/25 %; variance integrals (measured, 20 kHz truncated, extrapolated) | **Done** |
| **B3** | Podboy Fig 18: max tangential velocity at 61.7 %, LDV1 + LDV2 (2 points) | Digitised: 4 speeds × 2 stations (8 points); B3 deliverable 61.7 % published with uncertainty; report cross-check consistent | **Done** |
| **B4** | Podboy Fig 16, 61.7 % panel: radial profile via colourbar | Not touched (optional, 2 h max) | **Not done** |

## Match by processing rule

| Rule | Status | Note |
|------|--------|------|
| BPF = 22 × 7808 / 60 = 2863 Hz; convert frequency to BPF | **Done** | `BPF = 2862.933 Hz`; all spectra CSVs carry `BPF_order`; plots mark BPF/2BPF |
| A1–A3: mask tones ±3 % around k·BPF, keep broadband envelope | **Not done** | Raw spectra still include tones; no masked-envelope CSVs |
| B2: 97 % span whole; 81 %/25 % only floor between tones | **Done** | 97 % whole curve; 81/25 % floor between tones used for integrals |
| Hot-wire variance integral cut at 20 kHz; truncated + extrapolated, both reported | **Done** | `podboy_fig26_variance_integrals.csv` has both variants |
| Hughes Tables 1–3 transcribed, no digitising | **Done** | Table 1 9/9, Table 2 64 cells (56 numeric + 4 `---` + 4 blank), Table 3 24/24 |
| Nallasamy Table 1 operating points | **Done** | Transcribed to `data/geometry/nallasamy_table1.csv` (Approach 7808/750; Cutback 11074/1060; Takeoff 12656/1215); all QA checks pass |
| Podboy text distances: HW1/HW2 = 2.96/5.89 in; LDV1/LDV2 = 3.12/6.49 in | **Done** | Transcribed to `data/geometry/podboy_survey_locations.csv`; inch/cm consistency verified; HW upstream of LDV confirmed |
| Do not digitise Podboy Fig 27 levels | **Honoured** | Fig 27 is screening-only; no values claimed |
| Do not digitise Fernandes Fig 10 | **Honoured** | Not used; Fig 18c used instead |
| Do not digitise Nallasamy Fig 16 | **Honoured** | Not digitised |
| Do not digitise Nallasamy Fig 17; only overlay as cross-check | **Not done as check** | Not digitised (correct), but the overlay cross-check was not performed |

## What the screening CSV means now

`data/qa/raster_figure_screening.csv` is evidence for why Fig. 27 was skipped — it records that Fig. 27 is a ~500 px 1-bit scan whose curves merge with the dotted log-grid. Figs. 26 and 28 were also screened but were subsequently digitised using exact palette masks and deterministic curve tracing (no vision model), so the screening CSV documents the difficulty, not a permanent block.

## Primary deliverables produced

| Item | File | Rows | Notes |
|------|------|------|-------|
| A1–A3 Data curves | `tm2002_p14_fig09_54vane_exhaust_1_6oct.csv` etc. | 253 / 272 / 230 | Dashed curves only; 1/6-octave binning |
| A5 Fernandes | `fernandes_fig18c_54vane_exhaust_pwl.csv` | 433 | Vector path; 865–28 660 Hz |
| B1 Fig 28 | `podboy_fig28_integral_length_scales.csv` | 1446 | 3 comps × HW1/HW2; overlap flagged |
| B2 Fig 26 | `podboy_fig26_psd.csv` | 1572 | 6 curves; tone flags; floor for 81/25 % |
| B2 integrals | `podboy_fig26_variance_integrals.csv` | 6 | Measured / 20 kHz / extrapolated |
| B3 Fig 18 | `podboy_fig18_points.csv` | 8 | 4 speeds × 2 stations; B3 61.7 % primary |
| B3 curve | `podboy_fig18_curve.csv` | 556 | Full traced curves |