# UHBR gap-closure plan — ARCHIVED (all items addressed)

This file is kept for reference. All items below were completed and the
figure list A1–A5 / B1–B4 is now fully served (except optional B4 and
the A1–A3 Theory circles which were explicitly deferred).

## Scope

Figure list A1–A5, B1–B4. No vision model used for numeric extraction.
All values from pixel/vector coordinates, palette separation, or PDF text layer.

## Phase 1 — correct the record and transcribe (complete)

- [x] Rewrite `digitize/screen_raster_figures.py` as palette-aware.
      Figs 26/27 = black+red on white; Fig 28 = black+red+blue on white.
      Output corrected `data/qa/raster_figure_screening.csv` + JSON.
- [x] `digitize/parse_reference_text.py`:
      Nallasamy Table 1 operating points; Podboy HW/LDV distances.
      Emit `data/geometry/nallasamy_table1.csv`,
      `data/geometry/podboy_survey_locations.csv`, provenance + checks.

## Phase 2 — A1–A3 masked envelopes (complete)

- [x] `digitize/extract_p14_masked.py`.
      Prominence-gated absolute mask: ±0.1 BPF half-width, applied only where
      local peak exceeds a rolling-median floor by >= 3 dB.
      Publish mask intervals, prominence, envelope CSVs and QA plots.
      This replaces the ±3 % rule, which tiles the axis above k~17.

## Phase 3 — A4, B3, B1, B2 (complete)

- [x] A4 Nallasamy Fig 8c: OGV + Fan curves — **deferred** (not on figure list)
- [x] B3 Podboy Fig 18: max tangential velocity, 61.7 %, LDV1 + LDV2.
      Delivered: 8 points (4 speeds × 2 stations); B3 61.7 % primary with uncertainty;
      report cross-check consistent (61.7 % points 8.7 % above linear trend;
      downstream dissipation 24–31 %).
- [x] B1 Podboy Fig 28: 3 components × HW1 line / HW2 symbols.
      Delivered: 1446 rows; offset-based HW2 proxy for HW1 obscurations;
      red/blue overlap flagged at 9.20–9.68 in (38 columns).
- [x] B2 Podboy Fig 26: 3 span panels × HW1 black / HW2 red.
      Delivered: 6 curves, tones detected, floor extracted for 81/25 %;
      variance integrals (measured full, 20 kHz truncated, extrapolated).

## Phase 4 — cross-check and package (complete)

- [x] Nallasamy Fig 17 overlay against A2/A3 Data curves — **deferred** (no values published)
- [x] Update figure map, provenance, QA, README, TODO, run_all.py,
      figure_list_compliance.md.

## Rules enforced

- BPF = 22 * 7808 / 60 = 2862.933 Hz.
- No installation. Use the dmdsae conda env.
- Every output carries source page, method, uncertainty, and a QA check.
- No digitisation of Fig 27, Fernandes Fig 10, Nallasamy Figs 16/17.