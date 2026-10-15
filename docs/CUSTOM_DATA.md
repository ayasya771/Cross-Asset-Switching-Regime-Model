# Using your own data

Two paths, same transforms. Missing values are never imputed: leave a cell
empty (or `NaN`) and the filter masks it, exactly as it does for the TED
spread after discontinuation or macro releases between print dates.

## Option A — browser upload (no install)

1. Open the demo and choose **My CSV** in the Data source panel.
2. Upload a CSV with a `date` column (YYYY-MM-DD) plus **any** of these:
   - yields/credit/vol levels: `y_3m, y_2y, y_5y, y_10y, y_30y, hy_oas,
     ig_oas, vix, ted`
   - raw market levels: `spx_close, ndx_close, usd_idx, wti` — daily returns
     and 21-day realized vol are derived automatically, with WTI guarded the
     same way as the Python pipeline (floored at |prev| = 5, clipped at ±50%)
   - precomputed market features: `usd_ret, oil_ret, spx_ret, spx_rv21, ndx_ret`
   - macro: `cpi_yoy, indpro_yoy, unrate, gdp_yoy`, or raw `cpi, indpro, gdp`
     levels (YoY is derived against ~365 days earlier)
3. Pick **Scaling**: *From my upload* recomputes mean/std from the first 85%
   of your rows (min. 50 observations per series, else the shipped scaler fills
   in); *Shipped scaler* keeps the original normalization.
4. Press **Run the filter**. Withholding toggles, regime count, smoothing and
   the date scrubber all work on uploads. The agreement panel is hidden for
   uploads (the shipped Python reference covers the shipped panel only).

Need a starting point? Press **Template** in the upload box, or copy
`data/custom_template.csv`. At least 60 dated rows are required; 250+ gives
stable BIC regime counts. Column names are case-insensitive and common
aliases work (`DGS10` → `y_10y`, `^GSPC`/`spx` → `spx_close`,
`DTWEXBGS`/`DXY` → `usd_idx`, `DCOILWTICO`/`oil` → `wti`, `CPIAUCSL` → `cpi`).

## Option B — Python CLI

```bash
python scripts/ingest_custom.py --template > my_data.csv   # skeleton
# ... fill in your series (same columns/aliases as above) ...
python scripts/ingest_custom.py my_data.csv

python scripts/extract_regimes.py   # filter with the shipped checkpoint
python scripts/export_web_demo.py   # optionally refresh the demo payload
# or retrain first: python scripts/train.py --epochs 15
```

The ingester keeps the full 18-feature schema (absent series stay masked), so
the shipped checkpoint filters custom panels without retraining. Provenance
lands in `data/processed/custom_source.json`; `meta.json` records
`"source": "custom:<filename>"` plus per-feature coverage.
