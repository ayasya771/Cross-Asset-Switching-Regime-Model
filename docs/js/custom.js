/* Custom-data support for the browser demo: parse a user-supplied CSV, map
   flexible column names onto the model's 18-feature schema, derive returns /
   realized-vol / YoY transforms the same way regimelab/data/preprocess.py
   does, and standardize the panel for the frozen filter.

   Dependency-free and importable from both the page (app.js) and Node
   (scripts/verify_web_demo.mjs), so the browser path stays tested. */

export const FEATURES = [
  "y_3m", "y_2y", "y_5y", "y_10y", "y_30y",
  "hy_oas", "ig_oas", "vix", "ted",
  "usd_ret", "oil_ret", "spx_ret", "spx_rv21", "ndx_ret",
  "cpi_yoy", "indpro_yoy", "unrate", "gdp_yoy",
];
export const LOG_FEATURES = new Set(["vix", "spx_rv21"]);
export const YIELD_FEATURES = ["y_3m", "y_2y", "y_5y", "y_10y", "y_30y"];
export const DISPLAY_COLS = ["spx_close", "ndx_close", "usd_idx", "wti"];

/* Canonical-name aliases (all matched case-insensitively after trimming).
   Covers FRED IDs, common vendor labels and the panel.csv header itself. */
const ALIASES = {
  date: ["date", "ds", "time", "timestamp", "datetime", "day", "trading_date", "obs_date"],
  y_3m: ["y_3m", "dgs3mo", "3m", "3mo", "us3m", "y3m"],
  y_2y: ["y_2y", "dgs2", "2y", "2yr", "us2y", "y2y"],
  y_5y: ["y_5y", "dgs5", "5y", "5yr", "us5y", "y5y"],
  y_10y: ["y_10y", "dgs10", "10y", "10yr", "us10y", "y10y"],
  y_30y: ["y_30y", "dgs30", "30y", "30yr", "us30y", "y30y"],
  hy_oas: ["hy_oas", "hy", "bamlh0a0hym2", "high_yield", "hy_spread", "hyoas"],
  ig_oas: ["ig_oas", "ig", "bamlc0a0cm", "ig_spread", "igoas", "investment_grade"],
  vix: ["vix", "vixcls", "volatility"],
  ted: ["ted", "tedrate", "ted_spread", "tedspread"],
  usd_ret: ["usd_ret", "dollar_ret", "dxy_ret"],
  oil_ret: ["oil_ret", "wti_ret", "crude_ret"],
  spx_ret: ["spx_ret", "sp500_ret", "equity_ret"],
  spx_rv21: ["spx_rv21", "rv21", "rv", "realized_vol", "realised_vol", "spx_rv"],
  ndx_ret: ["ndx_ret", "nasdaq_ret", "ndxret"],
  cpi_yoy: ["cpi_yoy", "cpi", "cpiaucsl", "inflation"],
  indpro_yoy: ["indpro_yoy", "indpro", "industrial_production"],
  unrate: ["unrate", "unemployment", "unemp"],
  gdp_yoy: ["gdp_yoy", "gdp", "gdpc1"],
  spx_close: ["spx_close", "spx", "^spx", "^gspc", "sp500", "sp500_close", "close_spx", "spxclose", "equity_close"],
  ndx_close: ["ndx_close", "ndx", "^ndx", "nasdaq", "nasdaq_close", "ndxclose"],
  usd_idx: ["usd_idx", "usd", "dxy", "dtwexbgs", "dollar_idx", "broad_usd"],
  wti: ["wti", "oil", "crude", "dcoilwtico", "wti_close", "oil_close"],
  cpi_level: ["cpi_level", "cpi_idx"],
  indpro_level: ["indpro_level", "indpro_idx"],
  gdp_level: ["gdp_level", "gdp_idx"],
};
const CANON = {};
for (const [canon, names] of Object.entries(ALIASES))
  for (const n of names) CANON[n.toLowerCase()] = canon;

export function normalizeHeader(name) {
  const k = String(name || "").trim().toLowerCase();
  if (CANON[k]) return CANON[k];
  // Tolerate suffixes like " (pct)" or prefixes like "fred:".
  const cleaned = k.replace(/^(fred|stooq|yahoo)\s*:\s*/, "").replace(/[^a-z0-9_]/g, "");
  if (CANON[cleaned]) return CANON[cleaned];
  return null;
}

/* Small RFC-4180 reader: handles quoted fields, embedded commas, CRLF and
   surrounding whitespace. Far more forgiving than String.split(",") for
   spreadsheets saved from Excel/Sheets. */
export function parseCsvRobust(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  const src = String(text).replace(/^\uFEFF/, "");
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (quoted) {
      if (c === '"') {
        if (src[i + 1] === '"') { field += '"'; i++; }
        else quoted = false;
      } else field += c;
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(field); field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && src[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.length > 1 || row[0] !== "") rows.push(row);
  if (!rows.length) return { head: [], rows: [] };
  const head = rows[0].map((h) => String(h).trim());
  const body = rows.slice(1).filter((r) => r.some((c) => String(c).trim() !== ""));
  return { head, rows: body };
}

const MISSING_TOKENS = new Set(["", "nan", "na", "n/a", "nat", "null", "none", "nil", ".", "-", "--", "---", "#n/a", "#na", "?"]);
export function isMissingCell(cell) {
  if (cell === null || cell === undefined) return true;
  const t = String(cell).trim().toLowerCase();
  return MISSING_TOKENS.has(t);
}

export function parseNum(cell) {
  if (isMissingCell(cell)) return NaN;
  const t = String(cell).trim().replace(/,/g, "");
  if (t === "") return NaN;
  const v = parseFloat(t);
  return Number.isFinite(v) ? v : NaN;
}

/* Accept YYYY-MM-DD plus the formats spreadsheets commonly produce. Returns
   an ISO YYYY-MM-DD string or null. */
export function parseDateFlexible(s) {
  if (s === null || s === undefined) return null;
  let t = String(s).trim();
  if (!t) return null;
  // ISO datetime prefix.
  const iso = t.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (iso) {
    const y = +iso[1], m = +iso[2], d = +iso[3];
    if (m >= 1 && m <= 12 && d >= 1 && d <= 31)
      return `${iso[1]}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
    void y; return null;
  }
  let m = t.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2,4})/);
  if (m) {
    let yy = +m[3]; if (yy < 100) yy += yy < 50 ? 2000 : 1900;
    return `${String(yy).padStart(4, "0")}-${String(+m[1]).padStart(2, "0")}-${String(+m[2]).padStart(2, "0")}`;
  }
  m = t.match(/^(\d{1,2})-(\d{1,2})-(\d{4})/);
  if (m) return `${m[3]}-${String(+m[1]).padStart(2, "0")}-${String(+m[2]).padStart(2, "0")}`;
  m = t.match(/^(\d{4})\/(\d{1,2})\/(\d{1,2})/);
  if (m) return `${m[1]}-${String(+m[2]).padStart(2, "0")}-${String(+m[3]).padStart(2, "0")}`;
  const ms = Date.parse(t);
  if (Number.isFinite(ms)) {
    const d = new Date(ms);
    return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
  }
  return null;
}

function logdiffFromLevels(levels) {
  const out = new Array(levels.length).fill(NaN);
  for (let t = 1; t < levels.length; t++) {
    const a = levels[t - 1], b = levels[t];
    if (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0)
      out[t] = Math.log(b) - Math.log(Math.max(a, 1e-12));
  }
  return out;
}

function oilRetFromLevels(wti, clip = 0.5) {
  const out = new Array(wti.length).fill(NaN);
  for (let t = 1; t < wti.length; t++) {
    const prev = wti[t - 1], cur = wti[t];
    if (!Number.isFinite(prev) || !Number.isFinite(cur)) continue;
    const r = (cur - prev) / Math.max(Math.abs(prev), 5.0);
    out[t] = Math.max(-clip, Math.min(clip, r));
  }
  return out;
}

function rv21FromRets(rets, window = 21) {
  const out = new Array(rets.length).fill(NaN);
  const r2 = rets.map((v) => (Number.isFinite(v) ? v * v : NaN));
  for (let t = window; t < rets.length; t++) {
    let s = 0, n = 0;
    for (let k = t - window + 1; k <= t; k++)
      if (Number.isFinite(r2[k])) { s += r2[k]; n++; }
    if (n >= window - 2) out[t] = 100 * Math.sqrt((252 * s) / Math.max(n, 1));
  }
  return out;
}

/* Year-on-year % change of a sparse level series against the observation
   closest to 365 days earlier (within a 60-day tolerance), mirroring the
   _yoy + _attach_sparse combination in preprocess.py for user-supplied
   monthly/quarterly levels. Falls back to a 12-observation lag when dates
   are unparseable (should not happen after date normalization). */
function yoyFromLevels(datesMs, levels) {
  const out = new Array(levels.length).fill(NaN);
  const DAY = 86400000;
  for (let t = 0; t < levels.length; t++) {
    if (!Number.isFinite(levels[t])) continue;
    let best = -1, bestErr = Infinity;
    for (let k = t - 1; k >= 0; k--) {
      if (!Number.isFinite(levels[k])) continue;
      const gap = datesMs[t] - datesMs[k];
      const err = Math.abs(gap - 365 * DAY);
      if (err < bestErr) { bestErr = err; best = k; }
      if (gap > 425 * DAY) break;
    }
    if (best >= 0 && bestErr <= 60 * DAY && levels[best] !== 0)
      out[t] = 100 * (levels[t] / levels[best] - 1);
  }
  return out;
}

/* Build the model-ready panel from a robust-parsed CSV ({head, rows}).
   shipped = {features, log_features, mean, std} provides the fallback
   normalization. Returns everything app.js needs to run the filter. */
export function buildCustomPanel(parsed, shipped, opts = {}) {
  const warnings = [];
  const oilClip = opts.oilClip ?? 0.5;
  const rvWindow = opts.rvWindow ?? 21;
  const splitFrac = opts.splitFrac ?? 0.85;
  const scalerMode = opts.scalerMode ?? "recompute";
  const n = FEATURES.length;

  const colCanon = parsed.head.map(normalizeHeader);
  const dateIx = colCanon.indexOf("date");
  if (dateIx < 0)
    return { ok: false, error: "No date column found. Name one column 'date' (YYYY-MM-DD)." };

  // Column index per canonical name (first occurrence wins).
  const ixOf = {};
  colCanon.forEach((c, i) => { if (c && !(c in ixOf)) ixOf[c] = i; });

  // Gather dated records.
  const recs = [];
  for (const r of parsed.rows) {
    const iso = parseDateFlexible(r[dateIx]);
    if (!iso) continue;
    recs.push({ iso, row: r });
  }
  if (!recs.length)
    return { ok: false, error: "No parseable dates. Use YYYY-MM-DD (one row per observation)." };
  recs.sort((a, b) => (a.iso < b.iso ? -1 : a.iso > b.iso ? 1 : 0));
  // Deduplicate: keep the last row for a repeated date.
  const dedup = [];
  for (const rec of recs) {
    if (dedup.length && dedup[dedup.length - 1].iso === rec.iso) dedup[dedup.length - 1] = rec;
    else dedup.push(rec);
  }
  const droppedDates = recs.length - dedup.length;
  const T = dedup.length;
  if (T < 60)
    return { ok: false, error: `Only ${T} dated rows found; at least 60 daily observations are needed (250+ recommended for stable regimes).` };
  if (T < 250)
    warnings.push(`Small sample (${T.toLocaleString()} rows). Regime counts chosen by BIC are less stable below ~250 observations.`);

  const col = (canon) => {
    const ix = ixOf[canon];
    if (ix === undefined) return new Array(T).fill(NaN);
    return dedup.map(({ row }) => parseNum(row[ix]));
  };

  const dates = dedup.map((d) => d.iso);
  const datesMs = dates.map((d) => Date.parse(d + "T00:00:00Z"));

  // Direct feature columns as uploaded.
  const featRaw = {};
  for (const f of FEATURES) featRaw[f] = col(f);
  // Raw closes / levels for derivation.
  const spxClose = ixOf.spx_close !== undefined ? col("spx_close") : new Array(T).fill(NaN);
  const ndxClose = ixOf.ndx_close !== undefined ? col("ndx_close") : new Array(T).fill(NaN);
  const usdIdx = ixOf.usd_idx !== undefined ? col("usd_idx") : new Array(T).fill(NaN);
  const wtiLvl = ixOf.wti !== undefined ? col("wti") : new Array(T).fill(NaN);

  const useDerived = (feat, derived, label) => {
    let missing = 0;
    for (let t = 0; t < T; t++) if (!Number.isFinite(featRaw[feat][t])) missing++;
    if (missing === 0) return;
    let filled = 0;
    for (let t = 0; t < T; t++)
      if (!Number.isFinite(featRaw[feat][t]) && Number.isFinite(derived[t])) {
        featRaw[feat][t] = derived[t]; filled++;
      }
    if (filled > 0) warnings.push(`Derived ${feat} from ${label} for ${filled.toLocaleString()} rows.`);
  };
  useDerived("spx_ret", logdiffFromLevels(spxClose), "equity closes");
  useDerived("ndx_ret", logdiffFromLevels(ndxClose), "Nasdaq closes");
  useDerived("usd_ret", logdiffFromLevels(usdIdx), "USD index levels");
  useDerived("oil_ret", oilRetFromLevels(wtiLvl, oilClip), "WTI levels");
  useDerived("spx_rv21", rv21FromRets(featRaw.spx_ret, rvWindow), "equity returns (21-day window)");

  // Macro levels -> YoY where the YoY column itself was not supplied.
  const macroLevels = { cpi_yoy: "cpi_level", indpro_yoy: "indpro_level", gdp_yoy: "gdp_level" };
  for (const [yoy, lvl] of Object.entries(macroLevels)) {
    if (ixOf[lvl] === undefined) continue;
    const derived = yoyFromLevels(datesMs, col(lvl));
    useDerived(yoy, derived, "uploaded levels");
  }
  // Convenience: a bare "cpi"/"gdp"/"indpro" column header normalizes to the
  // YoY feature directly (levels would need a time-aware YoY step, so if the
  // values look like index levels rather than percents we still accept them
  // as YoY input and note the assumption).
  for (const f of ["cpi_yoy", "indpro_yoy", "gdp_yoy"]) {
    let obs = 0;
    for (let t = 0; t < T; t++) if (Number.isFinite(featRaw[f][t])) obs++;
    if (obs === 0) warnings.push(`${f} not observed; it will be masked throughout.`);
  }

  // Coverage before standardization.
  const coverage = {};
  for (const f of FEATURES) {
    let c = 0;
    for (let t = 0; t < T; t++) if (Number.isFinite(featRaw[f][t])) c++;
    coverage[f] = c / T;
  }
  const nObserved = FEATURES.filter((f) => coverage[f] > 0).length;
  if (nObserved < 3)
    return { ok: false, error: `Only ${nObserved} of the 18 model series were recognized. See the template for accepted column names.` };

  // Standardization: recompute from the upload's first splitFrac rows where
  // possible, else fall back to the shipped scaler so dims stay comparable.
  const splitIdx = Math.max(1, Math.floor(T * splitFrac));
  const usedMean = new Array(n), usedStd = new Array(n);
  const fromUpload = new Array(n).fill(true);
  FEATURES.forEach((f, j) => {
    let vals = [];
    // Apply the log transform before measuring moments, mirroring preprocess.
    for (let t = 0; t < splitIdx; t++) {
      let v = featRaw[f][t];
      if (!Number.isFinite(v)) continue;
      if (LOG_FEATURES.has(f)) v = Math.log(Math.max(v, 1e-4));
      vals.push(v);
    }
    if (vals.length >= 50) {
      const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
      const va = vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length;
      usedMean[j] = mu;
      usedStd[j] = Math.sqrt(va) || 1;
    } else {
      usedMean[j] = shipped.mean[j];
      usedStd[j] = shipped.std[j] || 1;
      fromUpload[j] = false;
      if (vals.length > 0)
        warnings.push(`${f}: only ${vals.length} training-region observations; using shipped normalization.`);
    }
  });
  if (scalerMode === "shipped") {
    for (let j = 0; j < n; j++) { usedMean[j] = shipped.mean[j]; usedStd[j] = shipped.std[j] || 1; fromUpload[j] = false; }
    warnings.push("Using shipped normalization for all series (upload statistics ignored).");
  }

  const X = new Float64Array(T * n), M = new Float64Array(T * n);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < n; j++) {
      let v = featRaw[FEATURES[j]][t];
      if (!Number.isFinite(v)) continue;
      if (LOG_FEATURES.has(FEATURES[j])) v = Math.log(Math.max(v, 1e-4));
      M[t * n + j] = 1;
      X[t * n + j] = (v - usedMean[j]) / usedStd[j];
    }

  const DAY = 86400000;
  const dt = new Float64Array(T);
  for (let t = 1; t < T; t++)
    dt[t] = (datesMs[t] - datesMs[t - 1]) / DAY / 365.25;

  // Display levels for the charts (NaN where the upload lacks them).
  const raw = {};
  const put = (name, arr) => { raw[name] = Float64Array.from(arr.map((v) => (Number.isFinite(v) ? v : NaN))); };
  put("spx_close", spxClose);
  put("ndx_close", ndxClose);
  put("usd_idx", usdIdx);
  put("wti", wtiLvl);
  for (const f of ["vix", "hy_oas", "y_10y", "y_3m", "spx_ret"]) put(f, featRaw[f]);
  for (const f of YIELD_FEATURES) if (!raw[f]) put(f, featRaw[f]);

  return {
    ok: true, T, n, dates, X, M, dt, raw, coverage,
    usedMean, usedStd, fromUpload, warnings, droppedDates,
    start: dates[0], end: dates[T - 1],
  };
}

export function templateCsv() {
  const header = ["date", ...FEATURES, ...DISPLAY_COLS].join(",");
  const rows = [
    "2024-01-02,0.15,0.45,0.60,0.85,1.10,4.20,1.35,16.5,0.28,0.0008,0.0042,0.0006,15.2,0.0009,3.1,1.8,3.7,2.4,4783.83,16833.38,101.2,73.5",
    "2024-01-03,0.16,0.47,0.62,0.87,1.12,4.25,1.37,17.1,0.29,-0.0004,-0.0061,-0.0012,15.4,-0.0018,,,,,4778.12,16806.45,101.1,72.9",
  ];
  return header + "\n" + rows.join("\n") + "\n";
}
