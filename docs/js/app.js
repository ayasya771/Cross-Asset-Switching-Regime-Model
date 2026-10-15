/* Wiring for the demo: fetch the shipped dataset, reproduce the preprocessing,
   run the encoder and the extended Kalman filter in the browser, cluster the
   filtered trajectory, and render the panels.

   Training happens offline in Python. What runs here is the inference half of
   the pipeline, against the weights exported from the trained checkpoint. */

import { RegimeModel, ekfStep, gmm, modalSmooth, adjustedRand } from "./engine.js";
import {
  FEATURES as CUSTOM_FEATURES,
  parseCsvRobust,
  parseNum,
  parseDateFlexible,
  buildCustomPanel,
  templateCsv,
} from "./custom.js";
import * as charts from "./charts.js";

const DATA = "data/";
const $ = (id) => document.getElementById(id);
const S = {};

const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "n/a");
const pct = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) + "%" : "n/a");

function status(text, busy = false) {
  $("status-text").textContent = text;
  $("lamp").classList.toggle("live", busy);
}

function parseCsv(text) {
  // Robust reader (quotes, CRLF, whitespace) shared with the upload path.
  // Missing cells ("", "NaN", ".", "NA", "null", ...) are left for the caller:
  // parseFloat on them yields NaN, which Number.isFinite rejects, so the
  // observation stays masked — the same mechanism as genuinely missing data.
  const { head, rows } = parseCsvRobust(text);
  return { head, rows };
}

function fetchOk(url, as = "json") {
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(url + " → HTTP " + r.status);
    return as === "json" ? r.json() : r.text();
  });
}
async function boot() {
  status("Loading dataset", true);
  const [scaler, modelJson, panelTxt, refTxt, summary, refLatentTxt] = await Promise.all([
    fetchOk(DATA + "scaler.json", "json"),
    fetchOk(DATA + "model.json", "json"),
    fetchOk(DATA + "panel.csv", "text"),
    fetchOk(DATA + "reference.csv", "text"),
    fetchOk(DATA + "summary.json", "json"),
    fetchOk(DATA + "reference_latent.csv", "text"),
  ]);

  S.scaler = scaler;
  S.summary = summary;
  S.model = new RegimeModel(modelJson);
  S.features = scaler.features;

  const { head, rows } = parseCsv(panelTxt);
  const n = S.features.length, T = rows.length;
  const colIx = S.features.map((f) => head.indexOf(f));
  const dateIx = head.indexOf("date");
  const isLog = new Set(scaler.log_features);

  S.T = T;
  S.n = n;
  S.dates = new Array(T);
  S.X = new Float64Array(T * n);
  S.M = new Float64Array(T * n);
  S.dt = new Float64Array(T);
  S.raw = {};
  for (const c of ["spx_close", "vix", "hy_oas", "y_10y", "y_3m", "spx_ret", ...scaler.yield_features])
    if (head.indexOf(c) >= 0) S.raw[c] = new Float64Array(T).fill(NaN);

  for (let t = 0; t < T; t++) {
    const p = rows[t];
    S.dates[t] = p[dateIx];
    for (let j = 0; j < n; j++) {
      const cell = p[colIx[j]];
      if (cell === "" || cell === undefined) continue;
      let v = parseFloat(cell);
      if (!Number.isFinite(v)) continue;
      if (isLog.has(S.features[j])) v = Math.log(Math.max(v, 1e-4));
      S.M[t * n + j] = 1;
      S.X[t * n + j] = (v - scaler.mean[j]) / scaler.std[j];
    }
    for (const c of Object.keys(S.raw)) {
      const ix = head.indexOf(c);
      const v = parseFloat(p[ix]);
      if (Number.isFinite(v)) S.raw[c][t] = v;
    }
  }
  const DAY = 86400000;
  for (let t = 1; t < T; t++)
    S.dt[t] = (Date.parse(S.dates[t]) - Date.parse(S.dates[t - 1])) / DAY / 365.25;

  S.refLabels = parseCsv(refTxt).rows.map((r) => +r[1]);
  S.refLatent = refLatentTxt.trim().split("\n").map((l) => l.split(",").map(Number));

  $("meta-span").textContent = `${scaler.start} to ${scaler.end}`;
  $("meta-rows").textContent = T.toLocaleString();
  $("meta-features").textContent = n;
  $("meta-params").textContent = Object.values(modelJson.tensors)
    .reduce((a, t) => a + t.shape.reduce((x, y) => x * y, 1), 0).toLocaleString();
  $("meta-epoch").textContent = modelJson.epoch;
  $("meta-val").textContent = modelJson.val_loss.toFixed(4);

  buildFeatureToggles();
  $("scrub").max = String(T - 1);
  $("scrub").value = String(T - 1);
  S.mode = "shipped";
  S.shipped = { X: S.X, M: S.M, dt: S.dt, dates: S.dates, raw: S.raw, T, n,
                refLabels: S.refLabels, refLatent: S.refLatent };
  wireCustomSource();
  await run();
}

function buildFeatureToggles() {
  const box = $("feature-list");
  box.innerHTML = "";
  S.features.forEach((f, j) => {
    let obs = 0;
    for (let t = 0; t < S.T; t++) obs += S.M[t * S.n + j];
    const id = `f-${j}`;
    const row = document.createElement("label");
    row.className = "toggle";
    row.innerHTML =
      `<input type="checkbox" id="${id}" checked data-ix="${j}">` +
      `<span class="tname">${f}</span>` +
      `<span class="tcov">${((obs / S.T) * 100).toFixed(0)}%</span>`;
    box.appendChild(row);
  });
  box.onchange = () => {
    $("run").classList.add("dirty");
    const off = [...box.querySelectorAll("input")].filter((i) => !i.checked).length;
    $("mask-note").textContent = off ? `${off} series withheld` : "All series observed";
  };
}

function maskedCopy() {
  const off = [...$("feature-list").querySelectorAll("input")]
    .filter((i) => !i.checked).map((i) => +i.dataset.ix);
  if (!off.length) return { M: S.M, off };
  const M = Float64Array.from(S.M);
  for (let t = 0; t < S.T; t++) for (const j of off) M[t * S.n + j] = 0;
  return { M, off };
}

/* ---- Custom CSV upload ---------------------------------------------- */

function setSourceTab(which) {
  $("src-shipped").classList.toggle("active", which === "shipped");
  $("src-upload").classList.toggle("active", which === "upload");
  $("upload-box").hidden = which !== "upload";
}

function useShippedSource() {
  if (!S.shipped) return;
  Object.assign(S, {
    X: S.shipped.X, M: S.shipped.M, dt: S.shipped.dt,
    dates: S.shipped.dates, raw: S.shipped.raw,
    T: S.shipped.T, n: S.shipped.n, mode: "shipped",
    refLabels: S.shipped.refLabels, refLatent: S.shipped.refLatent,
    customMeta: null, customName: null,
  });
  $("meta-span").textContent = `${S.scaler.start} to ${S.scaler.end}`;
  $("meta-rows").textContent = S.T.toLocaleString();
  $("src-note").textContent = `Shipped panel: ${S.scaler.start} to ${S.scaler.end}.`;
  $("upload-note").textContent = "No file chosen.";
  setSourceTab("shipped");
  buildFeatureToggles();
  $("scrub").max = String(S.T - 1);
  $("scrub").value = String(S.T - 1);
  run();
}

async function handleCustomFile(file) {
  if (!file) return;
  status(`Reading ${file.name}`, true);
  // 5 MB guard keeps a pathological spreadsheet from freezing the tab.
  if (file.size > 8 * 1024 * 1024) {
    $("upload-note").textContent = `File too large (${(file.size / 1048576).toFixed(1)} MB). Keep uploads under 8 MB.`;
    status("Upload rejected", false);
    return;
  }
  let text;
  try {
    text = await file.text();
  } catch (e) {
    $("upload-note").textContent = "Could not read that file: " + e.message;
    status("Upload failed", false);
    return;
  }
  let parsed;
  try {
    parsed = parseCsvRobust(text);
  } catch (e) {
    $("upload-note").textContent = "Could not parse that CSV: " + e.message;
    status("Upload failed", false);
    return;
  }
  if (!parsed.head.length) {
    $("upload-note").textContent = "Empty file: expected a header row plus data rows.";
    status("Upload failed", false);
    return;
  }
  const built = buildCustomPanel(parsed, {
    mean: S.scaler.mean, std: S.scaler.std,
  }, { scalerMode: $("scaler-mode").value });
  if (!built.ok) {
    $("upload-note").textContent = built.error;
    status("Upload rejected", false);
    return;
  }
  Object.assign(S, {
    X: built.X, M: built.M, dt: built.dt, dates: built.dates, raw: built.raw,
    T: built.T, n: built.n, mode: "custom",
    refLabels: null, refLatent: null,
    customMeta: built, customName: file.name,
  });
  $("meta-span").textContent = `${built.start} to ${built.end} (upload)`;
  $("meta-rows").textContent = built.T.toLocaleString();
  const cov = FEATURES_COVERAGE_LINE(built.coverage);
  const warn = built.warnings.length ? " " + built.warnings.join(" ") : "";
  $("upload-note").textContent =
    `${file.name}: ${built.T.toLocaleString()} rows, ${built.start} to ${built.end}. ${cov}${warn}`;
  $("src-note").textContent = `Custom upload: ${file.name} (${built.T.toLocaleString()} rows).`;
  setSourceTab("upload");
  buildFeatureToggles();
  $("scrub").max = String(S.T - 1);
  $("scrub").value = String(S.T - 1);
  await run();
}

function FEATURES_COVERAGE_LINE(coverage) {
  const present = Object.entries(coverage).filter(([, c]) => c > 0.01).length;
  const missing = Object.entries(coverage).filter(([, c]) => c <= 0.01).map(([f]) => f);
  let s = `${present}/18 series observed.`;
  if (missing.length) s += ` Masked throughout: ${missing.join(", ")}.`;
  return s;
}

function wireCustomSource() {
  $("src-shipped").addEventListener("click", () => {
    if (S.mode !== "shipped") useShippedSource();
    else setSourceTab("shipped");
  });
  $("src-upload").addEventListener("click", () => setSourceTab("upload"));
  $("file").addEventListener("change", (e) => {
    handleCustomFile(e.target.files[0]).catch((err) => {
      $("upload-note").textContent = "Upload failed: " + err.message;
      status("Upload failed", false);
    });
    e.target.value = "";
  });
  $("scaler-mode").addEventListener("change", () => {
    if (S.mode === "custom" && S.customName)
      $("upload-note").textContent += " (Scaling choice applies to the next file you choose.)";
  });
  $("reset-src").addEventListener("click", useShippedSource);
  $("template").addEventListener("click", () => {
    const blob = new Blob([templateCsv()], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "regime_template.csv";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  });
}

/* Yield to the renderer between chunks so the progress bar and the latent
   trace paint as the filter advances. Falls back to a timer when
   requestAnimationFrame is throttled (background tabs) or never fires
   (headless browsers) — without this the run stalls with no feedback. */
const nextFrame = () => new Promise((r) => {
  let done = false;
  const go = () => { if (!done) { done = true; r(); } };
  if (typeof requestAnimationFrame === "function") {
    try { requestAnimationFrame(go); } catch (e) { go(); }
  }
  setTimeout(go, 50);
});

async function run() {
  if (S.running) return;
  S.running = true;
  $("run").classList.remove("dirty");
  $("run").disabled = true;
  const t0 = performance.now();
  const { M, off } = maskedCopy();
  S.activeMask = M;
  S.maskedOff = off;
  const { T, n, model } = S;
  const dz = model.dz;

  status("Encoding the initial state", true);
  await nextFrame();
  const { m0, P0 } = model.encodeInitial(S.X, M, T);
  const tEnc = performance.now();

  const mf = new Float64Array(T * dz);
  const xh = new Float64Array(T * n);
  S.mf = mf;
  S.xhat = xh;
  let m = m0, P = P0;
  const CHUNK = 220;
  for (let t = 0; t < T; ) {
    const end = Math.min(T, t + CHUNK);
    for (; t < end; t++) {
      const r = ekfStep(model, m, P, S.X.subarray(t * n, (t + 1) * n),
                        M.subarray(t * n, (t + 1) * n), S.dt[t]);
      m = r.m; P = r.P;
      mf.set(m, t * dz);
      xh.set(r.xhat, t * n);
    }
    S.Pdiag = P;
    charts.drawLatent($("c-latent"), { mf, dz, upTo: t, dates: S.dates });
    $("progress").style.width = ((t / T) * 100).toFixed(1) + "%";
    status(`Filtering ${t.toLocaleString()} of ${T.toLocaleString()} observations`, true);
    await nextFrame();
  }
  const tFilt = performance.now();

  status("Fitting mixtures", true);
  await nextFrame();
  const mu = new Float64Array(dz), sd = new Float64Array(dz);
  for (let j = 0; j < dz; j++) {
    let s = 0;
    for (let t = 0; t < T; t++) s += mf[t * dz + j];
    mu[j] = s / T;
    let v = 0;
    for (let t = 0; t < T; t++) v += (mf[t * dz + j] - mu[j]) ** 2;
    sd[j] = Math.sqrt(v / T) || 1;
  }
  const Z = new Float64Array(T * dz);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < dz; j++) Z[t * dz + j] = (mf[t * dz + j] - mu[j]) / sd[j];
  S.Z = Z;

  const ks = [2, 3, 4, 5, 6];
  const bics = [];
  S.fits = {};
  for (const k of ks) {
    const g = gmm(Z, T, dz, k, 7);
    S.fits[k] = g;
    bics.push(g.bic);
    await nextFrame();
  }
  S.ks = ks;
  S.bics = bics;
  S.bestK = ks[bics.indexOf(Math.min(...bics))];
  const sel = $("kselect");
  if (sel.value === "auto") S.k = S.bestK;
  else S.k = +sel.value;

  const tCluster = performance.now();
  S.timing = {
    encode: tEnc - t0,
    filter: tFilt - tEnc,
    cluster: tCluster - tFilt,
    total: tCluster - t0,
  };
  finalize();
  $("run").disabled = false;
  S.running = false;
}

/* Order regimes by average volatility and name them from their observable
   characteristics, matching the Python pipeline so colors and labels line up
   between the report and this page. */
function finalize() {
  const { T, k } = S;
  const raw = S.fits[k].labels;
  const empty = new Float64Array(T).fill(NaN);
  const vix = S.raw.vix || empty;
  const meanVix = [];
  for (let c = 0; c < k; c++) {
    let s = 0, cnt = 0;
    for (let t = 0; t < T; t++) if (raw[t] === c && Number.isFinite(vix[t])) { s += vix[t]; cnt++; }
    meanVix.push(cnt ? s / cnt : Infinity);
  }
  const order = meanVix.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
  const remap = new Int32Array(k);
  order.forEach((oldC, newC) => { remap[oldC] = newC; });
  const relabeled = new Int32Array(T);
  for (let t = 0; t < T; t++) relabeled[t] = remap[raw[t]];
  const labels = modalSmooth(relabeled, +$("smooth").value);
  S.labels = labels;

  const stats = [];
  for (let c = 0; c < k; c++) {
    const idx = [];
    for (let t = 0; t < T; t++) if (labels[t] === c) idx.push(t);
    const avg = (arr) => {
      let s = 0, n2 = 0;
      for (const t of idx) if (Number.isFinite(arr[t])) { s += arr[t]; n2++; }
      return n2 ? s / n2 : NaN;
    };
    const rets = idx.map((t) => (S.raw.spx_ret || empty)[t]).filter(Number.isFinite);
    const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
    const varr = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length || 1);
    let runs = [], cur = 0;
    for (let t = 0; t < T; t++) {
      if (labels[t] === c) cur++;
      else if (cur) { runs.push(cur); cur = 0; }
    }
    if (cur) runs.push(cur);
    const y10 = S.raw.y_10y || empty, y3 = S.raw.y_3m || empty;
    const slope = idx.map((t) => y10[t] - y3[t]).filter(Number.isFinite);
    stats.push({
      regime: c,
      days: idx.length,
      occupancy: (100 * idx.length) / T,
      vix: avg(S.raw.vix || empty),
      hy_oas: avg(S.raw.hy_oas || empty),
      slope: slope.reduce((a, b) => a + b, 0) / (slope.length || 1),
      ann_ret: 100 * 252 * mean,
      ann_vol: 100 * Math.sqrt(252 * varr),
      spell: runs.length ? runs.reduce((a, b) => a + b, 0) / runs.length : 0,
    });
  }
  S.stats = stats;
  S.names = regimeNames(stats);

  const P = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let t = 1; t < T; t++) P[labels[t - 1]][labels[t]]++;
  for (let i = 0; i < k; i++) {
    const s = P[i].reduce((a, b) => a + b, 0) || 1;
    for (let j = 0; j < k; j++) P[i][j] /= s;
  }
  S.P = P;
  render();
}

function regimeNames(rows) {
  const k = rows.length;
  const sorted = rows.map((r) => r.hy_oas).sort((a, b) => a - b);
  const mid = k >> 1;
  const hyMed = k % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const names = new Array(k).fill("");
  names[0] = "Calm Risk-On";
  if (k > 1) names[k - 1] = "Crisis / Stress";
  for (let i = 1; i < k - 1; i++) {
    const r = rows[i];
    names[i] = r.slope < 0.3 ? "Late-Cycle / Flat"
      : r.ann_ret < 0 ? "Risk-Off / Drawdown"
      : r.hy_oas > hyMed ? "Recovery / Wide Credit"
      : "Mid-Cycle";
  }
  const seen = {};
  return names.map((nm) => {
    seen[nm] = (seen[nm] || 0) + 1;
    return seen[nm] > 1 ? `${nm} ${seen[nm]}` : nm;
  });
}

function render() {
  const { T, k, labels, names, stats, timing } = S;
  const scrub = Math.max(0, Math.min(T - 1, +$("scrub").value));
  const seriesOf = (name) => (S.raw[name] ? Array.from(S.raw[name]) : []);

  charts.drawSeries($("c-equity"), {
    values: seriesOf("spx_close"), dates: S.dates, labels,
    log: true, title: S.raw.spx_close ? "Equity index, log scale" : "Equity index (not in upload)", marker: scrub,
  });
  charts.drawSeries($("c-vix"), {
    values: seriesOf("vix"), dates: S.dates, labels,
    title: S.raw.vix ? "Volatility index" : "Volatility index (not in upload)", marker: scrub,
  });
  charts.drawSeries($("c-hy"), {
    values: seriesOf("hy_oas"), dates: S.dates, labels,
    title: S.raw.hy_oas ? "High-yield credit spread, %" : "High-yield spread (not in upload)", marker: scrub,
  });
  charts.drawLatent($("c-latent"), {
    mf: S.mf, dz: S.model.dz, upTo: T, dates: S.dates, labels, marker: scrub,
  });
  charts.drawTransitions($("c-trans"), { P: S.P, names });
  charts.drawBic($("c-bic"), { ks: S.ks, bics: S.bics, chosen: k });

  const legend = $("legend");
  legend.innerHTML = "";
  names.forEach((nm, i) => {
    const el = document.createElement("div");
    el.className = "leg";
    el.innerHTML = `<i style="background:${charts.REGIME_COLORS[i]}"></i>` +
      `<span>${nm}</span><b>${stats[i].occupancy.toFixed(1)}%</b>`;
    legend.appendChild(el);
  });

  const tb = $("stats-body");
  tb.innerHTML = "";
  stats.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td><i class="swatch" style="background:${charts.REGIME_COLORS[i]}"></i>${names[i]}</td>` +
      `<td class="num">${pct(r.occupancy)}</td><td class="num">${r.spell.toFixed(0)}d</td>` +
      `<td class="num">${fmt(r.vix, 1)}</td><td class="num">${fmt(r.hy_oas)}</td>` +
      `<td class="num">${r.slope >= 0 ? "+" : ""}${fmt(r.slope)}</td>` +
      `<td class="num">${r.ann_ret >= 0 ? "+" : ""}${fmt(r.ann_ret, 1)}%</td>` +
      `<td class="num">${fmt(r.ann_vol, 1)}%</td>`;
    tb.appendChild(tr);
  });

  const custom = S.mode === "custom";
  let maxDev = 0, ari = NaN;
  if (!custom && S.refLatent && S.refLabels) {
    for (let t = 0; t < T; t++)
      for (let j = 0; j < S.model.dz; j++)
        maxDev = Math.max(maxDev, Math.abs(S.mf[t * S.model.dz + j] - S.refLatent[t][j]));
    ari = adjustedRand(Array.from(labels), S.refLabels);
  }
  const masked = S.maskedOff.length > 0;
  if (custom) {
    $("ver-title").textContent = "Custom upload: no shipped reference";
    $("ver-dev").textContent = "n/a";
    $("ver-dev-label").textContent = "The shipped reference covers the shipped panel only, so there is nothing to compare this upload against.";
    $("ver-ari").textContent = "n/a";
    $("ver-ari-label").textContent = masked
      ? `Regime agreement with the unmasked upload at k=${k}`
      : `Regimes fitted on your upload at k=${k}; try other counts and smoothing above`;
  } else {
    $("ver-title").textContent = masked
      ? "Effect of withholding those series"
      : "Agreement with the Python implementation";
    $("ver-dev").textContent = maxDev < 0.01 ? maxDev.toExponential(1) : maxDev.toFixed(3);
    $("ver-dev-label").textContent = masked
      ? "Largest shift in the latent path caused by withholding series"
      : "Largest deviation from the Python filter, which is the rounding precision of the shipped reference";
    $("ver-ari").textContent = ari.toFixed(3);
    $("ver-ari-label").textContent = masked
      ? `Regime agreement with the unmasked run at k=${k}`
      : `Regime agreement with the Python labelling, k=${k} against reference k=${S.summary.k}`;
  }
  $("ver-k").textContent = `${S.bestK}`;
  $("timing").textContent =
    `encoder ${timing.encode.toFixed(0)} ms · filter ${timing.filter.toFixed(0)} ms · ` +
    `clustering ${timing.cluster.toFixed(0)} ms`;
  $("progress").style.width = "100%";
  status(custom && !masked
    ? `Complete on upload (${S.customName || "custom CSV"})`
    : masked
      ? `Complete, ${S.maskedOff.length} series withheld`
      : "Complete", false);
  renderDetail();
}

function renderDetail() {
  if (!S.labels || !S.mf) return;
  const t = Math.max(0, Math.min(S.T - 1, +$("scrub").value));
  const { n, model } = S;
  $("scrub-date").textContent = S.dates[t];
  const lab = S.labels[t];
  $("scrub-regime").textContent = S.names[lab];
  $("scrub-regime").style.color = charts.REGIME_COLORS[lab];

  const z = S.mf.subarray(t * model.dz, (t + 1) * model.dz);
  $("scrub-latent").textContent = Array.from(z).map((v) => v.toFixed(2).padStart(6)).join(" ");

  const taus = S.scaler.maturities;
  const obs = S.scaler.yield_features.map((f) => S.raw[f] ? S.raw[f][t] : NaN);
  const curveAt = (tau) => model.yieldCurve(z, [tau])[0];
  charts.drawYieldCurve($("c-curve"), {
    taus, obs, fit: model.yieldCurve(z, taus), fitAt: curveAt,
    label: `Yield curve, ${S.dates[t]}`,
    color: charts.REGIME_COLORS[lab],
  });

  charts.drawRecon($("c-recon"), {
    features: S.features,
    x: S.X.subarray(t * n, (t + 1) * n),
    xhat: S.xhat.subarray(t * n, (t + 1) * n),
    mask: S.activeMask.subarray(t * n, (t + 1) * n),
  });
}

$("run").addEventListener("click", run);
$("scrub").addEventListener("input", () => {
  if (!S.labels) return;
  renderDetail();
  render();
});
$("kselect").addEventListener("change", () => {
  if (!S.fits) return;
  S.k = $("kselect").value === "auto" ? S.bestK : +$("kselect").value;
  finalize();
});
$("smooth").addEventListener("input", () => {
  $("smooth-val").textContent = $("smooth").value;
  if (S.fits) finalize();
});
$("reset-mask").addEventListener("click", () => {
  $("feature-list").querySelectorAll("input").forEach((i) => (i.checked = true));
  $("mask-note").textContent = "All series observed";
  run();
});
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.labels) render(); }, 150);
});

boot().catch((err) => {
  status("Could not load shipped data: " + err.message + " — you can still use My CSV above.");
  try { setSourceTab("upload"); } catch (e) { /* pre-wire fallback */ }
  try { wireCustomSource(); } catch (e) { /* already wired */ }
  console.error(err);
});
