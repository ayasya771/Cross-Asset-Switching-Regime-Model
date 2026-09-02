/* Wiring for the demo: fetch the shipped dataset, reproduce the preprocessing,
   run the encoder and the extended Kalman filter in the browser, cluster the
   filtered trajectory, and render the panels.

   Training happens offline in Python. What runs here is the inference half of
   the pipeline, against the weights exported from the trained checkpoint. */

import { RegimeModel, ekfStep, gmm, modalSmooth, adjustedRand } from "./engine.js";
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
  const lines = text.trim().split("\n");
  const head = lines[0].split(",");
  return { head, rows: lines.slice(1).map((l) => l.split(",")) };
}

async function boot() {
  status("fetching dataset", true);
  const [scaler, modelJson, panelTxt, refTxt, summary, refLatentTxt] = await Promise.all([
    fetch(DATA + "scaler.json").then((r) => r.json()),
    fetch(DATA + "model.json").then((r) => r.json()),
    fetch(DATA + "panel.csv").then((r) => r.text()),
    fetch(DATA + "reference.csv").then((r) => r.text()),
    fetch(DATA + "summary.json").then((r) => r.json()),
    fetch(DATA + "reference_latent.csv").then((r) => r.text()),
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
  box.addEventListener("change", () => {
    $("run").classList.add("dirty");
    const off = [...box.querySelectorAll("input")].filter((i) => !i.checked).length;
    $("mask-note").textContent = off ? `${off} feature${off > 1 ? "s" : ""} withheld` : "all features observed";
  });
}

function maskedCopy() {
  const off = [...$("feature-list").querySelectorAll("input")]
    .filter((i) => !i.checked).map((i) => +i.dataset.ix);
  if (!off.length) return { M: S.M, off };
  const M = Float64Array.from(S.M);
  for (let t = 0; t < S.T; t++) for (const j of off) M[t * S.n + j] = 0;
  return { M, off };
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

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

  status("encoding initial state", true);
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
    status(`filtering ${t.toLocaleString()} / ${T.toLocaleString()} observations`, true);
    await nextFrame();
  }
  const tFilt = performance.now();

  status("clustering latent trajectory", true);
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
  const vix = S.raw.vix;
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
    const rets = idx.map((t) => S.raw.spx_ret[t]).filter(Number.isFinite);
    const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
    const varr = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length || 1);
    let runs = [], cur = 0;
    for (let t = 0; t < T; t++) {
      if (labels[t] === c) cur++;
      else if (cur) { runs.push(cur); cur = 0; }
    }
    if (cur) runs.push(cur);
    const slope = idx.map((t) => S.raw.y_10y[t] - S.raw.y_3m[t]).filter(Number.isFinite);
    stats.push({
      regime: c,
      days: idx.length,
      occupancy: (100 * idx.length) / T,
      vix: avg(S.raw.vix),
      hy_oas: avg(S.raw.hy_oas),
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
  const scrub = +$("scrub").value;

  charts.drawSeries($("c-equity"), {
    values: Array.from(S.raw.spx_close), dates: S.dates, labels,
    log: true, title: "Equity index, log scale", marker: scrub,
  });
  charts.drawSeries($("c-vix"), {
    values: Array.from(S.raw.vix), dates: S.dates, labels,
    title: "Volatility index", marker: scrub,
  });
  charts.drawSeries($("c-hy"), {
    values: Array.from(S.raw.hy_oas), dates: S.dates, labels,
    title: "High-yield credit spread, %", marker: scrub,
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

  let maxDev = 0;
  for (let t = 0; t < T; t++)
    for (let j = 0; j < S.model.dz; j++)
      maxDev = Math.max(maxDev, Math.abs(S.mf[t * S.model.dz + j] - S.refLatent[t][j]));
  const ari = adjustedRand(Array.from(labels), S.refLabels);
  const masked = S.maskedOff.length > 0;
  $("ver-title").textContent = masked
    ? "Deviation from the unmasked reference run"
    : "Agreement with the Python implementation";
  $("ver-dev").textContent = maxDev < 0.01 ? maxDev.toExponential(1) : maxDev.toFixed(3);
  $("ver-dev-label").textContent = masked
    ? "max latent shift caused by withholding features"
    : "max latent deviation vs Python (float precision of the shipped reference)";
  $("ver-ari").textContent = ari.toFixed(3);
  $("ver-ari-label").textContent = masked
    ? `regime agreement with the unmasked run (k=${k})`
    : `regime agreement with the Python reference labeling (k=${k} vs reference k=${S.summary.k})`;
  $("ver-k").textContent = `${S.bestK}`;
  $("timing").textContent =
    `encoder ${timing.encode.toFixed(0)} ms · filter ${timing.filter.toFixed(0)} ms · ` +
    `clustering ${timing.cluster.toFixed(0)} ms`;
  $("progress").style.width = "100%";
  status(masked ? `complete, ${S.maskedOff.length} feature(s) withheld` : "complete", false);
  renderDetail();
}

function renderDetail() {
  const t = +$("scrub").value;
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
  $("mask-note").textContent = "all features observed";
  run();
});
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.labels) render(); }, 150);
});

boot().catch((err) => {
  status("failed to load: " + err.message);
  console.error(err);
});
