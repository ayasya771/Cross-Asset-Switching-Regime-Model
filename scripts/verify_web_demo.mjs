/* Cross-implementation check: runs the browser engine under Node against the
   shipped dataset and compares its filtered latent trajectory and regime
   labels with the ones the Python pipeline produced.

   Usage: node scripts/verify_web_demo.mjs
   Exits non-zero if the two implementations disagree beyond the precision of
   the shipped reference, so CI catches any drift between them. */
import { readFileSync } from "fs";
import { RegimeModel, ekfStep, kmeans, gmm, modalSmooth, adjustedRand } from "../docs/js/engine.js";
import { parseCsvRobust, buildCustomPanel, templateCsv } from "../docs/js/custom.js";

import { fileURLToPath } from "url";
import path from "path";
const DATA_DIR = fileURLToPath(new URL("../docs/data/", import.meta.url));
const scaler = JSON.parse(readFileSync(path.join(DATA_DIR, "scaler.json"), "utf8"));
const model = new RegimeModel(JSON.parse(readFileSync(path.join(DATA_DIR, "model.json"), "utf8")));

const lines = readFileSync(path.join(DATA_DIR, "panel.csv"), "utf8").trim().split("\n");
const head = lines[0].split(",");
const feats = scaler.features, n = feats.length, T = lines.length - 1;
const colIx = feats.map(f => head.indexOf(f));
const dateIx = head.indexOf("date");
const isLog = new Set(scaler.log_features);

const X = new Float64Array(T * n), M = new Float64Array(T * n), dt = new Float64Array(T);
const dates = [];
for (let t = 0; t < T; t++) {
  const p = lines[t + 1].split(",");
  dates.push(p[dateIx]);
  for (let j = 0; j < n; j++) {
    const raw = p[colIx[j]];
    if (raw === "" || raw === undefined) { M[t * n + j] = 0; X[t * n + j] = 0; continue; }
    let v = parseFloat(raw);
    if (!Number.isFinite(v)) { M[t * n + j] = 0; X[t * n + j] = 0; continue; }
    if (isLog.has(feats[j])) v = Math.log(Math.max(v, 1e-4));
    M[t * n + j] = 1;
    X[t * n + j] = (v - scaler.mean[j]) / scaler.std[j];
  }
}
const dayMs = 86400000;
for (let t = 1; t < T; t++)
  dt[t] = (Date.parse(dates[t]) - Date.parse(dates[t - 1])) / dayMs / 365.25;

console.log(`panel ${T} rows x ${n} features, coverage ${(M.reduce((a,b)=>a+b,0)/(T*n)*100).toFixed(1)}%`);

const t0 = Date.now();
let { m0, P0 } = model.encodeInitial(X, M, T);
console.log(`GRU encoder pass: ${Date.now() - t0} ms`);

const t1 = Date.now();
let m = m0, P = P0;
const mf = new Float64Array(T * model.dz);
const xh = new Float64Array(T * n);
for (let t = 0; t < T; t++) {
  const r = ekfStep(model, m, P, X.subarray(t*n,(t+1)*n), M.subarray(t*n,(t+1)*n), dt[t]);
  m = r.m; P = r.P;
  mf.set(m, t * model.dz);
  xh.set(r.xhat, t * n);
}
console.log(`EKF over ${T} steps: ${Date.now() - t1} ms`);

const ref = readFileSync(path.join(DATA_DIR, "reference_latent.csv"), "utf8").trim().split("\n").map(l => l.split(",").map(Number));
let maxAbs = 0, sse = 0, refVar = 0;
const refMean = ref.flat().reduce((a,b)=>a+b,0)/(ref.length*model.dz);
for (let t = 0; t < T; t++)
  for (let j = 0; j < model.dz; j++) {
    const d = Math.abs(mf[t*model.dz+j] - ref[t][j]);
    maxAbs = Math.max(maxAbs, d); sse += d*d; refVar += (ref[t][j]-refMean)**2;
  }
console.log(`\nlatent vs Python reference: max abs dev ${maxAbs.toExponential(3)}, RMSE ${Math.sqrt(sse/(T*model.dz)).toExponential(3)}`);
console.log(`relative to reference spread: ${(Math.sqrt(sse/refVar)*100).toFixed(6)}%`);

/* clustering path */
const dz = model.dz;
const mu = new Float64Array(dz), sd = new Float64Array(dz);
for (let j = 0; j < dz; j++) {
  let s = 0; for (let t = 0; t < T; t++) s += mf[t*dz+j]; mu[j] = s/T;
  let v = 0; for (let t = 0; t < T; t++) v += (mf[t*dz+j]-mu[j])**2; sd[j] = Math.sqrt(v/T) || 1;
}
const Z = new Float64Array(T*dz);
for (let t = 0; t < T; t++) for (let j = 0; j < dz; j++) Z[t*dz+j] = (mf[t*dz+j]-mu[j])/sd[j];

const refLab = readFileSync(path.join(DATA_DIR, "reference.csv"),"utf8").trim().split("\n").slice(1).map(l => +l.split(",")[1]);
for (const k of [2,3,4,5,6]) {
  const t = Date.now();
  const g = gmm(Z, T, dz, k, 7);
  const lab = modalSmooth(g.labels, 5);
  console.log(`k=${k}: browser GMM vs Python GMM labels ARI ${adjustedRand(Array.from(lab), refLab).toFixed(3)}  BIC ${g.bic.toFixed(0)}  (${Date.now()-t} ms)`);
}

const TOL = 1e-5;
let devOk = maxAbs < TOL;
console.log(`\n${devOk ? "PASS" : "FAIL"}: browser engine matches the Python filter within ${TOL}`);
if (!devOk) process.exit(1);

/* Custom-upload path: rebuilding the shipped panel through buildCustomPanel
   (shipped scaler) must reproduce the direct X/M/dt above, proving the upload
   pipeline applies the documented transforms instead of a parallel formula. */
const panelText = readFileSync(path.join(DATA_DIR, "panel.csv"), "utf8");
const rebuilt = buildCustomPanel(parseCsvRobust(panelText), scaler, { scalerMode: "shipped" });
if (!rebuilt.ok) { console.log(`FAIL: custom builder rejected panel.csv: ${rebuilt.error}`); process.exit(1); }
let maxX = 0, maxDt = 0, maskDiff = 0;
for (let t = 0; t < T; t++) {
  maxDt = Math.max(maxDt, Math.abs(rebuilt.dt[t] - dt[t]));
  for (let j = 0; j < n; j++) {
    maxX = Math.max(maxX, Math.abs(rebuilt.X[t * n + j] - X[t * n + j]));
    if ((rebuilt.M[t * n + j] ? 1 : 0) !== (M[t * n + j] ? 1 : 0)) maskDiff++;
  }
}
const tpl = parseCsvRobust(templateCsv());
const tplOk = tpl.head[0] === "date" && tpl.rows.length >= 1;
console.log(`custom builder vs direct: max |dX| ${maxX.toExponential(2)}, max |ddt| ${maxDt.toExponential(2)}, mask diffs ${maskDiff}, template ${tplOk ? "ok" : "BAD"}`);
if (!(maxX < 1e-12 && maxDt < 1e-12 && maskDiff === 0 && tplOk)) {
  console.log("FAIL: custom upload pipeline diverges from the shipped path");
  process.exit(1);
}
console.log("PASS: custom upload pipeline reproduces the shipped panel");
