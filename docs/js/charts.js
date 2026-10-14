/* Canvas renderers for the demo. No plotting library: every panel is drawn
   directly so the page stays dependency-free and loads from a static host.

   Shared conventions: device-pixel-ratio aware sizing, one axis per panel,
   regime identity carried by legend and label as well as color, and tabular
   figures for anything that lines up in a column. */

export const REGIME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
const INK = "#ececec";
const INK2 = "#a3a3a3";
const MUTED = "#737373";
const GRID = "#2e2e2e";
const SURFACE = "#1a1a1a";
const MONO = '11px "IBM Plex Mono", ui-monospace, monospace';
const MONO_SM = '10px "IBM Plex Mono", ui-monospace, monospace';

export function prepare(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width, 120), h = Math.max(rect.height, 60);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function axes(ctx, box, yTicks, fmt) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.fillStyle = MUTED;
  ctx.font = MONO_SM;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of yTicks) {
    const y = Math.round(box.y(t)) + 0.5;
    if (y < box.top || y > box.top + box.h) continue;
    ctx.beginPath();
    ctx.moveTo(box.left, y);
    ctx.lineTo(box.left + box.w, y);
    ctx.stroke();
    ctx.fillText(fmt(t), box.left - 6, y);
  }
}

function makeBox(w, h, pad, xMin, xMax, yMin, yMax, log = false) {
  const left = pad.l, top = pad.t;
  const bw = w - pad.l - pad.r, bh = h - pad.t - pad.b;
  const ly = (v) => (log ? Math.log(Math.max(v, 1e-9)) : v);
  const y0 = ly(yMin), y1 = ly(yMax);
  return {
    left, top, w: bw, h: bh,
    x: (v) => left + ((v - xMin) / (xMax - xMin || 1)) * bw,
    y: (v) => top + bh - ((ly(v) - y0) / (y1 - y0 || 1)) * bh,
  };
}

function regimeBands(ctx, box, labels, colors, alpha = 0.22, ribbon = 5) {
  const n = labels.length;
  let s = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || labels[i] !== labels[s]) {
      const x0 = box.x(s), x1 = box.x(i), w = Math.max(x1 - x0, 0.7);
      ctx.fillStyle = colors[labels[s] % colors.length];
      ctx.globalAlpha = alpha;
      ctx.fillRect(x0, box.top, w, box.h);
      if (ribbon) {
        ctx.globalAlpha = 1;
        ctx.fillRect(x0, box.top + box.h - ribbon, w, ribbon);
      }
      s = i;
    }
  }
  ctx.globalAlpha = 1;
}

function yearTicks(ctx, box, dates, n) {
  ctx.fillStyle = MUTED;
  ctx.font = MONO_SM;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  let lastYear = null, step = 0;
  const years = [];
  for (let i = 0; i < n; i++) {
    const y = dates[i].slice(0, 4);
    if (y !== lastYear) { years.push([i, y]); lastYear = y; }
  }
  step = Math.max(1, Math.ceil(years.length / 12));
  years.forEach(([i, y], ix) => {
    if (ix % step) return;
    ctx.fillText(y, box.x(i), box.top + box.h + 6);
  });
}

export function drawSeries(canvas, opts) {
  const { ctx, w, h } = prepare(canvas);
  const { values, dates, labels, log, title, marker } = opts;
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return;
  let lo = Math.min(...finite), hi = Math.max(...finite);
  const padY = (hi - lo) * 0.08 || 1;
  lo -= padY; hi += padY;
  if (log) { lo = Math.max(lo, Math.min(...finite) * 0.9); }
  const box = makeBox(w, h, { l: 52, r: 10, t: 20, b: 20 }, 0, values.length - 1, lo, hi, log);

  ctx.fillStyle = INK2;
  ctx.font = '600 11.5px "Source Sans 3", system-ui, sans-serif';
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(title, box.left, 3);

  if (labels) regimeBands(ctx, box, labels, REGIME_COLORS);
  const ticks = log
    ? niceTicks(Math.log10(Math.max(lo, 1e-9)), Math.log10(hi), 4).map((e) => Math.pow(10, e))
    : niceTicks(lo, hi, 4);
  axes(ctx, box, ticks, (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(Math.abs(v) < 10 ? 1 : 0)));

  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) { started = false; continue; }
    const X = box.x(i), Y = box.y(values[i]);
    if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
  }
  ctx.stroke();
  yearTicks(ctx, box, dates, values.length);
  if (marker != null) {
    ctx.strokeStyle = INK2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(box.x(marker), box.top);
    ctx.lineTo(box.x(marker), box.top + box.h);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  return box;
}

/* Latent traces. ``upTo`` lets the caller draw the trajectory as the filter
   produces it, so the running filter is visible rather than merely reported. */
export function drawLatent(canvas, { mf, dz, upTo, dates, labels, marker }) {
  const { ctx, w, h } = prepare(canvas);
  const n = upTo;
  if (n < 2) return;
  const rows = Math.min(dz, 4);
  const rowH = (h - 16) / rows;
  for (let k = 0; k < rows; k++) {
    let lo = Infinity, hi = -Infinity;
    for (let t = 0; t < n; t++) {
      const v = mf[t * dz + k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const pad = (hi - lo) * 0.12 || 1;
    const box = makeBox(w, rowH, { l: 52, r: 10, t: 12, b: 6 }, 0, mf.length / dz - 1, lo - pad, hi + pad);
    box.top += k * rowH;
    const yFn = box.y;
    box.y = (v) => yFn(v) + k * rowH;

    if (labels) regimeBands(ctx, box, labels, REGIME_COLORS, 0.11, 3);
    ctx.fillStyle = INK2;
    ctx.font = MONO_SM;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`z${k + 1}`, box.left, box.top - 10);

    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 0; t < n; t++) {
      const X = box.x(t), Y = box.y(mf[t * dz + k]);
      t === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
    }
    ctx.stroke();
    if (marker != null && marker < n) {
      ctx.fillStyle = INK;
      ctx.beginPath();
      ctx.arc(box.x(marker), box.y(mf[marker * dz + k]), 2.5, 0, 7);
      ctx.fill();
    }
  }
  if (dates) {
    const box = makeBox(w, h, { l: 52, r: 10, t: 12, b: 14 }, 0, mf.length / dz - 1, 0, 1);
    yearTicks(ctx, box, dates, mf.length / dz);
  }
}

export function drawYieldCurve(canvas, { taus, obs, fit, fitAt, label, color }) {
  const { ctx, w, h } = prepare(canvas);
  const all = [...obs.filter(Number.isFinite), ...fit];
  if (!all.length) return;
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.25 || 0.5;
  const box = makeBox(w, h, { l: 40, r: 12, t: 22, b: 22 }, 0, Math.max(...taus), lo - pad, hi + pad);

  ctx.fillStyle = INK2;
  ctx.font = '600 11.5px "Source Sans 3", system-ui, sans-serif';
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, box.left, 3);
  axes(ctx, box, niceTicks(lo - pad, hi + pad, 4), (v) => v.toFixed(1));

  ctx.strokeStyle = color || REGIME_COLORS[0];
  ctx.lineWidth = 2;
  ctx.beginPath();
  const hi_t = Math.max(...taus), lo_t = Math.min(...taus);
  for (let i = 0; i <= 140; i++) {
    const tau = lo_t + (i / 140) * (hi_t - lo_t);
    const v = fitAt ? fitAt(tau) : interp(taus, fit, tau);
    const X = box.x(tau), Y = box.y(v);
    i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  }
  ctx.stroke();

  ctx.fillStyle = INK;
  taus.forEach((t, i) => {
    if (!Number.isFinite(obs[i])) return;
    ctx.beginPath();
    ctx.arc(box.x(t), box.y(obs[i]), 3.2, 0, 7);
    ctx.fill();
  });
  ctx.fillStyle = MUTED;
  ctx.font = MONO_SM;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  let lastX = -Infinity;
  taus.forEach((t) => {
    const X = box.x(t);
    if (X - lastX < 26) return;
    lastX = X;
    ctx.fillText(t < 1 ? `${Math.round(t * 12)}m` : `${t}y`, X, box.top + box.h + 6);
  });
}

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++)
    if (x <= xs[i]) {
      const f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + f * (ys[i] - ys[i - 1]);
    }
  return ys[ys.length - 1];
}

/* Observed against one-step-ahead prediction for a single date, with the
   masked dimensions called out rather than silently blank. */
export function drawRecon(canvas, { features, x, xhat, mask }) {
  const { ctx, w, h } = prepare(canvas);
  const n = features.length;
  const rowH = (h - 8) / n;
  const labelW = 78;
  const mid = labelW + (w - labelW - 12) / 2;
  const scale = Math.min((w - mid - 16) / 3.2, 110);
  ctx.font = MONO_SM;
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const y = 4 + i * rowH + rowH / 2;
    ctx.fillStyle = mask[i] ? INK2 : MUTED;
    ctx.textAlign = "right";
    ctx.fillText(features[i], labelW - 8, y);
    if (!mask[i]) {
      ctx.fillStyle = MUTED;
      ctx.textAlign = "center";
      ctx.fillText("not observed", mid, y);
      continue;
    }
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(mid, y - rowH / 2 + 1);
    ctx.lineTo(mid, y + rowH / 2 - 1);
    ctx.stroke();
    const bx = (v) => mid + Math.max(-3.2, Math.min(3.2, v)) * scale;
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(Math.min(mid, bx(xhat[i])), y - 3, Math.abs(bx(xhat[i]) - mid), 6);
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(bx(x[i]), y, 3, 0, 7);
    ctx.fill();
  }
}

export function drawTransitions(canvas, { P, names, durations }) {
  const { ctx, w, h } = prepare(canvas);
  const k = P.length;
  const padL = 96, padT = 18, padB = 8, padR = 8;
  const cw = (w - padL - padR) / k, ch = (h - padT - padB) / k;
  ctx.font = MONO_SM;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = P[i][j];
      ctx.fillStyle = SURFACE;
      ctx.fillRect(padL + j * cw, padT + i * ch, cw - 1.5, ch - 1.5);
      ctx.globalAlpha = 0.07 + 0.93 * Math.sqrt(v);
      ctx.fillStyle = "#2a78d6";
      ctx.fillRect(padL + j * cw, padT + i * ch, cw - 1.5, ch - 1.5);
      ctx.globalAlpha = 1;
      ctx.fillStyle = v > 0.40 ? "#ececec" : INK2;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(v >= 0.001 ? (v * 100).toFixed(1) : "0", padL + j * cw + cw / 2, padT + i * ch + ch / 2);
    }
    ctx.fillStyle = INK2;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(names[i].split(" /")[0], padL - 8, padT + i * ch + ch / 2);
    ctx.fillStyle = REGIME_COLORS[i % REGIME_COLORS.length];
    ctx.fillRect(padL - 5, padT + i * ch + ch / 2 - 4, 3, 8);
  }
  ctx.fillStyle = MUTED;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("row = from, column = to, values in % per day", padL, 3);
}

export function drawBic(canvas, { ks, bics, chosen }) {
  const { ctx, w, h } = prepare(canvas);
  const lo = Math.min(...bics), hi = Math.max(...bics);
  const box = makeBox(w, h, { l: 46, r: 10, t: 18, b: 18 }, ks[0], ks[ks.length - 1], lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.1);
  ctx.fillStyle = MUTED;
  ctx.font = MONO_SM;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("BIC by regime count (lower is better)", box.left, 2);
  axes(ctx, box, niceTicks(lo, hi, 3), (v) => (v / 1000).toFixed(0) + "k");
  ctx.strokeStyle = INK2;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ks.forEach((k, i) => {
    const X = box.x(k), Y = box.y(bics[i]);
    i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  });
  ctx.stroke();
  ks.forEach((k, i) => {
    ctx.fillStyle = k === chosen ? INK : MUTED;
    ctx.beginPath();
    ctx.arc(box.x(k), box.y(bics[i]), k === chosen ? 4.5 : 3, 0, 7);
    ctx.fill();
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = k === chosen ? INK : MUTED;
    ctx.fillText(String(k), box.x(k), box.top + box.h + 4);
  });
}
