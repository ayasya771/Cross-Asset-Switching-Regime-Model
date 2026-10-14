/* Numerical core of the browser demo: dense linear algebra, a port of the
   trained model's forward pass, the continuous-discrete extended Kalman
   filter, and k-means regime clustering.

   The Jacobians the filter needs are written analytically here rather than by
   autodiff. Both networks are Linear-Tanh-Linear, so the derivatives are exact
   and match what torch.func.jacfwd computes in the Python implementation. */

export const EPS_JITTER = 1e-8;

export function mat(r, c, fill) {
  return { r, c, d: fill ? Float64Array.from(fill) : new Float64Array(r * c) };
}
export function eye(n, s = 1) {
  const A = mat(n, n);
  for (let i = 0; i < n; i++) A.d[i * n + i] = s;
  return A;
}
export function diag(v) {
  const n = v.length, A = mat(n, n);
  for (let i = 0; i < n; i++) A.d[i * n + i] = v[i];
  return A;
}
export function matmul(A, B) {
  const C = mat(A.r, B.c);
  for (let i = 0; i < A.r; i++)
    for (let k = 0; k < A.c; k++) {
      const a = A.d[i * A.c + k];
      if (a === 0) continue;
      for (let j = 0; j < B.c; j++) C.d[i * B.c + j] += a * B.d[k * B.c + j];
    }
  return C;
}
export function matmulNT(A, B) {
  const C = mat(A.r, B.r);
  for (let i = 0; i < A.r; i++)
    for (let j = 0; j < B.r; j++) {
      let s = 0;
      for (let k = 0; k < A.c; k++) s += A.d[i * A.c + k] * B.d[j * B.c + k];
      C.d[i * B.r + j] = s;
    }
  return C;
}
export function matvec(A, v) {
  const out = new Float64Array(A.r);
  for (let i = 0; i < A.r; i++) {
    let s = 0;
    for (let j = 0; j < A.c; j++) s += A.d[i * A.c + j] * v[j];
    out[i] = s;
  }
  return out;
}
export function symmetrize(A, jitter = 0) {
  const n = A.r, B = mat(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) B.d[i * n + j] = 0.5 * (A.d[i * n + j] + A.d[j * n + i]);
  for (let i = 0; i < n; i++) B.d[i * n + i] += jitter;
  return B;
}

/* Lower-triangular Cholesky factor. Falls back to a ridge if the matrix loses
   positive definiteness, which keeps a long recursion from derailing on a
   numerically marginal step. */
export function cholesky(A) {
  const n = A.r, L = mat(n, n);
  for (let ridge = 0; ridge < 6; ridge++) {
    L.d.fill(0);
    let ok = true;
    const bump = ridge === 0 ? 0 : Math.pow(10, -12 + ridge);
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A.d[i * n + j] + (i === j ? bump : 0);
        for (let k = 0; k < j; k++) s -= L.d[i * n + k] * L.d[j * n + k];
        if (i === j) {
          if (!(s > 0)) { ok = false; break; }
          L.d[i * n + i] = Math.sqrt(s);
        } else {
          L.d[i * n + j] = s / L.d[j * n + j];
        }
      }
    }
    if (ok) return L;
  }
  throw new Error("covariance lost positive definiteness");
}

/* Solve A X = B given the Cholesky factor L of A. */
export function cholSolve(L, B) {
  const n = L.r, m = B.c, X = mat(n, m);
  const Y = new Float64Array(n * m);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < n; i++) {
      let s = B.d[i * m + j];
      for (let k = 0; k < i; k++) s -= L.d[i * n + k] * Y[k * m + j];
      Y[i * m + j] = s / L.d[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = Y[i * m + j];
      for (let k = i + 1; k < n; k++) s -= L.d[k * n + i] * X.d[k * m + j];
      X.d[i * m + j] = s / L.d[i * n + i];
    }
  }
  return X;
}

const tanh = Math.tanh;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const softplus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));

function b64ToF64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float64Array(bytes.buffer);
}

/* The trained model, reconstructed from the exported weights. */
export class RegimeModel {
  constructor(json) {
    this.cfg = json.config;
    this.t = {};
    for (const [k, v] of Object.entries(json.tensors)) {
      const data = b64ToF64(v.data);
      this.t[k] = v.shape.length === 2
        ? { r: v.shape[0], c: v.shape[1], d: data }
        : data;
    }
    this.dz = this.cfg.latent_dim;
    this.dx = this.cfg.obs_dim;
    this.nY = this.t["decoder.taus"].length;

    const A = matmulNT(this.t["drift.L"], this.t["drift.L"]);
    for (let i = 0; i < this.dz * this.dz; i++) A.d[i] = -(A.d[i] + this.t["drift.eps_eye"].d[i]);
    this.A = A;

    this.nsLambda = softplus(this.t["decoder.raw_lambda"][0]) + 0.05;
    const taus = this.t["decoder.taus"];
    this.Lns = mat(this.nY, 3);
    for (let i = 0; i < this.nY; i++) {
      const x = this.nsLambda * taus[i], ex = Math.exp(-x), l1 = (1 - ex) / x;
      this.Lns.d[i * 3] = 1;
      this.Lns.d[i * 3 + 1] = l1;
      this.Lns.d[i * 3 + 2] = l1 - ex;
    }
    this.R = new Float64Array(this.dx);
    for (let i = 0; i < this.dx; i++) {
      const sd = softplus(this.t["decoder.rho"][i]) + 1e-3;
      this.R[i] = sd * sd;
    }
  }

  /* Residual MLP shared by the drift and diffusion networks: returns the value
     and the hidden activations needed for the analytic Jacobian. */
  _res(prefix, z) {
    const W1 = this.t[`${prefix}.res.net.0.weight`], b1 = this.t[`${prefix}.res.net.0.bias`];
    const W2 = this.t[`${prefix}.res.net.2.weight`], b2 = this.t[`${prefix}.res.net.2.bias`];
    const h = new Float64Array(W1.r);
    for (let i = 0; i < W1.r; i++) {
      let s = b1[i];
      for (let j = 0; j < W1.c; j++) s += W1.d[i * W1.c + j] * z[j];
      h[i] = tanh(s);
    }
    const out = new Float64Array(W2.r);
    for (let i = 0; i < W2.r; i++) {
      let s = b2[i];
      for (let j = 0; j < W2.c; j++) s += W2.d[i * W2.c + j] * h[j];
      out[i] = s;
    }
    return { out, h, W1, W2 };
  }

  _resJac(r) {
    const { h, W1, W2 } = r, J = mat(W2.r, W1.c);
    for (let i = 0; i < W2.r; i++)
      for (let j = 0; j < W1.c; j++) {
        let s = 0;
        for (let k = 0; k < W1.r; k++) s += W2.d[i * W2.c + k] * (1 - h[k] * h[k]) * W1.d[k * W1.c + j];
        J.d[i * W1.c + j] = s;
      }
    return J;
  }

  drift(z) {
    const r = this._res("drift", z);
    const out = matvec(this.A, z);
    const b = this.t["drift.bias"];
    for (let i = 0; i < this.dz; i++) out[i] += b[i] + r.out[i];
    return { f: out, J: this._addJac(this.A, this._resJac(r)) };
  }

  _addJac(A, B) {
    const C = mat(A.r, A.c);
    for (let i = 0; i < A.d.length; i++) C.d[i] = A.d[i] + B.d[i];
    return C;
  }

  diffusion(z) {
    const r = this._res("diffusion", z), c = this.t["diffusion.c"];
    const g = new Float64Array(this.dz);
    for (let i = 0; i < this.dz; i++) g[i] = softplus(c[i] + r.out[i]) + 1e-4;
    return g;
  }

  /* Emission decoder: Nelson-Siegel head for the yield block, free linear head
     for everything else, plus the exact Jacobian of both. */
  decode(z) {
    const W0 = this.t["decoder.trunk.0.weight"], b0 = this.t["decoder.trunk.0.bias"];
    const H = W0.r, trunk = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      let s = b0[i];
      for (let j = 0; j < W0.c; j++) s += W0.d[i * W0.c + j] * z[j];
      trunk[i] = tanh(s);
    }
    const dTrunk = mat(H, this.dz);
    for (let i = 0; i < H; i++) {
      const g = 1 - trunk[i] * trunk[i];
      for (let j = 0; j < this.dz; j++) dTrunk.d[i * this.dz + j] = g * W0.d[i * W0.c + j];
    }
    const Wn = this.t["decoder.ns_head.weight"], bn = this.t["decoder.ns_head.bias"];
    const beta = new Float64Array(3);
    for (let i = 0; i < 3; i++) {
      let s = bn[i];
      for (let j = 0; j < H; j++) s += Wn.d[i * Wn.c + j] * trunk[j];
      beta[i] = s;
    }
    const yPct = matvec(this.Lns, beta);
    const yMean = this.t["decoder.y_mean"], yStd = this.t["decoder.y_std"];
    const Wo = this.t["decoder.other_head.weight"], bo = this.t["decoder.other_head.bias"];
    const nO = Wo.r, out = new Float64Array(this.dx);
    for (let i = 0; i < this.nY; i++) out[i] = (yPct[i] - yMean[i]) / yStd[i];
    for (let i = 0; i < nO; i++) {
      let s = bo[i];
      for (let j = 0; j < H; j++) s += Wo.d[i * Wo.c + j] * trunk[j];
      out[this.nY + i] = s;
    }
    const dBeta = matmul(Wn, dTrunk);
    const dY = matmul(this.Lns, dBeta);
    const J = mat(this.dx, this.dz);
    for (let i = 0; i < this.nY; i++)
      for (let j = 0; j < this.dz; j++) J.d[i * this.dz + j] = dY.d[i * this.dz + j] / yStd[i];
    const dO = matmul(Wo, dTrunk);
    for (let i = 0; i < nO; i++)
      for (let j = 0; j < this.dz; j++) J.d[(this.nY + i) * this.dz + j] = dO.d[i * this.dz + j];
    return { x: out, J, beta, yPct };
  }

  yieldCurve(z, taus) {
    const { beta } = this.decode(z);
    return taus.map((tau) => {
      const x = this.nsLambda * tau, ex = Math.exp(-x), l1 = (1 - ex) / x;
      return beta[0] + beta[1] * l1 + beta[2] * (l1 - ex);
    });
  }

  /* GRU recognition network over the reversed sequence, giving the initial
     filtering distribution. Gate order follows PyTorch: reset, update, new. */
  encodeInitial(X, M, T) {
    const Wi = this.t["encoder.gru.weight_ih_l0"], Wh = this.t["encoder.gru.weight_hh_l0"];
    const bi = this.t["encoder.gru.bias_ih_l0"], bh = this.t["encoder.gru.bias_hh_l0"];
    const H = Wh.c, n = this.dx;
    let h = new Float64Array(H);
    const inp = new Float64Array(2 * n);
    const gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (let t = T - 1; t >= 0; t--) {
      for (let j = 0; j < n; j++) {
        inp[j] = X[t * n + j] * M[t * n + j];
        inp[n + j] = M[t * n + j];
      }
      for (let i = 0; i < 3 * H; i++) {
        let s = bi[i];
        for (let j = 0; j < 2 * n; j++) s += Wi.d[i * Wi.c + j] * inp[j];
        gi[i] = s;
        let s2 = bh[i];
        for (let j = 0; j < H; j++) s2 += Wh.d[i * Wh.c + j] * h[j];
        gh[i] = s2;
      }
      const hn = new Float64Array(H);
      for (let i = 0; i < H; i++) {
        const r = sigmoid(gi[i] + gh[i]);
        const zz = sigmoid(gi[H + i] + gh[H + i]);
        const nn = tanh(gi[2 * H + i] + r * gh[2 * H + i]);
        hn[i] = (1 - zz) * nn + zz * h[i];
      }
      h = hn;
    }
    const Wo = this.t["encoder.head.weight"], bo = this.t["encoder.head.bias"];
    const o = new Float64Array(Wo.r);
    for (let i = 0; i < Wo.r; i++) {
      let s = bo[i];
      for (let j = 0; j < H; j++) s += Wo.d[i * Wo.c + j] * h[j];
      o[i] = s;
    }
    const m0 = o.slice(0, this.dz), var0 = new Float64Array(this.dz);
    for (let i = 0; i < this.dz; i++) var0[i] = softplus(o[this.dz + i]) + 1e-4;
    return { m0, P0: diag(var0) };
  }
}

/* One predict/update cycle of the continuous-discrete EKF. Rows of the
   observation Jacobian for missing dimensions are zeroed and their noise set
   to one, which is exactly equivalent to running the lower-dimensional update
   on the observed subset. */
export function ekfStep(model, m, P, x, mask, dt) {
  const dz = model.dz, dx = model.dx;
  const { f, J } = model.drift(m);
  const F = eye(dz);
  for (let i = 0; i < dz * dz; i++) F.d[i] += J.d[i] * dt;
  const mPred = new Float64Array(dz);
  for (let i = 0; i < dz; i++) mPred[i] = m[i] + f[i] * dt;
  const g = model.diffusion(m);
  const Q = new Float64Array(dz);
  for (let i = 0; i < dz; i++) Q[i] = g[i] * g[i] * dt;
  let PPred = matmulNT(matmul(F, P), F);
  for (let i = 0; i < dz; i++) PPred.d[i * dz + i] += Q[i];
  PPred = symmetrize(PPred, EPS_JITTER);

  const { x: xhat, J: Hj } = model.decode(mPred);
  const Hm = mat(dx, dz);
  const innov = new Float64Array(dx);
  const Rm = new Float64Array(dx);
  for (let i = 0; i < dx; i++) {
    const s = mask[i];
    for (let j = 0; j < dz; j++) Hm.d[i * dz + j] = Hj.d[i * dz + j] * s;
    innov[i] = (x[i] - xhat[i]) * s;
    Rm[i] = model.R[i] * s + (1 - s);
  }
  let S = matmulNT(matmul(Hm, PPred), Hm);
  for (let i = 0; i < dx; i++) S.d[i * dx + i] += Rm[i];
  S = symmetrize(S, EPS_JITTER);
  const Lc = cholesky(S);
  const PHt = matmulNT(PPred, Hm);
  const K = matmul(PHt, cholSolve(Lc, eye(dx)));

  const mNew = new Float64Array(dz);
  const Kin = matvec(K, innov);
  for (let i = 0; i < dz; i++) mNew[i] = mPred[i] + Kin[i];
  const KH = matmul(K, Hm), ImKH = eye(dz);
  for (let i = 0; i < dz * dz; i++) ImKH.d[i] -= KH.d[i];
  const KR = mat(dz, dx);
  for (let i = 0; i < dz; i++)
    for (let j = 0; j < dx; j++) KR.d[i * dx + j] = K.d[i * dx + j] * Rm[j];
  let PNew = matmulNT(matmul(ImKH, PPred), ImKH);
  const KRK = matmulNT(KR, K);
  for (let i = 0; i < dz * dz; i++) PNew.d[i] += KRK.d[i];
  return { m: mNew, P: symmetrize(PNew, EPS_JITTER), xhat };
}

/* Deterministic k-means with k-means++ seeding, used to turn the filtered
   latent trajectory into discrete regimes. */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const d2 = (Z, i, c, d) => {
  let s = 0;
  for (let j = 0; j < d; j++) { const v = Z[i * d + j] - c[j]; s += v * v; }
  return s;
};

export function kmeans(Z, n, d, k, seed = 7, iters = 60) {
  const rnd = mulberry32(seed);
  const cent = [];
  const first = Math.floor(rnd() * n);
  cent.push(Float64Array.from(Z.subarray(first * d, first * d + d)));
  const dist = new Float64Array(n).fill(Infinity);
  while (cent.length < k) {
    let tot = 0;
    for (let i = 0; i < n; i++) {
      dist[i] = Math.min(dist[i], d2(Z, i, cent[cent.length - 1], d));
      tot += dist[i];
    }
    let r = rnd() * tot, pick = n - 1;
    for (let i = 0; i < n; i++) { r -= dist[i]; if (r <= 0) { pick = i; break; } }
    cent.push(Float64Array.from(Z.subarray(pick * d, pick * d + d)));
  }
  const lab = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const dd = d2(Z, i, cent[c], d); if (dd < bd) { bd = dd; best = c; } }
      if (lab[i] !== best) { lab[i] = best; moved++; }
    }
    const sums = Array.from({ length: k }, () => new Float64Array(d));
    const cnt = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      cnt[lab[i]]++;
      for (let j = 0; j < d; j++) sums[lab[i]][j] += Z[i * d + j];
    }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let j = 0; j < d; j++) cent[c][j] = sums[c][j] / cnt[c];
    if (!moved && it > 0) break;
  }
  return lab;
}

export function modalSmooth(lab, w) {
  if (w <= 1) return Int32Array.from(lab);
  const n = lab.length, out = new Int32Array(n), half = w >> 1;
  const counts = new Map();
  for (let i = 0; i < n; i++) {
    counts.clear();
    for (let j = Math.max(0, i - half); j < Math.min(n, i + half + 1); j++)
      counts.set(lab[j], (counts.get(lab[j]) || 0) + 1);
    let best = lab[i], bc = -1;
    for (const [v, c] of counts) if (c > bc) { bc = c; best = v; }
    out[i] = best;
  }
  return out;
}

/* Adjusted Rand index between two labelings. */
export function adjustedRand(a, b) {
  const n = a.length;
  let ka = 0, kb = 0;
  for (let i = 0; i < n; i++) {
    if (a[i] + 1 > ka) ka = a[i] + 1;
    if (b[i] + 1 > kb) kb = b[i] + 1;
  }
  const tab = new Float64Array(ka * kb), ra = new Float64Array(ka), rb = new Float64Array(kb);
  for (let i = 0; i < n; i++) { tab[a[i] * kb + b[i]]++; ra[a[i]]++; rb[b[i]]++; }
  const c2 = (x) => (x * (x - 1)) / 2;
  let sij = 0, sa = 0, sb = 0;
  for (let i = 0; i < tab.length; i++) sij += c2(tab[i]);
  for (let i = 0; i < ka; i++) sa += c2(ra[i]);
  for (let i = 0; i < kb; i++) sb += c2(rb[i]);
  const total = c2(n), exp = (sa * sb) / total, max = 0.5 * (sa + sb);
  return max === exp ? 0 : (sij - exp) / (max - exp);
}

/* Full-covariance Gaussian mixture fitted by EM, seeded from k-means. The
   Python pipeline selects regimes the same way, so the browser reproduces the
   reference labeling rather than approximating it with spherical clusters. */
export function gmm(Z, n, d, k, seed = 7, iters = 200, reg = 1e-6) {
  const lab0 = kmeans(Z, n, d, k, seed);
  const means = Array.from({ length: k }, () => new Float64Array(d));
  const covs = Array.from({ length: k }, () => eye(d));
  const weights = new Float64Array(k).fill(1 / k);
  const resp = new Float64Array(n * k);
  for (let i = 0; i < n; i++) resp[i * k + lab0[i]] = 1;

  const LOG2PI = Math.log(2 * Math.PI);
  let prev = -Infinity;
  for (let it = 0; it < iters; it++) {
    /* M step */
    for (let c = 0; c < k; c++) {
      let nk = 0;
      for (let i = 0; i < n; i++) nk += resp[i * k + c];
      nk = Math.max(nk, 1e-10);
      weights[c] = nk / n;
      means[c].fill(0);
      for (let i = 0; i < n; i++) {
        const r = resp[i * k + c];
        if (r === 0) continue;
        for (let j = 0; j < d; j++) means[c][j] += r * Z[i * d + j];
      }
      for (let j = 0; j < d; j++) means[c][j] /= nk;
      const S = mat(d, d);
      const df = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const r = resp[i * k + c];
        if (r < 1e-12) continue;
        for (let j = 0; j < d; j++) df[j] = Z[i * d + j] - means[c][j];
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) S.d[a * d + b] += r * df[a] * df[b];
      }
      for (let a = 0; a < d * d; a++) S.d[a] /= nk;
      for (let a = 0; a < d; a++) S.d[a * d + a] += reg;
      covs[c] = symmetrize(S);
    }
    /* E step */
    const chols = covs.map(cholesky);
    const logDet = chols.map((L) => {
      let s = 0;
      for (let i = 0; i < d; i++) s += Math.log(L.d[i * d + i]);
      return 2 * s;
    });
    let ll = 0;
    const lp = new Float64Array(k), df = new Float64Array(d), y = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        const L = chols[c];
        for (let j = 0; j < d; j++) df[j] = Z[i * d + j] - means[c][j];
        let quad = 0;
        for (let a = 0; a < d; a++) {
          let s = df[a];
          for (let b = 0; b < a; b++) s -= L.d[a * d + b] * y[b];
          y[a] = s / L.d[a * d + a];
          quad += y[a] * y[a];
        }
        lp[c] = Math.log(Math.max(weights[c], 1e-300)) - 0.5 * (quad + logDet[c] + d * LOG2PI);
      }
      let mx = -Infinity;
      for (let c = 0; c < k; c++) if (lp[c] > mx) mx = lp[c];
      let sum = 0;
      for (let c = 0; c < k; c++) { lp[c] = Math.exp(lp[c] - mx); sum += lp[c]; }
      ll += mx + Math.log(sum);
      for (let c = 0; c < k; c++) resp[i * k + c] = lp[c] / sum;
    }
    if (it > 4 && Math.abs(ll - prev) < 1e-7 * Math.abs(ll)) break;
    prev = ll;
  }
  const labels = new Int32Array(n);
  const probs = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    let best = 0, bv = -1;
    for (let c = 0; c < k; c++) {
      probs[i * k + c] = resp[i * k + c];
      if (resp[i * k + c] > bv) { bv = resp[i * k + c]; best = c; }
    }
    labels[i] = best;
  }
  const nParams = k - 1 + k * d + (k * d * (d + 1)) / 2;
  return { labels, probs, means, weights, logLik: prev, bic: -2 * prev + nParams * Math.log(n) };
}
