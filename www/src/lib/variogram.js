export function computeEmpiricalVariogram(lats, lons, values, binCount = 18) {
  const n = values.length;
  if (n < 2) return [];

  let maxDistance = 0;
  const pairs = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dLat = lats[i] - lats[j];
      const dLon = lons[i] - lons[j];
      const distance = Math.sqrt(dLat * dLat + dLon * dLon);
      const semivariance = 0.5 * (values[i] - values[j]) ** 2;
      maxDistance = Math.max(maxDistance, distance);
      pairs.push({ distance, semivariance });
    }
  }
  if (maxDistance <= 0) return [];

  const width = maxDistance / binCount;
  const bins = Array.from({ length: binCount }, () => ({
    distanceSum: 0,
    semivarianceSum: 0,
    count: 0,
  }));
  for (const pair of pairs) {
    const index = Math.min(binCount - 1, Math.floor(pair.distance / width));
    bins[index].distanceSum += pair.distance;
    bins[index].semivarianceSum += pair.semivariance;
    bins[index].count += 1;
  }
  return bins
    .filter((bin) => bin.count > 0)
    .map((bin) => ({
      distance: bin.distanceSum / bin.count,
      semivariance: bin.semivarianceSum / bin.count,
      count: bin.count,
    }));
}

function logGamma(x) {
  if (x < 0.5) {
    const s = Math.PI / Math.sin(Math.PI * x);
    return Math.log(Math.abs(s)) - logGamma(1 - x);
  }
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  let xx = x - 1;
  let t = c[0];
  for (let i = 1; i < g + 2; i++) t += c[i] / (xx + i);
  t *= Math.sqrt(2 * Math.PI) * Math.pow(xx + g + 0.5, xx + 0.5) * Math.exp(-(xx + g + 0.5));
  return Math.log(t);
}

function besselK(nu, x) {
  if (x <= 0) return nu === 0 ? Infinity : 0;
  if (x > 300) return 0;
  if (nu === 0.5) return Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x);
  const eps = 1e-10;
  const maxIter = 300;
  const x2 = x / 2;
  function besselI(n, xx) {
    let sum = 0;
    let term = 1;
    let k = 0;
    const logX2 = Math.log(x2);
    while (Math.abs(term) > eps * (Math.abs(sum) || 1) && k < maxIter) {
      const logTerm = (n + 2 * k) * logX2 - logGamma(k + 1) - logGamma(n + k + 1);
      term = Math.exp(logTerm);
      sum += term;
      k++;
    }
    return sum * Math.pow(x2, n);
  }
  const sinPiNu = Math.sin(Math.PI * nu);
  if (Math.abs(sinPiNu) < 1e-10) return besselK(0.5, x);
  const iNeg = besselI(-nu, x);
  const iPos = besselI(nu, x);
  return (Math.PI / 2) * (iNeg - iPos) / sinPiNu;
}

export function variogramSemivariance(distance, modelType, nugget, sill, range, shape) {
  const r = Math.max(range, 1e-9);
  const partial = Math.max(sill - nugget, 1e-9);
  const h = Math.max(distance, 0);

  if (modelType === "spherical") {
    if (h >= r) return sill;
    const x = h / r;
    return nugget + partial * (1.5 * x - 0.5 * x * x * x);
  }
  if (modelType === "gaussian") {
    return nugget + partial * (1 - Math.exp(-3 * (h * h) / (r * r)));
  }
  if (modelType === "exponential") {
    return nugget + partial * (1 - Math.exp(-3 * h / r));
  }
  if (modelType === "cubic") {
    if (h >= r) return sill;
    const x = h / r;
    // Standard cubic model: reaches 1 exactly at x = 1 (Chilès & Delfiner 2012).
    const poly = 7 * x * x - 8.75 * x * x * x + 3.5 * Math.pow(x, 5) - 0.75 * Math.pow(x, 7);
    return nugget + partial * poly;
  }
  if (modelType === "stable") {
    const alpha = typeof shape === "number" && shape > 0 ? shape : 1;
    const x = Math.pow(h / r, alpha);
    return nugget + partial * (1 - Math.exp(-x));
  }
  if (modelType === "matern") {
    const nu = typeof shape === "number" && shape > 0 ? shape : 0.5;
    if (h <= 0) return nugget;
    const x = (h / r) * Math.sqrt(2 * nu);
    const kNu = besselK(nu, x);
    const logGammaNu = logGamma(nu);
    const gammaNu = logGammaNu < -1e10 ? 1 : Math.exp(logGammaNu);
    const factor = (Math.pow(2, 1 - nu) / gammaNu) * Math.pow(x, nu) * kNu;
    const correlation = Math.min(Math.max(factor, 0), 1);
    return nugget + partial * (1 - correlation);
  }
  return nugget + partial * (1 - Math.exp(-3 * h / r));
}

export function normalQuantile(p) {
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269,
    -30.66479806614716, 2.506628277459239,
  ];
  const b = [
    -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972,
    -13.28068155288572,
  ];
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p <= 0) return Number.NEGATIVE_INFINITY;
  if (p >= 1) return Number.POSITIVE_INFINITY;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  const q = p - 0.5;
  const r = q * q;
  return (
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}
