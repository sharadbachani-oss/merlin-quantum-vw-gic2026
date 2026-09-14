"""THE CHEAP ADVERSARY for the VW certified-boundary claim.

Claim: validating the certificate against the measured 128-qubit collective spectrum puts the boundary at
0.60 rad/s, where white noise (100% safe everywhere) and AR(1) (25% at the boundary) both misjudge it.
Floor question: what is the CHEAPEST surrogate that reproduces the same boundary? If knowing one number —
the dominant collective line at 0.0625 cycles/step — is enough, then the spectrum's contribution to the
certificate is a lookup, not a computation, and the claim must say so.

Surrogates, in increasing order of what they know about the spectrum:
  W    white noise                                   (variance only)
  M    AR(1) matched lag-DT autocorrelation          (variance + one lag)
  AR2  AR(2) matched lag-1 and lag-2                 (variance + two lags)
  S1   single sinusoid at the dominant line          (ONE number: the peak frequency)
  BP   narrowband noise, dominant line +- 1 bin      (peak frequency + width)
  Q    the full measured spectrum                    (reference)
Writes floor_vw_surrogates.json.
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fable_vw_lyap as P
P.set_regime("hard") if "hard" in P.REGIMES else P.set_regime(list(P.REGIMES)[0])
DT = P.DT_EP; T_EP = 8.0; NST = int(T_EP / DT)
spec = json.load(open(os.path.join(HERE, "results", "vw_jam_relaxation_spectrum_v2_shim.json")))
SQ = spec["spectra"]["0.5"]
FREQS = np.array(SQ["freqs"][1:]); POWER = np.array(SQ["power"][1:]); POWER = POWER / POWER.sum()
F_PEAK = float(FREQS[int(np.argmax(POWER))])
DF = float(FREQS[1] - FREQS[0]) if len(FREQS) > 1 else 0.0625
print(f"measured spectrum: {len(FREQS)} bins, dominant line {F_PEAK:.4f} cyc/step, bin width {DF:.4f}")

def _norm(x): return x / (x.std(axis=1, keepdims=True) + 1e-12)

def synth(kind, n, dt_map, rng):
    t = np.arange(NST) * DT
    if kind == "W":
        return _norm(rng.normal(size=(n, NST)))
    if kind in ("M", "AR2"):
        # match the measured autocorrelation at lag DT (and lag 2DT for AR2)
        w = 2 * np.pi * FREQS / dt_map
        r1 = float((POWER * np.cos(w * DT)).sum())
        out = np.zeros((n, NST))
        if kind == "M":
            a = r1; x = rng.normal(size=n)
            for k in range(NST):
                x = a * x + math.sqrt(max(1 - a * a, 1e-9)) * rng.normal(size=n); out[:, k] = x
        else:
            r2 = float((POWER * np.cos(w * 2 * DT)).sum())
            a1 = r1 * (1 - r2) / max(1 - r1 * r1, 1e-9); a2 = (r2 - r1 * r1) / max(1 - r1 * r1, 1e-9)
            x1 = rng.normal(size=n); x2 = rng.normal(size=n)
            for k in range(NST):
                x = a1 * x1 + a2 * x2 + rng.normal(size=n) * 0.5
                out[:, k] = x; x2 = x1; x1 = x
        return _norm(out)
    if kind == "S1":                      # ONE number: the dominant line, random phase per episode
        w = 2 * np.pi * F_PEAK / dt_map
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        return _norm(np.cos(w * t[None, :] + ph))
    if kind == "BP":                      # dominant line +- 1 bin, flat within the band
        band = FREQS[(FREQS >= F_PEAK - DF) & (FREQS <= F_PEAK + DF)]
        w = 2 * np.pi * band / dt_map; out = np.zeros((n, NST))
        for i in range(n):
            ph = rng.uniform(0, 2 * np.pi, len(band))
            out[i] = np.cos(np.outer(w, t) + ph[:, None]).sum(0)
        return _norm(out)
    if kind == "Q":                       # the full measured spectrum
        w = 2 * np.pi * FREQS / dt_map; out = np.zeros((n, NST))
        for i in range(n):
            ph = rng.uniform(0, 2 * np.pi, len(FREQS))
            out[i] = (np.sqrt(POWER)[:, None] * np.cos(np.outer(w, t) + ph[:, None])).sum(0)
        return _norm(out)
    raise ValueError(kind)

def safe_rate(dist, A, seed):
    rng = np.random.default_rng(seed); n = dist.shape[0]
    starts = rng.uniform(-0.4, 0.4, (n, 2)) * P.REG
    v_ep = P.V_CAR * (1 + 0.2 * (rng.random(n) - 0.5) * 2)
    x = starts.copy(); alive = np.ones(n, bool)
    for t in range(NST):
        dx = np.stack([v_ep * np.sin(x[:, 1]),
                       (v_ep / P.L_WB) * np.tan(np.clip(-P.K1 * x[:, 0] - P.K2 * x[:, 1], -P.DMAX, P.DMAX))], axis=1)
        dx[:, 1] += A * dist[:, t]
        x = x + DT * dx
        if t % 50 == 25: x[alive] += rng.normal(0, [0.08, 0.015], (alive.sum(), 2))
        alive &= (np.abs(x[:, 0]) <= P.REG[0]) & (np.abs(x[:, 1]) <= P.REG[1])
    return float(alive.mean())

N = 400; AMPS = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]
KINDS = ["W", "M", "AR2", "S1", "BP", "Q"]
res = {k: {} for k in KINDS}
for k in KINDS:
    for seed in (21, 22, 23):
        rng = np.random.default_rng(1000 + seed)
        d = synth(k, N, 1.0, rng)
        for A in AMPS:
            res[k].setdefault(A, []).append(safe_rate(d, A, seed))
    print(f"{k:4s} " + "  ".join(f"{A}:{np.mean(res[k][A])*100:5.1f}" for A in AMPS), flush=True)

def boundary(curve, target=0.956):
    xs = sorted(curve); ys = [np.mean(curve[a]) for a in xs]
    for i in range(len(xs) - 1):
        if ys[i] >= target >= ys[i + 1]:
            f = (ys[i] - target) / max(ys[i] - ys[i + 1], 1e-9)
            return xs[i] + f * (xs[i + 1] - xs[i])
    return None

print("\ncertified boundary (amplitude where the safe rate crosses 95.6%):")
bq = boundary(res["Q"])
out = {}
def rate_at(curve_seeds, A):
    """per-seed safe rate at amplitude A by linear interpolation on the grid"""
    xs = sorted(curve_seeds); vals = []
    for sd in range(3):
        ys = [curve_seeds[a][sd] for a in xs]; vals.append(float(np.interp(A, xs, ys)))
    return vals
for k in KINDS:
    b = boundary(res[k]); out[k] = dict(boundary=b, curve={str(a): float(np.mean(res[k][a])) for a in AMPS},
                                       curve_seeds={str(a): [float(v) for v in res[k][a]] for a in AMPS})
    # OUTCOME-2 METRIC: episodes ending safely (under the MEASURED class) at the operating point this surrogate certifies
    if b is not None:
        tr = rate_at(res["Q"], b); out[k]["true_safe_rate_at_certified_boundary"] = dict(mean=float(np.mean(tr)), sd=float(np.std(tr)), seeds=tr)
    else:   # no crossing on the grid: white noise certifies everything -> operating point = grid max; AR(1) certifies nothing -> None
        top = max(AMPS); tr = rate_at(res["Q"], top) if np.mean(res[k][top]) >= 0.956 else None
        out[k]["true_safe_rate_at_certified_boundary"] = (dict(mean=float(np.mean(tr)), sd=float(np.std(tr)), seeds=tr, note=f"certifies every amplitude on the grid; evaluated at {top} rad/s") if tr else dict(note="certifies no amplitude on the grid"))
    tag = "REFERENCE" if k == "Q" else ("reproduces Q" if (b and bq and abs(b - bq) <= 0.05) else "misses Q")
    print(f"  {k:4s} boundary {('%.3f' % b) if b else '  none':>7s} rad/s    {tag}")
out["_meta"] = dict(f_peak_cyc_per_step=F_PEAK, n_bins=len(FREQS), N_episodes=N, seeds=3, target=0.956, Q_boundary=bq)
json.dump(out, open(os.path.join(HERE, "floor_vw_surrogates_seeds.json"), "w"), indent=1)
print("\n-> floor_vw_surrogates.json")
