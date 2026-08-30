# -*- coding: utf-8 -*-
"""
vw_cert_gap_gate.py — MODEL GATE for the certification-gap product.

Question the gate answers before any claim: does the hardware-computed
collective disturbance spectrum expose failures that heuristic-validated
envelopes call safe, at MATCHED disturbance power?

Ensembles (identical rms power, only the SPECTRUM differs):
  W — white noise            (what naive validation uses)
  M — AR(1), variance + lag-1 autocorrelation matched to Q
      (the best cheap surrogate a classical pipeline would fit)
  Q — hardware spectrum: colored noise with the flown S(q,omega) power
      (dominant collective line 0.0625 cycles/step, fez receipt)

Plant: the scored safety track's kinematic-bicycle lane-keeping under the
certified regime; disturbance enters as heading-rate bias (the track's
measured collapse axis: safe to 1.0 rad/s, collapse inside [1.2, 1.6]).

Sweep: rms amplitude A around the certified boundary x timescale mapping
dt_map (hardware step -> seconds), because the step->s mapping is a
modeling choice and the gap must be robust across it, not tuned.

GATE: at some (A, dt_map) with A at/below the level where W and M grade
>= 95% safe, Q's safe rate is lower by >= 10 points, 3 seeds.
"""
import json, math, sys
import numpy as np

sys.path.insert(0, r"C:\quantum ai 2026\VW_SUBMISSION_PACKAGE")
import fable_vw_lyap as P

P.set_regime("hard") if "hard" in P.REGIMES else P.set_regime(list(P.REGIMES)[0])
DT = P.DT_EP
T_EP = 8.0
NST = int(T_EP / DT)

spec = json.load(open(
    r"C:\quantum ai 2026\VW_SUBMISSION_PACKAGE\results\vw_jam_relaxation_spectrum.json"))
SQ = spec["spectra"]["0.5"]
FREQS = np.array(SQ["freqs"][1:])          # cycles per hardware step (drop DC)
POWER = np.array(SQ["power"][1:])
POWER = POWER / POWER.sum()


def synth_Q(n, nst, dt_map, rng):
    """colored noise with the hardware spectrum; unit rms."""
    t = np.arange(nst) * DT
    out = np.zeros((n, nst))
    for i in range(n):
        ph = rng.uniform(0, 2 * np.pi, len(FREQS))
        w = 2 * np.pi * FREQS / dt_map          # rad/s
        out[i] = (np.sqrt(POWER)[:, None] *
                  np.cos(np.outer(w, t) + ph[:, None])).sum(0)
    out /= out.std(axis=1, keepdims=True) + 1e-12
    return out


def synth_M(n, nst, rho_dt, rng):
    """AR(1) with lag-DT autocorrelation rho_dt; unit rms."""
    a = rho_dt
    out = np.zeros((n, nst))
    x = rng.normal(size=n)
    for t in range(nst):
        x = a * x + math.sqrt(max(1 - a * a, 1e-9)) * rng.normal(size=n)
        out[:, t] = x
    out /= out.std(axis=1, keepdims=True) + 1e-12
    return out


def synth_W(n, nst, rng):
    out = rng.normal(size=(n, nst))
    out /= out.std(axis=1, keepdims=True) + 1e-12
    return out


def safe_rate(dist, A, seed):
    """episodes from certified-interior starts under heading-rate bias
    d(t) = A * dist realization; impulses as in the scored suite."""
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    starts = rng.uniform(-0.4, 0.4, (n, 2)) * P.REG
    v_ep = P.V_CAR * (1 + 0.2 * (rng.random(n) - 0.5) * 2)
    x = starts.copy()
    alive = np.ones(n, bool)
    for t in range(NST):
        dx = np.stack([v_ep * np.sin(x[:, 1]),
                       (v_ep / P.L_WB) * np.tan(
                           np.clip(-P.K1 * x[:, 0] - P.K2 * x[:, 1],
                                   -P.DMAX, P.DMAX))], axis=1)
        dx[:, 1] += A * dist[:, t]
        x = x + DT * dx
        if t % 50 == 25:
            x[alive] += rng.normal(0, [0.08, 0.015], (alive.sum(), 2))
        alive &= (np.abs(x[:, 0]) <= P.REG[0]) & (np.abs(x[:, 1]) <= P.REG[1])
    return alive.mean()


def main():
    N = 1000 if len(sys.argv) > 1 and sys.argv[1] == "score" else 400
    print(f"plant: V={P.V_CAR}, REG={P.REG}, collapse axis ~[1.2,1.6] rad/s")
    print("\ndt_map | A_rms | W safe | M safe | Q safe | gap(minWM - Q)")
    best = None
    allrows = []
    score = len(sys.argv) > 1 and sys.argv[1] == 'score'
    for dt_map in ((1.0, 2.0) if score else (0.5, 1.0, 2.0, 4.0)):
        # Q first (to match M's autocorrelation to it)
        rng = np.random.default_rng(11)
        q0 = synth_Q(64, NST, dt_map, rng)
        rho = float(np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in q0]))
        for A in ((0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3) if score else (0.6, 0.8, 1.0, 1.2)):
            rows = []
            for seed in (21, 22, 23):
                rng = np.random.default_rng(seed)
                sw = safe_rate(synth_W(N, NST, rng), A, seed)
                sm = safe_rate(synth_M(N, NST, rho, rng), A, seed)
                sq = safe_rate(synth_Q(N, NST, dt_map, rng), A, seed)
                rows.append((sw, sm, sq))
            sw, sm, sq = (np.mean([r[i] for r in rows]) for i in range(3))
            dw, dm, dq = (np.std([r[i] for r in rows]) for i in range(3))
            allrows.append(dict(dt_map=dt_map, A=A,
                W=[float(sw), float(dw)], M=[float(sm), float(dm)],
                Q=[float(sq), float(dq)]))
            gap = min(sw, sm) - sq
            flag = ""
            if min(sw, sm) >= 0.95 and gap >= 0.10:
                flag = "  <-- GATE CANDIDATE"
                if best is None or gap > best[-1]:
                    best = (dt_map, A, sw, sm, sq, gap)
            print(f"  {dt_map:4.1f} | {A:4.2f} | {100*sw:5.1f}%±{100*dw:3.1f} | "
                  f"{100*sm:5.1f}%±{100*dm:3.1f} | {100*sq:5.1f}%±{100*dq:3.1f} | "
                  f"{100*gap:+5.1f} pts{flag}", flush=True)
    if best:
        dt_map, A, sw, sm, sq, gap = best
        print(f"\nGATE PASS: at dt_map={dt_map}s/step, A={A} rad/s rms — "
              f"heuristic validation grades >={100*min(sw,sm):.0f}% safe while "
              f"the hardware-spectrum ensemble exposes {100*gap:.0f} points of "
              f"hidden failure.")
    else:
        print("\nGATE FAIL: no (A, dt_map) where heuristics pass and Q exposes "
              ">=10 pts — the spectrum does not move this plant. Honest null.")
    json.dump(dict(card="CERT-GAP scored sweep: certified boundary under the "
                        "measured collective disturbance spectrum",
                   source_spectrum="vw_jam_relaxation_spectrum.json "
                        "(fez d9rm4j9dsedc73agrb70)",
                   seeds=[21, 22, 23], episodes_per_cell=N,
                   rows=allrows),
              open(r"C:\quantum ai 2026\vw_cert_gap_score.json", "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
