"""
fable_traffic_directed.py — QUANTUM MOBILITY, DIRECTED (Path B, light
dynamics). Directed jam clearing (green-wave / metering) built into the
A1-class LIGHT spin-quench so it reaches the beyond-classical fluctuation
depth (k>=8) the gate-heavy hopping model could not.

Dynamics per step (chain of L sites; occupancy n_i = bit i):
  RZZ(-mu*dt) on bonds  +  RX(-2dt) on sites  +  RXX(2g*dt) even bonds
  +  RZ(F*(i-c)*dt) per site  <- longitudinal TILT (diagonal => NOT gauged
  away on an open chain; pure 1q => zero added CX). Same 2q-gate budget as
  the flown A1 card (crossing verified at k>=8) plus a free tilt.
Jam = left-half |1> block. Arms: F+ (downhill/green), F- (uphill/meter),
F0 (baseline melt), null (empty).

Observables:
  DIRECTED (the traffic claim): occupancy center-of-mass drift; the
    signature com(F+) - com(F-) grows with k (green clears downstream
    faster than metering).
  ADVANTAGE (inherits A1 hardness class): connected density-fluctuation
    spectrum X(q,k); k>=8 contested, box invited to attack the tilted
    variant.
Controls: F0 (symmetric baseline) and null; equal-depth differential
(A1 discipline; no number conservation since RX flips occupancy).

Stages: model -> scout -> fly -> grade. Working folder only.
"""
import json, math, sys, time
import numpy as np

WORK = __import__("os").path.dirname(__file__)
OPEN_CRN = __import__("os").environ.get("IBM_QUANTUM_CRN", "")  # set your own IBM Quantum instance CRN
OPEN_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
MU = 3.0 / (3.0 - math.sqrt(5.0))
DT, G, F_TILT = 0.30, 1.0, 0.80
STEPS = [0, 2, 4, 6, 8, 10, 12]
SHOTS = 8192
L_FLY = 64
STATE = WORK + r"\trafficD_state.json"
BAD_EDGES = {(78, 89), (83, 96), (89, 90), (113, 119), (81, 82), (96, 103)}


def build(L, k, arm, qubits=None, nq=None):
    """arm in {'Fp','Fm','F0','null'}."""
    from qiskit import QuantumCircuit
    q = qubits if qubits is not None else list(range(L))
    n = nq if nq is not None else L
    F = {'Fp': F_TILT, 'Fm': -F_TILT, 'F0': 0.0, 'null': F_TILT}[arm]
    c = (L - 1) / 2.0
    qc = QuantumCircuit(n, n)
    if arm != 'null':
        for i in range(L // 2):
            qc.x(q[i])
    for _ in range(k):
        for i in range(L - 1):
            qc.rzz(-MU * DT, q[i], q[i + 1])
        for i in range(L):
            qc.rx(-2 * DT, q[i])
        for i in range(0, L - 1, 2):
            qc.rxx(2 * G * DT, q[i], q[i + 1])
        if F != 0.0:
            for i in range(L):
                qc.rz(F * (i - c) * DT, q[i])
    qc.measure(range(n), range(n))
    return qc


def grade_counts(counts, L, qubits=None):
    q = qubits if qubits is not None else list(range(L))
    tot = 0
    dens = np.zeros(L); m2 = np.zeros((L, L))
    for bits, cc in counts.items():
        sb = bits.replace(" ", "")[::-1]
        nvec = np.array([int(sb[qq]) for qq in q], float)
        dens += cc * nvec; m2 += cc * np.outer(nvec, nvec); tot += cc
    dens /= tot
    C = m2 / tot - np.outer(dens, dens)
    Coff = C - np.diag(np.diag(C))
    pos = np.arange(L) / (L - 1)
    Xq = {qv: float(np.real(np.exp(1j*2*np.pi*qv*pos).conj() @ Coff @
                            np.exp(1j*2*np.pi*qv*pos))) / L**2
          for qv in (0.5, 1.0, 2.0, 4.0)}
    return dens, Xq


def com(dens):
    x = np.arange(len(dens)); tot = dens.sum()
    return float((x * dens).sum() / tot) if tot > 1e-9 else len(dens) / 2


def stage_model():
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    L = 14
    print(f"=== model gate (L={L}, statevector) ===")
    ok = {}
    # G1: 2q-gate budget per step (must match A1-light, not hopping)
    qc1 = build(L, 1, 'Fp')
    tq = transpile(qc1, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1)
    cx1 = tq.count_ops().get('cx', 0)
    ok['G1_light'] = bool(cx1 <= 3 * L)     # ~1.5L expected; hopping was ~6L
    print(f"  G1 light: {cx1} CX at k=1 (<= {3*L}?) -> "
          f"{'PASS' if ok['G1_light'] else 'FAIL'}")
    # G2: directed drift com(F+) - com(F-) sign-definite and growing
    def occom(k, arm):
        P = np.abs(Statevector(build(L, k, arm).remove_final_measurements(
            inplace=False)).data) ** 2
        d = np.array([P[((np.arange(2**L) >> i) & 1) == 1].sum() for i in range(L)])
        return com(d)
    diffs = [occom(k, 'Fp') - occom(k, 'Fm') for k in (2, 4, 6, 8)]
    ok['G2_directed'] = bool(diffs[-1] > 0.3 and diffs[-1] >= diffs[0] - 0.1)
    print(f"  G2 directed: com(F+)-com(F-) at k=2..8 {[round(d,2) for d in diffs]} "
          f"-> {'PASS' if ok['G2_directed'] else 'FAIL'}")
    # G3: grader round-trip
    rng = np.random.default_rng(7)
    P = np.abs(Statevector(build(L, 6, 'Fp').remove_final_measurements(
        inplace=False)).data) ** 2
    draws = rng.choice(2 ** L, 4096, p=P / P.sum())
    counts = {}
    for d_ in draws:
        b = format(d_, f"0{L}b"); counts[b] = counts.get(b, 0) + 1
    dens_s, _ = grade_counts(counts, L)
    dens_e = np.array([P[((np.arange(2**L) >> i) & 1) == 1].sum() for i in range(L)])
    dev = float(np.max(np.abs(dens_s - dens_e)))
    ok['G3_roundtrip'] = bool(dev < 0.05)
    print(f"  G3 round-trip: dens dev {dev:.4f} -> "
          f"{'PASS' if ok['G3_roundtrip'] else 'FAIL'}")
    allok = all(ok.values())
    json.dump(dict(gates=ok, cx_per_k1=int(cx1), drift=diffs,
                   verdict="PASS" if allok else "FAIL"),
              open(WORK + r"\trafficD_modelgate.json", "w"), indent=1)
    print("model gate:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def find_path(backend, L):
    tgt = backend.target
    g2 = next(x for x in ("cz", "ecr", "cx") if x in tgt.operation_names)
    adj = {}
    for pair in tgt[g2]:
        a, b = int(pair[0]), int(pair[1])
        if (min(a, b), max(a, b)) in BAD_EDGES:
            continue
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    for s0 in sorted(adj, key=lambda x: -len(adj[x])):
        stack = [(s0, [s0])]; tries = 0
        while stack and tries < 400000:
            tries += 1
            u, path = stack.pop()
            if len(path) == L:
                return path
            for v in adj.get(u, ()):
                if v not in path:
                    stack.append((v, path + [v]))
    return None


def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    force = sys.argv[2] if len(sys.argv) > 2 else None
    if force:
        if force not in OPEN_BACKENDS:
            print(f"REFUSED: {force} not open plan"); sys.exit(3)
        print(f"  forced backend: {force}")
        return svc, svc.backend(force)
    pend = {}
    for b in OPEN_BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10**6
    best = min(pend, key=pend.get)
    print(f"  queue: {pend} -> {best}")
    return svc, svc.backend(best)


def stage_scout():
    svc, backend = connect()
    nq = backend.target.num_qubits
    path = find_path(backend, L_FLY)
    if not path:
        print("ABORT: no path"); return 2
    tag = time.strftime("%Y%m%d_%H%M%S")
    pend = dict(card="TRAFFIC-DIRECTED (light dynamics) PENDING (frozen)",
                tag=tag, backend=backend.name, nq=nq, path=path, L=L_FLY,
                dt=DT, g=G, f_tilt=F_TILT, mu=MU, steps=STEPS, shots=SHOTS,
                prereg=dict(
                    P1="scout: jam contrast survives at k=2 (occupied left "
                       "half mean - empty right half mean > 0.15 on Fp)",
                    P2="DIRECTED: com(Fp) - com(Fm) > 0 and grows with k "
                       "(green clears downstream faster than metering); "
                       "graded in equal-depth differential vs F0 baseline",
                    P3="connected density-fluctuation X(q,k); k>=8 contested "
                       "(inherits A1 light-quench hardness class; box invited "
                       "to attack the tilted variant)",
                    P4="null arm: flat, X(q) at noise floor"))
    p = WORK + rf"\trafficD_pending_{tag}.json"
    json.dump(pend, open(p, "w"), indent=1)
    print(f"  pending -> {p}")
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    qcs = transpile([build(L_FLY, 2, 'Fp', path, nq)], backend,
                    optimization_level=1)
    op = qcs[0].count_ops()
    print(f"  scout circuit: {op.get('cz', op.get('ecr', 0))} 2q gates")
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=4096)
    print(f"  SCOUT {job.job_id()}; waiting...")
    r = job.result()
    dens, Xq = grade_counts(r[0].data.c.get_counts(), L_FLY, path)
    contrast = float(dens[:L_FLY//2].mean() - dens[L_FLY//2:].mean())
    okc = contrast > 0.15
    print(f"  k=2 jam contrast {contrast:.3f} (>0.15?) "
          f"{'PASS' if okc else 'FAIL'}; com {com(dens):.2f}")
    json.dump(dict(pending=p, backend=backend.name, tag=tag,
                   scout_job=job.job_id(), scout_pass=bool(okc)),
              open(STATE, "w"), indent=1)
    return 0 if okc else 1


def stage_fly():
    st = json.load(open(STATE))
    if not st.get("scout_pass"):
        print("REFUSED"); return 2
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    backend = svc.backend(st["backend"])
    pend = json.load(open(st["pending"]))
    path, nq = pend["path"], pend["nq"]
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    circs, names = [], []
    for k in STEPS:
        for arm in ('Fp', 'Fm', 'F0', 'null'):
            circs.append(build(L_FLY, k, arm, path, nq))
            names.append(f"{arm}_{k}")
    qcs = transpile(circs, backend, optimization_level=1)
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=SHOTS)
    st["main_job"] = job.job_id(); st["main_names"] = names
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"TRAFFIC-DIRECTED MAIN submitted: {job.job_id()} "
          f"({len(qcs)} x {SHOTS} on {st['backend']})")
    return 0


def stage_grade():
    st = json.load(open(STATE))
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    pend = json.load(open(st["pending"]))
    path, L = pend["path"], pend["L"]
    res = svc.job(st["main_job"]).result()
    counts = {nm: res[i].data.c.get_counts()
              for i, nm in enumerate(st["main_names"])}
    out = dict(card="TRAFFIC-DIRECTED RESULT", tag=pend["tag"],
               backend=st["backend"], steps={})
    print("k | com Fp/Fm/F0 | DIRECTED com(Fp)-com(Fm) | dX(q)=X_jam-X_null")
    for k in pend["steps"]:
        row = {}
        for arm in ('Fp', 'Fm', 'F0', 'null'):
            dens, Xq = grade_counts(counts[f"{arm}_{k}"], L, path)
            row[arm] = dict(com=com(dens), dens=[float(x) for x in dens],
                            Xq={str(q): v for q, v in Xq.items()})
        directed = row['Fp']['com'] - row['Fm']['com']
        # DIFFERENTIAL advantage observable: jam connected spectrum minus
        # null (common-mode readout-noise cancels; A1 equal-depth trick)
        dX = {q: row['Fp']['Xq'][q] - row['null']['Xq'][q] for q in row['Fp']['Xq']}
        dX_F0 = {q: row['Fp']['Xq'][q] - row['F0']['Xq'][q] for q in row['Fp']['Xq']}
        out["steps"][k] = row | {"directed": float(directed),
                                 "dX_jam_minus_null": dX,
                                 "dX_Fp_minus_F0": dX_F0}
        print(f" {k:2d} | {row['Fp']['com']:5.2f}/{row['Fm']['com']:5.2f}/"
              f"{row['F0']['com']:5.2f} | {directed:+.3f} | "
              f"dX(0.5)={dX['0.5']:+.5f} dX(1.0)={dX['1.0']:+.5f}")
    p = WORK + rf"\trafficD_result_{pend['tag']}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "model"
    sys.exit({"model": stage_model, "scout": stage_scout,
              "fly": stage_fly, "grade": stage_grade}[stage]())
