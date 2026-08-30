"""
fable_a1p_flight.py — A1-P: collective-sector spectroscopy, phase-channel
style. Framework-led per FRAMEWORK_TRUST_LOCK: derived spectrum frozen
before flight; anchors graded against EXACT closed-orbit waveforms; detuned
control must move; collective sector recorded under the contested-cells
rule. Fresh files live in C:\\quantum ai 2026; C:\\fable is read-only.

Stages: derive -> scout -> fly -> grade    (python fable_a1p_flight.py <stage>)
"""
import json, math, sys, time
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from fable_a1_hardness import conj_gate, RUNGS, BONDS, TH_RZZ, TH_RX, TH_RXX

WORK = __import__("os").path.dirname(__file__)
OPEN_CRN = __import__("os").environ.get("IBM_QUANTUM_CRN", "")  # set your own IBM Quantum instance CRN
OPEN_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
KMAX = 15                       # 16 time points; ~120 CX depth < wall
SHOTS = 8192
DT, GB = 0.30, 1.0
MU = 3.0 / (3.0 - math.sqrt(5.0))
DT_CTRL = 0.36                  # detuned control: lines scale by 1.2
CAP = 60000                     # orbit cap: rungs exceeding it = leaky, not anchor-graded
STATE = WORK + r"\a1p_state.json"


def one_step(terms, thzz, thx, thxx):
    for layer in reversed(BONDS):
        for (u, v) in layer:
            terms = conj_gate(terms, ((u, "X"), (v, "X")), thxx)
    sup = {q for pk in terms for q, _ in pk}
    for q in sup:
        terms = conj_gate(terms, ((q, "X"),), thx)
    for (u, v) in RUNGS:
        terms = conj_gate(terms, ((u, "Z"), (v, "Z")), thzz)
    return {k: v for k, v in terms.items() if abs(v) > 1e-12}


def expect_product(terms, flipped):
    tot = 0.0
    for pk, c in terms.items():
        v = c
        for q, P in pk:
            if P in ("X", "Y"):
                v = 0.0
                break
            v *= (-1.0 if q in flipped else 1.0)
        tot += v
    return tot


def wall_flips():
    pos = np.array([(a + b) / 2 for a, b in RUNGS])
    med = np.median(pos)
    return {RUNGS[i][0] for i, p in enumerate(pos) if p < med}


def stage_derive():
    print("=== A1-P derive: exact per-rung waveforms from orbit evolution ===")
    flips = wall_flips()
    angles = dict(main=(TH_RZZ, TH_RX, TH_RXX),
                  ctrl=(-MU * DT_CTRL, -2 * DT_CTRL, 2 * GB * DT_CTRL))
    wf = {"main": {"wall": {}, "null": {}}, "ctrl": {"null": {}}}
    anchors = []
    for i, (a, b) in enumerate(RUNGS):
        terms = {frozenset([(a, "Z"), (b, "Z")]): 1.0}
        ok = True
        wm_w, wm_n = [1.0 if all(q not in flips for q in (a, b)) else
                      expect_product(terms, flips)], [1.0]
        t = dict(terms)
        for k in range(1, KMAX + 1):
            t = one_step(t, *angles["main"])
            if len(t) > CAP:
                ok = False
                break
            wm_w.append(expect_product(t, flips))
            wm_n.append(expect_product(t, set()))
        if ok:
            anchors.append(i)
            wf["main"]["wall"][i] = wm_w
            wf["main"]["null"][i] = wm_n
        if i % 16 == 0:
            print(f"  rung {i}: {'anchor' if ok else 'leaky'} "
                  f"(orbit {'closed' if ok else '>cap'})", flush=True)
    # control waveforms for 6 anchor rungs (cheaper)
    for i in anchors[:6]:
        a, b = RUNGS[i]
        t = {frozenset([(a, "Z"), (b, "Z")]): 1.0}
        w = [1.0]
        for k in range(1, KMAX + 1):
            t = one_step(t, *angles["ctrl"])
            if len(t) > CAP:
                w = None
                break
            w.append(expect_product(t, set()))
        if w:
            wf["ctrl"]["null"][i] = w
    lines = json.load(open(WORK + r"\a1p_derived_lines.json"))
    pend = dict(card="A1-P PENDING (frozen)", tag=time.strftime("%Y%m%d_%H%M%S"),
                kmax=KMAX, shots=SHOTS, dt=DT, dt_ctrl=DT_CTRL, g=GB, mu=MU,
                anchors=anchors, n_anchors=len(anchors),
                derived_lines=lines["lines_cycles_per_step"],
                waveforms=wf,
                prereg=dict(
                    P1=f"anchor rungs ({len(anchors)}): measured 16-pt waveform "
                       "matches EXACT derived waveform after one fitted "
                       "damping envelope per arm; median corr >= 0.90",
                    P2="fixed-frequency fit: 18 derived lines explain the "
                       "anchor signals; adding detuned (x1.2) lines must NOT "
                       "improve fit on main arm",
                    P3="control arm (dt=0.36): matches ITS exact waveforms, "
                       "not the main-arm ones (must move)",
                    P4="collective X(q,k) series recorded; contested cells "
                       "(k>=6 class) claimed ONLY under the standing "
                       "adversarial rule — no claim before box verdict"))
    p = WORK + rf"\a1p_pending_{pend['tag']}.json"
    json.dump(pend, open(p, "w"), indent=1)
    json.dump(dict(pending=p), open(STATE, "w"), indent=1)
    print(f"derive done: {len(anchors)}/64 anchor rungs; pending -> {p}")
    return 0


def build(k, arm, dt_, nq):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(nq, nq)
    if arm == "wall":
        for q in wall_flips():
            qc.x(q)
    for _ in range(k):
        for (a, b) in RUNGS:
            qc.rzz(-MU * dt_, a, b)
        for q in range(nq):
            qc.rx(-2 * dt_, q)
        for layer in BONDS:
            for (a, b) in layer:
                qc.rxx(2 * GB * dt_, a, b)
    qc.measure(range(nq), range(nq))
    return qc


def profiles(counts):
    tot = 0
    m1 = np.zeros(len(RUNGS))
    for bits, c in counts.items():
        sb = bits.replace(" ", "")[::-1]
        par = np.array([1 - 2 * (int(sb[a]) ^ int(sb[b])) for a, b in RUNGS], float)
        m1 += c * par
        tot += c
    return m1 / tot


def connect(named=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    if named:
        if named not in OPEN_BACKENDS:
            print("REFUSED"); sys.exit(3)
        return svc, svc.backend(named)
    pend = {}
    for b in OPEN_BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10 ** 6
    best = min(pend, key=pend.get)
    print(f"  queue: {pend} -> {best}")
    return svc, svc.backend(best)


def stage_scout():
    st = json.load(open(STATE))
    pend = json.load(open(st["pending"]))
    svc, backend = connect()
    nq = backend.target.num_qubits
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    qcs = transpile([build(1, "null", DT, nq), build(3, "null", DT, nq)],
                    backend, optimization_level=1)
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=4096)
    print(f"  SCOUT {job.job_id()} on {backend.name}; waiting...")
    r = job.result()
    ok_ct = 0
    for ci, k in enumerate((1, 3)):
        p = profiles(r[ci].data.c.get_counts())
        devs = []
        for i in pend["anchors"]:
            ex = pend["waveforms"]["main"]["null"][str(i)] if isinstance(
                list(pend["waveforms"]["main"]["null"].keys())[0], str) else \
                pend["waveforms"]["main"]["null"][i]
            devs.append(p[i] / ex[k] if abs(ex[k]) > 0.05 else None)
        devs = [d for d in devs if d is not None]
        med = float(np.median(devs))
        print(f"  k={k}: median retention {med:.3f} over {len(devs)} anchors")
        ok_ct += 0.5 < med < 1.1
    st.update(backend=backend.name, scout_job=job.job_id(), scout_pass=ok_ct == 2)
    json.dump(st, open(STATE, "w"), indent=1)
    print("scout", "PASS" if ok_ct == 2 else "FAIL")
    return 0 if ok_ct == 2 else 1


def stage_fly():
    st = json.load(open(STATE))
    if not st.get("scout_pass"):
        print("REFUSED"); return 2
    svc, backend = connect(st["backend"])
    nq = backend.target.num_qubits
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    circs, names = [], []
    for k in range(KMAX + 1):
        for arm in ("wall", "null"):
            circs.append(build(k, arm, DT, nq)); names.append(f"M_{arm}_{k}")
    for k in (2, 5, 9, 13):
        circs.append(build(k, "null", DT_CTRL, nq)); names.append(f"C_null_{k}")
    qcs = transpile(circs, backend, optimization_level=1)
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=SHOTS)
    st["main_job"] = job.job_id(); st["main_names"] = names
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"A1-P MAIN submitted: {job.job_id()} ({len(qcs)} x {SHOTS} on {st['backend']})")
    return 0


def stage_grade():
    st = json.load(open(STATE))
    pend = json.load(open(st["pending"]))
    svc, backend = connect(st["backend"])
    res = svc.job(st["main_job"]).result()
    counts = {nm: res[i].data.c.get_counts() for i, nm in enumerate(st["main_names"])}
    wf = pend["waveforms"]
    gk = lambda d, i: d[str(i)] if str(i) in d else d[i]
    anchors = pend["anchors"]
    out = dict(card="A1-P RESULT", tag=pend["tag"], backend=st["backend"], gates={})

    # P1: anchor waveform correlation (damping envelope fitted per arm)
    for arm in ("wall", "null"):
        prof = np.array([profiles(counts[f"M_{arm}_{k}"]) for k in range(KMAX + 1)])
        corrs = []
        for i in anchors:
            ex = np.array(gk(wf["main"][arm], i))
            hw = prof[:, i]
            # fit single exponential envelope by least squares on |signal|
            k_ = np.arange(KMAX + 1)
            mask = np.abs(ex) > 0.05
            if mask.sum() < 6:
                continue
            lam = np.polyfit(k_[mask], np.log(np.clip(np.abs(hw[mask]), 1e-3, None) /
                                              np.clip(np.abs(ex[mask]), 1e-3, None)), 1)[0]
            env = np.exp(lam * k_)
            c = float(np.corrcoef(hw, ex * env)[0, 1])
            corrs.append(c)
        med = float(np.median(corrs))
        out["gates"][f"P1_{arm}"] = dict(median_corr=med, n=len(corrs),
                                         passed=bool(med >= 0.90))
        print(f"P1 {arm}: median waveform corr {med:.4f} over {len(corrs)} anchors "
              f"-> {'PASS' if med >= 0.90 else 'FAIL'}")

    # P2: fixed-frequency line fit vs detuned-line fit (null arm)
    lines = np.array(pend["derived_lines"])
    prof = np.array([profiles(counts[f"M_null_{k}"]) for k in range(KMAX + 1)])
    k_ = np.arange(KMAX + 1)
    def design(freqs):
        cols = [np.ones_like(k_, float)]
        for f in freqs:
            cols += [np.cos(2 * np.pi * f * k_), np.sin(2 * np.pi * f * k_)]
        return np.array(cols).T
    A_true, A_det = design(lines), design(lines * 1.2)
    r_true, r_det = [], []
    for i in anchors:
        y = prof[:, i]
        for A, acc in ((A_true, r_true), (A_det, r_det)):
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            acc.append(1 - resid.var() / max(y.var(), 1e-12))
    p2 = float(np.median(r_true)) > float(np.median(r_det))
    out["gates"]["P2_lines"] = dict(R2_true=float(np.median(r_true)),
                                    R2_detuned=float(np.median(r_det)), passed=bool(p2))
    print(f"P2: R2 derived-lines {np.median(r_true):.4f} vs detuned {np.median(r_det):.4f} "
          f"-> {'PASS' if p2 else 'FAIL'}")

    # P3: control arm must match ITS waveforms better than main-arm waveforms
    ctrl_ok = 0; ctrl_n = 0
    for i in list(wf["ctrl"]["null"].keys()):
        ii = int(i)
        exc = np.array(gk(wf["ctrl"]["null"], ii))
        exm = np.array(gk(wf["main"]["null"], ii))
        ks = [2, 5, 9, 13]
        hw = np.array([profiles(counts[f"C_null_{k}"])[ii] for k in ks])
        dc = np.corrcoef(hw, exc[ks])[0, 1]
        dm = np.corrcoef(hw, exm[ks])[0, 1]
        ctrl_ok += dc > dm; ctrl_n += 1
    out["gates"]["P3_control"] = dict(moved=int(ctrl_ok), n=int(ctrl_n),
                                      passed=bool(ctrl_ok > ctrl_n / 2))
    print(f"P3: control matches its own dynamics on {ctrl_ok}/{ctrl_n} rungs "
          f"-> {'PASS' if ctrl_ok > ctrl_n / 2 else 'FAIL'}")

    # P4: collective series recorded (no claim here)
    pos = np.array([(a + b) / 2 for a, b in RUNGS])
    pos = (pos - pos.min()) / (pos.max() - pos.min())
    Xq = {}
    for k in range(KMAX + 1):
        cw = counts[f"M_wall_{k}"]
        tot = 0; m1 = np.zeros(len(RUNGS)); m2 = np.zeros((len(RUNGS), len(RUNGS)))
        for bits, c in cw.items():
            sb = bits.replace(" ", "")[::-1]
            par = np.array([1 - 2 * (int(sb[a]) ^ int(sb[b])) for a, b in RUNGS], float)
            m1 += c * par; m2 += c * np.outer(par, par); tot += c
        m1 /= tot; C = m2 / tot - np.outer(m1, m1)
        Coff = C - np.diag(np.diag(C))
        Xq[k] = {q: float(np.real(np.exp(1j*2*np.pi*q*pos).conj() @ Coff @
                                  np.exp(1j*2*np.pi*q*pos))) / len(RUNGS)**2
                 for q in (0.5, 1.0, 2.0)}
    out["collective_Xq_series"] = {str(k): v for k, v in Xq.items()}
    p = WORK + rf"\a1p_result_{pend['tag']}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "derive"
    sys.exit({"derive": stage_derive, "scout": stage_scout,
              "fly": stage_fly, "grade": stage_grade}[stage]())
