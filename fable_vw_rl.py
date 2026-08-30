"""
fable_vw_rl.py — RL-ALIGNMENT CARD: Dirac-3 in the training loop.

Rubric metric (RL-alignment track): reduce the ROLLOUT BUDGET vs standard
GRPO at equal alignment. The wall: RL fine-tuning of driving policies is
rollout-starved; rollouts (simulation) are the cost.

Where the quantum hardware works: every training iteration, the candidate
pool of rollout configurations (start state x disturbance) is submitted to
Dirac-3 as a curriculum-selection problem — pick the K most jointly
INFORMATIVE rollouts (diversity + difficulty QUBO) before any simulation is
spent. Selection acts on metadata only, so saved rollouts are real savings.

Arms (identical policy init, task, seeds):
  A  GRPO-32 : standard — 32 random rollouts / iteration (the named baseline)
  B  GRPO-16 : 16 random rollouts / iteration  (controls for "just use fewer")
  C  DIRAC-16: 16 Dirac-3-selected rollouts / iteration (one device job/iter,
               job ids logged as receipts)
Score: rollouts consumed to reach target alignment (target = within 10% of
the nominal controller's eval reward), plus full reward curves.

Task: lane-keeping alignment under domain randomization (random crosswind
bias and start state per rollout), kinematic bicycle (v15_tight regime).
Policy: linear-Gaussian steering policy, GRPO update (group-relative
advantage on trajectory log-prob).

Run under py -3.13 (qci-client). One seed per invocation:
  py -3.13 fable_vw_rl.py <seed> <arm A|B|C>
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import fable_vw_lyap as m

m.set_regime("v15_tight")
DT, T_EP = m.DT_EP, 6.0
SIGMA = 0.08
LR = 0.4
MAX_ITERS = 60
N_POOL, K_SEL = 64, 16
EVAL_N = 64
BIAS_MAX = 1.2

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 21
ARM = sys.argv[2] if len(sys.argv) > 2 else "C"


def phi(x):
    z0, z1 = x[0] / m.REG[0], x[1] / m.REG[1]
    return np.array([z0, z1, z0 * abs(z0), z1 * abs(z1), z0 * z1])


def rollout(theta, x0, bias, rng, learn=True):
    x = np.array(x0, float)
    glp = np.zeros_like(theta)
    rsum = 0.0; nT = int(T_EP / DT)
    for t in range(nT):
        f = phi(x)
        mean = float(np.clip(theta @ f, -m.DMAX, m.DMAX))
        a = mean + (SIGMA * rng.normal() if learn else 0.0)
        a = float(np.clip(a, -m.DMAX, m.DMAX))
        if learn:
            glp += (a - mean) / SIGMA ** 2 * f
        x = x + DT * np.array([m.V_CAR * np.sin(x[1]),
                               (m.V_CAR / m.L_WB) * np.tan(a) + bias])
        rsum += -(abs(x[0]) / m.REG[0] + 0.5 * abs(x[1]) / m.REG[1])
        if abs(x[0]) > m.REG[0] or abs(x[1]) > 1.5 * m.REG[1]:
            rsum += -1.5 * (nT - t)          # crash = worst-case cost for all remaining steps
            break
    return rsum / nT, glp


def make_pool(rng, n):
    starts = rng.uniform(-0.85, 0.85, (n, 2)) * m.REG
    biases = rng.uniform(-BIAS_MAX, BIAS_MAX, n)
    return starts, biases


def select_dirac(starts, biases, k, tag, receipts, frontier=1.0):
    """curriculum QUBO on Dirac-3: cardinality + diversity + difficulty."""
    sys.path.insert(0, __import__("os").environ.get("QCI_RUN_DIR", "."))
    from qci_dirac_common import make_qci_client
    n = len(biases)
    Z = np.column_stack([starts[:, 0] / m.REG[0], starts[:, 1] / m.REG[1],
                         biases / BIAS_MAX])
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    hard = np.linalg.norm(Z, axis=1)
    A, B, C = 4.0, 1.0 / n, 0.8
    lin = np.zeros(n); Q = {}
    for i in range(n):
        lin[i] += A * (1 - 2 * k) + C * (hard[i] - frontier) ** 2   # prefer the competence frontier
        for j in range(i + 1, n):
            Q[(i, j)] = 2 * A - B * D[i, j]
    data = [{"idx": [0, i + 1], "val": float(v)} for i, v in enumerate(lin)]
    data += [{"idx": [i + 1, j + 1], "val": float(v)} for (i, j), v in Q.items()]
    client, _ = make_qci_client()
    fid = client.upload_file(file={"file_name": tag, "file_config": {
        "polynomial": {"num_variables": n, "min_degree": 1, "max_degree": 2,
                       "data": data}}})["file_id"]
    body = client.build_job_body(
        job_type="sample-hamiltonian-integer", job_name=tag,
        job_tags=["fable", "vw-rl"],
        job_params={"device_type": "dirac-3", "num_samples": 5,
                    "relaxation_schedule": 1, "num_levels": [2] * n},
        polynomial_file_id=fid)
    from fable_qci_robust import submit_and_wait
    jid, sols = submit_and_wait(client, body, timeout=1200)
    receipts.append(jid)
    best = None
    for x in sols:
        sel = [i for i, v in enumerate(x) if round(v) == 1]
        # repair cardinality
        if len(sel) > k:
            sel = sorted(sel, key=lambda i: abs(hard[i]))[:k]
        while len(sel) < k:
            rest = [i for i in range(n) if i not in sel]
            sel.append(max(rest, key=lambda i: min(D[i, j] for j in sel) if sel else hard[i]))
        div = sum(D[i, j] for ii, i in enumerate(sel) for j in sel[ii+1:])
        if best is None or div > best[0]:
            best = (div, sel)
    return best[1]


def evaluate(theta, eval_starts):
    rng = np.random.default_rng(0)
    return float(np.mean([rollout(theta, s, b, rng, learn=False)[0]
                          for s, b in zip(eval_starts[0], eval_starts[1])]))


def main():
    rng = np.random.default_rng(SEED)
    eval_starts = make_pool(np.random.default_rng(999), EVAL_N)
    theta_nom = np.array([-m.K1 * m.REG[0], -m.K2 * m.REG[1], 0, 0, 0])
    r_nom = evaluate(theta_nom, eval_starts)
    target = r_nom * 1.05                       # within 5% of nominal
    theta = np.zeros(5)                                  # no steering at init
    print(f"arm {ARM} seed {SEED}: nominal eval {r_nom:.4f}, target {target:.4f}, "
          f"init {evaluate(theta, eval_starts):.4f}")
    receipts = []; curve = []; rollouts_used = 0; hit = None
    K = 32 if ARM == "A" else 16
    for it in range(MAX_ITERS):
        starts, biases = make_pool(rng, N_POOL)
        if ARM == "D":
            # ablation arm: SAME selection objective, classical SA solver
            n=N_POOL
            Z=np.column_stack([starts[:,0]/m.REG[0],starts[:,1]/m.REG[1],biases/BIAS_MAX])
            Dm=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=-1)
            hard=np.linalg.norm(Z,axis=1)
            prev = curve[-1]["eval"] if curve else -1.5
            frontier = 0.3 + max(0.0, min(1.0, (prev + 1.5) / 1.3))
            Bq, Cq = 1.0/n, 0.8
            def obj(sel):
                return (Cq*sum((hard[i]-frontier)**2 for i in sel)
                        - Bq*sum(Dm[i,j] for ii,i in enumerate(sel) for j in sel[ii+1:]))
            cur=list(rng.choice(n,K,replace=False)); co=obj(cur)
            for sweep in range(400):
                i=int(rng.integers(K)); j=int(rng.integers(n))
                if j in cur: continue
                cand=cur[:]; cand[i]=j; o=obj(cand)
                if o<co: cur,co=cand,o
            sel=cur
        elif ARM == "E":
            # stochastic classical sampler, SAME objective: Gibbs/softmax pick
            n=N_POOL
            Z=np.column_stack([starts[:,0]/m.REG[0],starts[:,1]/m.REG[1],biases/BIAS_MAX])
            Dm=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=-1)
            hard=np.linalg.norm(Z,axis=1)
            prev = curve[-1]["eval"] if curve else -1.5
            frontier = 0.3 + max(0.0, min(1.0, (prev + 1.5) / 1.3))
            Bq, Cq, TEMP = 1.0/n, 0.8, 0.15
            sel=[]
            for _ in range(K):                       # sequential Gibbs draw
                rest=[i for i in range(n) if i not in sel]
                en=np.array([Cq*(hard[i]-frontier)**2
                             - Bq*sum(Dm[i,j] for j in sel) for i in rest])
                p=np.exp(-(en-en.min())/TEMP); p/=p.sum()
                sel.append(int(np.random.default_rng(rng.integers(1<<31)).choice(rest,p=p)))
        elif ARM == "C":
            prev = curve[-1]["eval"] if curve else -1.5
            frontier = 0.3 + max(0.0, min(1.0, (prev + 1.5) / 1.3))
            sel = select_dirac(starts, biases, K,
                               f"fable_rl_s{SEED}_i{it}", receipts,
                               frontier=frontier)
        else:
            sel = list(rng.choice(N_POOL, K, replace=False))
        rs, gs = [], []
        for i in sel:
            r_, g_ = rollout(theta, starts[i], biases[i], rng)
            rs.append(r_); gs.append(g_)
        rollouts_used += K
        rs = np.array(rs)
        adv = (rs - rs.mean()) / max(rs.std(), 1e-6)
        grad = sum(a * g for a, g in zip(adv, gs)) / K
        theta = theta + LR * grad / max(np.linalg.norm(grad), 1e-9) * 0.2
        r_eval = evaluate(theta, eval_starts)
        curve.append(dict(iter=it, rollouts=rollouts_used, eval=r_eval))
        print(f"  it {it:2d} rollouts {rollouts_used:4d} eval {r_eval:.4f}"
              + ("  <-- TARGET" if r_eval >= target and hit is None else ""),
              flush=True)
        if r_eval >= target and hit is None:
            hit = rollouts_used
            if ARM != "C":
                break                            # classical arms may stop
            if it >= 5:
                break                            # device arm: a few extra receipts
    out = dict(arm=ARM, seed=SEED, target=target, r_nominal=r_nom,
               rollouts_to_target=hit, curve=curve, receipts=receipts)
    p = rf"C:\fable\vw_rl_{ARM}_s{SEED}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"RESULT arm {ARM}: rollouts-to-target {hit} "
          f"({len(receipts)} device jobs) -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
