# Merlin Digital — GIC 2026 Volkswagen Enterprise Track

| | |
|---|---|
| **Team** | Merlin Digital |
| **Project** | Quantum Advantage of Route for Mobility Response, with Quantum-in-the-Loop Training and Certified Runtime Assurance |
| **Tracks** | RL Alignment (primary scored) · Safety (secondary scored) · Context: Autonomous Driving / Mobility |
| **Write-up** | `VW_REPORT_v2.md` (advantage exhibit + two scored tracks + hardware annex + negatives) |
| **Prior result** | GIC 2026 — **dual-track finalist** (Mitsubishi/AIST materials track); same framework, same instrument discipline |
| **Public repository** | https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026 |

## The claim in one line

For interacting mobility systems the decisive quantity is the real-frequency
collective response S(q,ω). The scalable classical route (imaginary-time +
analytic continuation) is **mathematically ill-posed at any size**; quantum
hardware travels the real-time route natively; we demonstrated it on 156
qubits and **measured, then adversarially adjudicated, the classical
boundary from both sides** — reproduced at k≤6 (0.4–1.4σ), diverged at k≥8
(8.9σ/11.5σ, hostile-referee verdict shipped).

## Verify the headline claims

| Claim (§) | Command / artifact | Expected |
|---|---|---|
| Advantage of route: S(q,ω), no continuation (§1) | `results/vw_jam_relaxation_spectrum.json` | dominant collective line 0.0625 cyc/step, q=0.5–2.0 |
| Exact-theorem anchors (§5) | `python fable_a1p_flight.py model` | 4/4 gates; waveform corr ~0.98 |
| RL 35% budget + product robustness (§2) | `results/rl_finalquality_classical.json` | E 76.4% vs A 67.1% safe (+9.3 pts) |
| RL device arm (§2) | `python fable_vw_rl.py 21 A` (baseline, CPU) | rollouts-to-target ~448 |
| Flow encoding advantage (§4) | `results/flow_encoding_test.json` | simplex −2.41 vs binary −1.91, 3.5× tighter |
| Directed traffic (demo, §7) | `python fable_traffic_directed.py model` | G1/G2/G3 PASS |
| Full receipt map | `RECEIPTS.md` | claim → file → job-ID |

## What is in this package

```
VW_REPORT_v2.md                the write-up (advantage of route + 2 scored tracks + annex + negatives)
QUANTUM_ADVANTAGE_EXHIBIT.md   the advantage exhibit, standalone
RECEIPTS.md                    claim -> file -> job-ID map (hardware auditable without credentials)
results/                       result JSONs behind every headline number
fable_a1p_flight.py            exact-theorem anchor derivation + flight (model gate = CPU)
fable_traffic_directed.py      directed mobility card (model gate = CPU, statevector)
fable_vw_rl.py                 quantum-in-the-loop RL training pipeline
```

## Discipline (why the claims are auditable)

Every result carries a frozen pre-registration written **before** execution,
≥3 seeds with mean ± SD, in-job controls, and cloud job IDs. Key results are
replicated on a second independent machine. Every negative result (§7 of the
write-up) ships with the same receipt class as every pass — including a
hostile classical attack we commissioned against our own advantage claim,
whose verdict we adopted verbatim. Model gates and grading run
credential-free; flight (scout→fly→grade) needs IBM Quantum / QCi Dirac-3
access. Open-plan spend guard pinned in every flight script.

## Note on scope

The advantage is an advantage **of route**, stated and bounded — not an
absolute-supremacy claim. The scored track results (RL budget/robustness,
safety certificate) are working deliverables on today's devices; the
route advantage is the Phase-II direction, with its activation boundary
already measured and published here.
