# Merlin Digital — GIC 2026 Volkswagen Enterprise Track

| | |
|---|---|
| **Team** | Merlin Digital |
| **Project** | Quantum-in-the-Loop VLAM Alignment and Certified Control Safety for in-vehicle / robot-controller compute |
| **Tracks** | RL Alignment (primary scored) · Safety (secondary scored) |
| **Context** | VLAM-grade perception and reasoning on ISO 26262 / IEC 62061 controllers, ≤100 ms inference |
| **Write-up** | `VW_REPORT_v2.md` (challenge map → two VLAM stages → ablation → instrument → negatives) |
| **Prior result** | GIC 2026 — **dual-track finalist** (Mitsubishi/AIST materials track); same framework, same instrument |
| **Public repository** | https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026 |

## The claim in one line

A VLAM does not fit on a vehicle. This package puts a quantum / QI
component into the two stages that decide whether it can ship:
**RL alignment of the action policy (−35.4% rollouts vs named GRPO,
+9.3 pt safer at equal budget)** and **certified runtime assurance of
the control output (77.9% → 100% safe, 22 µs, 4,500× under 100 ms)**.
The 7B backbone stays a swappable black box. Footprint and O(N²)
attention are not claimed — the compression kill-switch fired because
classical INT4 already clears the bar.

## Challenge demand → this package

| Demand | Deliverable |
|---|---|
| Component in ≥1 VLAM stage | Alignment (Dirac-3 curriculum) + control-output safety wrap |
| ≥10% efficiency or ≤2× @ ≤5% | −35.4% RL budget (bar 10%); compression not scored |
| ≥3 runs, mean ± SD; Q/QI ablation | 3 seeds; six-arm RL ladder + rank ablation |
| ≤100 ms inference | 22 µs monitor; quantum is train-time only |
| 4–8 page report; clean-env README | `VW_REPORT_v2.md`; this file + `python verify.py` |

## Verify the headline claims

| Claim (§) | Command / artifact | Expected |
|---|---|---|
| RL 35% budget + product robustness (§2) | `results/rl_finalquality_classical.json` | E 76.4% vs A 67.1% safe (+9.3 pts) |
| RL device arm (§2) | `python fable_vw_rl.py 21 A` (baseline, CPU) | rollouts-to-target ~448 |
| Safety 78→100%, 22 µs (§3) | `results/vw_rta_demo.json` | 100% wrapped; ~22e-6 s / decision |
| Flow encoding of the RL quantum stage (§5) | `results/flow_encoding_test.json` | simplex −2.41 vs binary −1.91, 3.5× tighter |
| Supporting S(q,ω), no continuation (annex) | `results/vw_jam_relaxation_spectrum.json` | dominant collective line 0.0625 cyc/step |
| Exact-theorem anchors (annex) | `python fable_a1p_flight.py model` | 4/4 gates; waveform corr ~0.98 |
| Full receipt map | `RECEIPTS.md` | claim → file → job-ID |

Clean-environment check of every headline number:

```
pip install -r requirements.txt
python verify.py
```

No cloud credentials required. Flight (scout→fly→grade) needs IBM Quantum
/ QCi Dirac-3 access and is not part of verification.

## What is in this package

```
VW_REPORT_v2.md                4–8 page technical report (challenge-aligned)
QUANTUM_ADVANTAGE_EXHIBIT.md   supporting mobility-spectrum annex (not scored)
RECEIPTS.md                    claim -> file -> job-ID map
results/                       result JSONs behind every headline number
verify.py                      credential-free headline check
fable_a1p_flight.py            exact-theorem anchor derivation (model gate = CPU)
fable_traffic_directed.py      directed mobility card (model gate = CPU)
fable_vw_rl.py                 quantum-in-the-loop RL training pipeline
```

## Discipline

Every result carries a frozen pre-registration written **before** execution,
≥3 seeds with mean ± SD, in-job controls, and cloud job IDs. Key results are
replicated on a second independent machine. Every negative result (§7 of the
write-up) ships with the same receipt class as every pass. Open-plan spend
guard pinned in every flight script.
