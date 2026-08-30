# RECEIPTS — claim → file → job-ID map (VW Enterprise Track, GIC 2026)

Scored deliverable: quantum / QI component in two VLAM stages (RL
alignment + control-output safety). Supporting annex: jam-relaxation
spectrum. Every headline claim in `VW_REPORT_v2.md` maps to a shipped
artifact and, where it is a hardware claim, an IBM Quantum / QCi Dirac-3
job ID. Hardware figures regenerate from archived data with no
credentials; submitting new jobs needs IBM Quantum / qBraid access.

## Advantage of route (§1)

| Claim | Artifact | Job ID / receipt |
|---|---|---|
| S(q,ω) jam-relaxation spectrum, no continuation step | `results/vw_jam_relaxation_spectrum.json` | IBM Heron fez `d9rm4j9dsedc73agrb70` (real-time series source) |
| Real-time collective interface, 156 qubits | `results/a1p_result_20260808_200224.json` | fez `d9rm4j9dsedc73agrb70` (36×8,192) |
| Adjudicated classical boundary (two-sided) | `A1_FINAL_VERDICT.md` (in session record) | kingston `d9rdfb1dsedc73agh5ng` (16×32,768) + attack receipts |
| Exact-theorem anchors 0.9836/0.9806; 18-line spectrum vs detuned | `results/a1p_result_20260808_200224.json` | fez `d9rm4j9dsedc73agrb70`; scout `d9rm47opdb6s73e53kqg` |
| CHSH in-kind anchor +35.9σ, replicated | `chsh_*_20260805_*.json` (GIC ledger) | fez, same-day independent replica |

## RL alignment (§2, scored)

| Claim | Artifact | Job ID / receipt |
|---|---|---|
| 35.4% under GRPO budget, 3/3 seeds | `vw_rl_{A,B,C,D,E}_s{21,22,23}.json` | 71 QCi Dirac-3 job IDs listed in the C-arm JSONs |
| +9.3-pt robustness at equal budget | `results/rl_finalquality_classical.json` | device arm receipts in `rl_finalquality_dirac.json` |
| ~2× budget-equivalence + variance collapse, two-box | `RL_FINALQUALITY_BOX.md` (independent machine) | replicated, CPU-only second box |
| Six-arm ablation complete (D breaks, E parity, C′ confirms) | `rl_cflow_result.json` + arm JSONs | Dirac-3 job IDs per arm |

## Safety (§3, scored)

| Claim | Artifact | Receipt |
|---|---|---|
| 78→100% under RTA, 22 µs monitor | `vw_rta_demo.json` | 3 seeds × 500 episodes |
| Envelope predicted then confirmed, two machines | `vw_lyap_bias_sweep.json`, `vw_lyap_seeds.json` | [1.2,1.6] predicted; measured inside |
| Rank ablation r=3 interior-optimal | `vw_lyap_validation.json` | 3 seeds |
| Dirac-trained certificate (methods) | `results/trackb_seed{21,22,23}.json` | Dirac-3 job IDs in JSONs |

## Framework-led device physics (§4)

| Claim | Artifact | Receipt |
|---|---|---|
| Flow encoding −26%, 3.5× tighter (frozen prediction) | `results/flow_encoding_test.json` | 12 paired Dirac-3 jobs |

## Full-capability instrument (§5.1) — banked, no copy flights

| Claim | Artifact | Receipt |
|---|---|---|
| ζ atlas + sector arithmetic, kingston 9/9 | `results/npoint2_result_20260830_014105.json` | kingston `da9l3t1qtnsc73d1nhd0` / `da9l9rkjbipc73ff0aqg` |
| ζ atlas + sector arithmetic, fez 8/9 | `results/npoint2_result_20260829_182700.json` | fez `da9eoe6rbfbs73chiq0g` / `da9lqeerbfbs73chq63g` |
| T2 composition 7%/9% on 18 edges | `results/t2_bridge_result.json` | banked re-analysis of the v2 raws |
| Native operator E₀ inverted, 5/5 live tiles | `results/ncomp_regrade_20260830_122917.json`, `results/ncomp_regrade_20260830_123936.json` | kingston `da9vgsmrbfbs73ci44d0`; fez `da9vblkjbipc73ffaio0` |
| Estimator inversion (best tile −0.006) | `results/nc1_energy_inverted.json` | same jobs; V = L0 |

## Negatives, receipted (§7)

| Negative | Artifact | Receipt |
|---|---|---|
| Compression empty on both sides of INT4 | `TRACKA_REPORT.md`, `tracka2_*` | box M1/M3 real-accuracy eval |
| Directed Path B converges classically | `results/trafficD_kingston_result.json` + `TRAFFIC_ATTACK_VERDICT.md` | kingston refly + hostile attack (~5 min) |
| Validation-ensemble null on single vehicle | `vw_qens_result.json` | 3 seeds × 500 |
| Rank universality refuted at d=3 | `vw_ceiling_rank_d3.json` | 3 seeds |

## Reproduce the framework-native flights

```
python fable_a1p_flight.py model      # exact-theorem anchor derivation + gates (CPU)
python fable_traffic_directed.py model# directed card model gate (CPU, statevector)
python fable_vw_rl.py <seed> A        # RL baseline arm (CPU)
python fable_vw_rl.py <seed> C        # RL device arm (needs QCi Dirac-3 credentials)
```

Flight (scout→fly→grade) needs IBM Quantum / QCi credentials; all grading and
model gates run credential-free. Open-plan spend guard is pinned in every
flight script (refuses any backend outside {ibm_fez, ibm_kingston,
ibm_marrakesh}).
