# Merlin Quantum — GIC 2026 · Volkswagen — Vision-Language-Action Models

Phase-1 concept proposal, 2026 Global Quantum + AI Challenge.

| | |
|---|---|
| **Submitted proposal** | [`VW_PROPOSAL_v7.md`](VW_PROPOSAL_v7.md) — rendered as `report.pdf` |
| **Team** | Merlin Quantum, the quantum applications division of Merlin Digital (Dubai) |
| **Independent validation** | GIC 2026 **dual-track finalist** — Mitsubishi/AIST materials and QCi tracks; same framework, same instrument |
| **Repository** | https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026 |
| **Superseded material** | `archive/` — earlier drafts and planning notes, kept for provenance; not part of the submission |

## Claim → receipt

Every measurement the proposal reports, with the file in this repository that backs it. Each entry resolves inside this repo.

| # | Measurement | Receipt |
|---:|---|---|
| 1 | Control safety — the statement's metric against its named baseline (3 seeds × 500 episodes per arm, identical suite, identical starts | [`fable_vw_rubric.py`](fable_vw_rubric.py) · [`vw_rubric_headline.json`](results/cited/vw_rubric_headline.json) · [`vw_rta_demo.json`](results/external_receipts/vw_rta_demo.json) |
| 2 | Certification under the measured disturbance class. The ensembles are the statement's synthetic perturbation suite; three of them, at identical rms power, 3 seeds × 1,000 episodes per cell: against the measured collective… | [`floor_vw_surrogates.json`](floor_vw_surrogates.json) |
| 3 | Safety stage, scored on the statement's metric — episodes ending in a safe state at the operating point the certificate certifies (mean ± SD, 3 seeds × 400 episodes; the quantum component, the disturbance class, is the only… | [`floor_vw_surrogates_seeds.json`](floor_vw_surrogates_seeds.json) |
| 4 | Resource allocation. The statement's Compression sub-track specifies LLaVA-1.5-7B on nuScenes or Waymo against INT8 via bitsandbytes; we measured a vision tower outside it (CLIP ViT-B/32, CIFAR-100 probe, protocol frozen),… | [`TRACKA_REPORT.md`](results/external_receipts/TRACKA_REPORT.md) |
| 5 | Directed-path traffic card — converges classically, kept as negative control | [`trafficD_kingston_result.json`](results/trafficD_kingston_result.json) |
| 6 | Flow-encoding test, 6 paired trials (12 jobs) | [`flow_encoding_test.json`](results/flow_encoding_test.json) |
| 7 | Surrogate-class ablation, seed-level (3 × 400 episodes × 6 classes × 11 amplitudes) | [`floor_vw_surrogates.py`](floor_vw_surrogates.py) |
| 8 | The unscoped compression study and the independent RL replication ran on a second internal workstation — the "GPU box" of the receipts | [`RL_FINALQUALITY_BOX.md`](results/external_receipts/RL_FINALQUALITY_BOX.md) |

## Verifying

```
python verify.py
```

Replays the headline numbers from archived counts — no credentials, no network, numpy only.
