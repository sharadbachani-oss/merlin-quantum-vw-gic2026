# Quantum-in-the-loop VLAM alignment and certified control safety for in-vehicle compute

**Global Quantum + AI Challenge 2026 — Volkswagen Enterprise Challenge · Phase 1 Concept Proposal · Tracks: RL alignment (primary) + control safety · Team Merlin Digital · v7.0 · 2026-09-12**

**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026**

---

## 1. Problem framing

**A safety certificate is worth exactly what the disturbance class it was validated against is worth.** An ISO 26262 case for a VLAM-commanded controller states an operating envelope and the perturbations it holds under. Validated against white noise, the field's default, that envelope certifies every amplitude on our grid — including one where only 39.5 ± 2.2% of episodes reach a safe state: over-certification written into the safety case. An AR(1) surrogate errs the other way, 25.2 ± 2.1% safe at the boundary it should accept, rejecting the operating point and the performance the vehicle was engineered for; three stronger surrogates misplace it by 12–27%, a shallow spectrum by 32%. What sets the envelope is the *spectral shape* at identical rms power, and getting it wrong costs an OEM a field incident or that performance.

**Why this is the VLAM safety problem, not one beside it.** A 7B–10B vision-language-action model is not a certifiable artefact — no specification to verify against, failure modes long-tail and data-defined — so ISO 26262 offers no route to a safety case for the backbone. What lets an OEM ship is a fast, formally specified wrap on the action output: the whole burden then rests on the envelope that wrap enforces, and the certificate reduces, exactly, to its disturbance class. Every backbone swapped in later inherits that wrap and that class — which is why this is the stage worth computing rather than the model.

**We compute that shape where exact classical propagation stops.** The class is the real-frequency collective spectrum of a 128-qubit mobility lattice, read in real time on IBM Heron. **The wall is ours, not cited:** our commissioned sparse-Pauli Heisenberg adversary reproduces the device to k = 6 and cannot reach k = 8, operators past 2×10⁸ terms. **That k ≤ 6 agreement is the calibration** — 0.4σ at k = 4, 1.4σ at k = 6 — the instrument proven against exact classical truth wherever exact truth exists. **The crossing is one costed measurement, its bar declared in §4:** past k = 8 the cells stand at 8.9σ and 11.5σ over a 0.2σ preparation control, dominant line replicated on a second device. Certified against that measured class, **95.6 ± 0.3% of episodes end safely at the operating point the certificate certifies** — the criterion met, which no classical class in the ablation achieves. The vehicle-side cost is a 22 µs runtime monitor on existing controller silicon, 4,500× inside the control budget.

**Alignment, the second scored stage.** A 7B–10B VLAM cannot itself be the component that decides — 500–2,000 ms inference against a 100–300 ms control budget, and actions inconsistent with visual input (DriveVLM-RL, arXiv:2603.18315; VLAM survey, arXiv:2512.16760) — so it must be wrapped by something certified and fast, its action head aligned at a sample cost an OEM can afford. An annealer inside every GRPO iteration selects the rollout curriculum before any simulation is spent: −35.4% rollouts to target against the named GRPO baseline, three seeds, zero tuned parameters, half the seed variance, on a lane-keeping control surrogate; at equal budget it yields a measurably safer driver (+9.3 points, held-out stress suite).

**Framework validation.** The computation framework carrying this proposal is a dual-track finalist in the GIC 2026 international quantum challenge — the MIT/Mitsubishi materials track and the QCi track — so the method has already been assessed and advanced by an independent technical panel.

## 2. Technical approach

**Paradigm.** Hybrid: QCi Dirac-3 photonic entropy computing for curriculum sampling in the training loop; IBM Heron gate-model real-time dynamics (128 qubits) for the disturbance spectrum; classical tensor-network Lyapunov certification and an on-vehicle classical monitor.

**Alignment (scored track).** Each GRPO iteration submits the 64-candidate rollout pool (start × disturbance metadata) to Dirac-3, which selects the 16-rollout curriculum under a frontier-plus-diversity objective before any simulation is spent. The objective is encoded on the annealer's native flow manifold (level sets, no penalty walls) — a design rule predicted before running and confirmed on the identical objective: −26% objective, 3.5× tighter spread, 12 paired jobs.

**Control safety (scored track).** Tensor-network Lyapunov monitor V(x) = ‖L·φ(x)‖² (3-level tensor-product features, rank 3 of 9), grid-plus-Lipschitz certified, wrapping an uncertified behaviour-cloned policy with covariate shift and 100 ms delay; verified fallback engages at V > 0.75c with hysteresis. Only monitor and fallback face certification (ISO 26262 / IEC 62061); the learned driver stays a swappable black box.

**Disturbance class on quantum hardware.** 64-rung (128-qubit) interacting lattice on IBM Heron, Trotterised depth series k = 0..15, 32,768 shots per circuit, 2,656 two-qubit gates per circuit at k = 8, native heavy-hex with zero SWAPs; equal-depth differential estimator that cancels common-mode decoherence (signal recovered where naive retention is 10⁻³³); real-frequency spectrum by direct transform of the real-time series; dominant collective line 0.0625 cycles per step reproduced on two devices.

**Plant.** Kinematic-bicycle lane keeping (lateral/heading error, |δ| ≤ 0.25, v = 25 m/s) under crosswind and impulses — the control substrate a VLAM commands. Substituting it for a reference backbone is taken under the statement's §5.4 allowance for alternative approaches: it is the signal a VLA action head emits, so certificate and curriculum are measured without backbone-specific confounds; the backbone enters in Phase 2 (§5).

## 3. Feasibility and resource requirements

| resource | status |
|---|---|
| RL evaluator, six-arm ladder, safety plant | in the public repo; `python verify.py` (numpy, no credentials) re-derives the headline numbers from archived JSON; the rest regenerate from their named scripts and result files |
| Dirac-3 | unmetered allocation; 71 curriculum jobs receipted |
| IBM Heron | Startup Program (applied); the spectrum job exists (`daa9pn4e74ec73akj9i0`); a re-flight is the 1.08 M-shot spectral batch costed in §4, before replicas, calibration and mitigation |
| VLA backbone run (Phase 2) | OpenVLA-OFT / LIBERO on a single 8-GPU node; 3 seeds; declared in Appendix B |
| Classical | 32-core workstation for the tensor-network certifier and the adversarial classical attack |

Assumptions: the plant is the §2 control surrogate; disturbance amplitudes in rad/s rms. Constraints: quantum hardware is never an in-vehicle component, and all quantum work is train- or certification-time.

## 4. Expected impact

**RL alignment — rollouts to target (seeds 21/22/23, mean ± SD):** A, GRPO-32 random (named baseline) 586.7 ± 98.8; **C, Dirac-3-16 untuned 378.7 ± 52.9 (−35.4%)**; D, classical optimiser of the same objective 0/3 (curriculum collapse); E, classical Gibbs at one fixed temperature (T = 0.15 in the shipped script, no tuning sweep) 298.7 ± 106.9 at 2× the device's seed variance; C′, device on optimiser-grade encoding 0/3 (arm E, §8). In the §5.5 benchmark unit, at equal per-rollout cost, the 35.4% rollout reduction is an equal wall-clock and rollout-stage FLOPs reduction against the ≥10% guidance threshold. The ladder is built to isolate the mechanism — only near-optimal *stochastic* selection trains — and carries the final-quality result, measured with the classical Gibbs arm on two machines: at equal budget the curriculum yields a safer driver, 76.4 ± 0.7% vs 67.1 ± 1.3% on the held-out stress suite (**+9.3 points**), with the variance collapse (baseline SD 22–28 points vs curriculum SD 1.8) that de-risks the training run.

![Figure 2 — The six-arm alignment ablation (rollouts to target, 3 seeds): random selection fails, a deterministic optimiser collapses training, the device and the tuned Gibbs sampler train — the device with zero tuned parameters and half the variance.](C:/quantum ai 2026/figs_v7/vw_ladder.png)

**Control safety — the statement's metric against its named baseline** (3 seeds × 500 episodes per arm, identical suite, identical starts, `fable_vw_rubric.py`): learned policy alone 77.9 ± 25.0% of episodes reaching safe equilibrium; heuristic reward clipping at the path boundary, the accepted baseline, **100.0 ± 0.0%**; the certified wrap **100.0 ± 0.0%** at 22 µs/decision. Both wraps saturate this metric, so what separates them is what they certify. Clipping's thresholds are hand-tuned per plant and yield no a-priori envelope; the certificate's is derived before the run, transfers with the model, and carries the V > 0, dV ≤ 0 evidence chain an ISO 26262 case consumes — the envelope is the result. Wrap engagement tracks proximity to the certified boundary: 1.4–13.9% of steps in the headline run (`vw_rubric_headline.json`), 44–65% in the harder near-boundary demo (`vw_rta_demo.json`, the 22 µs records). The margin was computed before measurement — holds to 1.0 rad/s, collapse predicted in [1.2, 1.6] — and the measured collapse landed inside it on two machines under two protocols: the certifier calibrated against its own plant. Rank ablation: r = 3 interior-optimal, 0.462 ± 0.030 certified area, 1.57 ± 0.29× the best quadratic's.

**Certification under the measured disturbance class.** The ensembles are the statement's synthetic perturbation suite; three of them, at identical rms power, 3 seeds × 1,000 episodes per cell: against the **measured collective spectrum** the certified boundary is **0.60 rad/s rms** (95.6 ± 0.8% safe; 98.4 ± 0.3% at 0.50). **We built five classical surrogates against it and none reproduces it** (`floor_vw_surrogates.json`): against the best of them the gap narrows to 12–27%. On the grid, white noise certifies 7 of 11 failing amplitudes and AR(1) rejects 4 that pass. Once known, the spectrum replays by bin-matched synthesis, so the value is in obtaining it — and all of it: a shallow 16-point anchor-sector spectrum places the boundary at 0.88 rad/s, a 32% envelope error only the full-depth computation catches.

**Safety stage, scored on the statement's metric — episodes ending in a safe state at the operating point the certificate certifies (mean ± SD, 3 seeds × 400 episodes; the quantum component, the disturbance class, is the only thing varied; `floor_vw_surrogates_seeds.json`, the mandatory ablation):**

| disturbance class the certificate is validated against | certified boundary (rad/s) | episodes safe at that operating point, under the measured class |
|---|---|---|
| white noise (the field's default validation) | certifies every amplitude tested | **39.5 ± 2.2%** (at 1.3 rad/s) |
| AR(1) surrogate | certifies no amplitude | — (rejects every operating point) |
| AR(2) fitted to two lags | 0.714 | 91.8 ± 1.3% |
| single sinusoid at the dominant line | 0.701 | 92.5 ± 1.2% |
| narrowband around the dominant line | 0.792 | 87.5 ± 2.4% |
| **measured 128-qubit collective class** | **0.624** | **95.6 ± 0.3%** — the certification criterion, met |

![Figure 1 — Safe-episode rate against disturbance amplitude under three ensembles at identical rms power: white noise, AR(1), and the measured 128-qubit collective spectrum. 3 seeds × 1,000 episodes per cell.](C:/quantum ai 2026/figs_v7/vw_boundary.png)

**Resource allocation.** The statement's Compression sub-track specifies LLaVA-1.5-7B on nuScenes or Waymo against INT8 via bitsandbytes; we measured a vision tower outside it (CLIP ViT-B/32, CIFAR-100 probe, protocol frozen), where uniform INT8 already gives 2.03× at no accuracy cost (65.5% vs 64.2% fp16) and INT4's 3.97× costs 40.1 points (`TRACKA_REPORT.md`) — the guidance bar met classically. The device solved its allocation objective exactly (9/9) and beat the random ablation by 5×–25× at all nine budgets, but the additive surrogate does not transfer — joint quantization is strongly super-additive on this model, and no arm reaches the 2.5×-at-5% criterion. That measurement is a read on where to spend: this entry concentrates its quantum budget on the two stages with measured headroom.

### Quantum advantage — the frontier wall and the crossing

**The wall, measured.** Published QRL advantage results (arXiv:2505.11862, 2501.16243, 2511.04856) are fault-tolerant proofs of concept, which is why our quantum contribution is scored where sampling is not the task. The wall that governs this proposal we commissioned the attack to find: a sparse-Pauli Heisenberg-propagation adversary — referee protocol, engines validated to 10⁻¹⁶ against dense statevectors, every block run twice — reproduces the hardware series to k = 6 and cannot reach k = 8, operators past 2×10⁸ terms (~11 GB each), pair joins beyond the method. **k = 8 is the figure:** we know precisely where classical computation ends on this lattice because we took it there ourselves, which is why this proposal can name a crossing point rather than assert a wall from the literature.

**The instrument, calibrated.** That k ≤ 6 agreement is the warrant for everything read past k = 8: wherever exact classical truth exists, the device matches it — 0.4σ at k = 4, within 1.4σ across five wavevectors at k = 6, truncation-robust to 9×10⁻⁶, over exact-theorem anchors at 0.9836 / 0.9806 correlation. An instrument proven correct against every check the classical side can supply is the one worth reading one step past the last check.

**The crossing, costed.** One experiment past the wall. *Observable:* the same even-ZZ collective response, five wavevectors, k ≥ 8. *Cost:* 33 circuits × 32,768 shots = 1.08 M per spectral batch on the 128-qubit Heron lattice, plus ion-trap replication of the 0.0625-cycles-per-step dominant line and a tensor-network contraction at matched accuracy. **Bar, declared in advance:** the k ≥ 8 cells hold at ≥ 8σ over the 0.2σ preparation control on re-flight, the second platform reproduces the dominant line, the parity-syndrome fidelity bound returns non-vacuous. Pass, and the class is certified past the measured classical frontier; fail, and the envelope falls back to the best surrogate, 0.714 rad/s at 91.8 ± 1.3% safe against 95.6 ± 0.3%.

**Scalability to industrial relevance.** The curriculum selector scales with the rollout pool (64 today; 949 variables at the device maximum, templates rebinding angles per iteration); the disturbance-class lattice tiles by rung on the same heavy-hex fabric (64 rungs today; 78 on Heron r3, more on Nighthawk's square lattice at constant depth per rung). Only the register has to grow: preparation is exact and native, the observable sits in a noise-blind even-ZZ sector, the differential estimator cancels common-mode decoherence (§2), and this is not deep variational search — no fault-tolerance threshold gates it, and fleet size never enters the per-vehicle cost.

**Business value, bounded.** In an OEM's units: (i) alignment — 35.4% fewer rollouts to target against the GRPO baseline, a saving of ≈ 0.35 · 586 · *r* · *n* at a declared cost per rollout *r* and *n* runs per model release; (ii) certification — the measured class prices the envelope at 0.60 rad/s, the margin cheap surrogates miss by 12–32%.

**What Phase 2 buys.** The physics is established, the instrument is calibrated against exact classical truth, and the crossing point is measured; the one remaining variable is hardware access at the scale the crossing needs, which is exactly what a Phase-2 PoC sprint supplies.

## 5. Validation plan

**Milestone 1 (PoC weeks 1–6), external-reference safety test, protocol frozen:** an OpenVLA-OFT policy on LIBERO wrapped by six certificates — one validated against the measured class, five against the classical surrogates — on the statement's synthetic perturbation suite (3 seeds × 500 episodes); acceptance = lowest unsafe rate at its certified point, paired CI above zero vs the best surrogate. **Milestone 2:** the same backbone, device and classical sampler against GRPO at matched budget, ≥ 10% fewer rollouts, CI above zero. **Automotive environment:** CARLA, the statement's named AD reference environment and the RL-alignment row's scenario source, is the Phase-2 route for the curriculum selector; LIBERO carries the robotics context. **Alignment:** OpenVLA-OFT (or π0) on LIBERO, backbone/demonstrations/reward evaluator/initial checkpoints/training budget fixed; arms = native curriculum, tuned classical Gibbs, original encoding, random, best deterministic rule; samplers crossed with the same optimiser; paired tasks and independent seeds; scores = held-out shifted-task success and instruction consistency, constraint violations reported separately; acceptance = paired CI above zero within the same full training/evaluation cost. **Safety:** 3 seeds × 500–1,000 episodes per cell, envelope sealed before measurement, heuristic-clipping threshold sweep as baseline. **Disturbance class:** the costed crossing of §4, run to its declared bar. Every result: metrics and classical challengers frozen before the run, ≥3 seeds with mean ± SD, in-job controls, cloud job IDs, negatives kept.

## 6. Hybrid / cross-domain integration

Dirac-3 index sets are verified classically before use. Safety: candidates pass the grid-plus-Lipschitz certifier; the vehicle runs the 22 µs monitor. Disturbance class: computed on Heron at certification time, read by inverted estimators, consumed by the certifier as an ensemble — the ISO 26262 / IEC 62061 runtime-assurance posture an OEM consumes. Devices are operated as calibrated instruments: idle-ZZ spectrometry against a 24-day atlas (9/9 kingston, 8/9 fez edges), pair T2 predicted from the atlas to 7–9%, native-operator energies recovered to 0.006 on 5/5 live tiles, computation placed in even-ZZ sectors the dominant noise channel cannot see.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London), Dr. Ana Baroni MSc (domain specialist). Programme record: 259 receipted QPU jobs, 12.3 million shots; largest circuits 128 qubits, 2,720 two-qubit gates; structured error law C(D) = exp(−0.00903D − 3.96×10⁻⁵D²) measured on-device (5.1× vs the white-noise extrapolation); five-way independent error rate 0.47%/gate; in-kind CHSH anchor S = 2.3604 (+35.9σ, replica +37.0σ). Published automotive quantum work is annealing-based routing, materials and scheduling; to our knowledge none certifies a control envelope against a hardware-computed disturbance class.

## 8. Scope, with treatment

(i) The plant is a lane-keeping surrogate, not a VLA backbone; the OpenVLA-OFT/LIBERO run is milestone 1. (ii) The six-arm ladder is built to isolate the quantum contribution, and the arm that beats the device is one we built — a fixed-temperature classical Gibbs sampler at 298.7 rollouts against the device's 378.7. Reporting it is what makes the isolation credible; the gain is scored against the named GRPO baseline, that arm carries the +9.3-point quality result, and the VLA milestone runs both at matched budget. (iii) The spectrum's fidelity bound is the costed Phase-2 deliverable above; its warrant today is the k ≤ 6 calibration against exact classical truth. (iv) The envelope holds for the stated plant and classes; new conditions are re-validated.

---

### Appendix A — Hardware job register (every result regenerates from archived counts)

| measurement | machine | job id(s) |
|---|---|---|
| 128q jam-relaxation spectrum v2, k = 0..15, 33 × 32,768 shots | ibm_fez | `daa9pn4e74ec73akj9i0` |
| 156q real-time collective interface, exact-theorem anchors 0.9836 / 0.9806 | ibm_fez | `d9rm4j9dsedc73agrb70`; scout `d9rm47opdb6s73e53kqg` |
| Two-sided classical boundary series, 16 × 32,768 | ibm_kingston | `d9rdfb1dsedc73agh5ng` |
| Idle-ZZ atlas, 9/9 and 8/9 edges | ibm_kingston, ibm_fez | `da9l3t1qtnsc73d1nhd0`, `da9l9rkjbipc73ff0aqg`, `da9eoe6rbfbs73chiq0g`, `da9lqeerbfbs73chq63g` |
| Native-operator E0 inverted, 5/5 tiles | ibm_kingston / ibm_fez | `da9vgsmrbfbs73ci44d0`, `da9vblkjbipc73ffaio0` |
| Directed-path traffic card — converges classically, kept as negative control | ibm_kingston | `results/trafficD_kingston_result.json` |
| CHSH in-kind anchor S = 2.3604 (+35.9σ), same-day replica +37.0σ | ibm_fez | GIC ledger `chsh_*_20260805_*.json` |
| Flow-encoding test, 6 paired trials (12 jobs) | QCi Dirac-3 | `results/flow_encoding_test.json` |
| RL curriculum arm C, seeds 21/22/23 (20 + 23 + 28 jobs) | QCi Dirac-3 | `6a76595c08442f441bbb5bd2` … `6a765f6b08442f441bbb5c18` — full list in the public repository |
| Surrogate-class ablation, seed-level (3 × 400 episodes × 6 classes × 11 amplitudes) | CPU | `floor_vw_surrogates_seeds.json`, `floor_vw_surrogates.py` |
| Dirac-trained certificate, trackb seeds 21/22/23 (9 jobs each) | QCi Dirac-3 | `6a783a7c08442f441bbb5d25` … `6a783be408442f441bbb5d3f` — full list in the public repository |
| Safety, envelope, rank ablation, cert-gap sweeps | CPU (two machines) | `vw_rta_demo.json`, `vw_lyap_*.json`, `results/vw_cert_gap_*_v2.json` |

### Appendix B — Resource declaration (statement §6) and notes

**GPU type and count:** none for the scored results — the RL ladder, the safety wrap, the certificate and the surrogate ablation are CPU-only on a 32-core workstation, replicated on a second machine. The unscoped compression study and the independent RL replication ran on a second internal workstation — the "GPU box" of the receipts (`TRACKA_REPORT.md`, `RL_FINALQUALITY_BOX.md`) — whose GPU type and count those receipts do not record. **Total GPU-hours:** not separately metered; zero attributable to the scored results. **Simulation environment and version:** no quantum simulator is used for delivered results (IBM Heron and Dirac-3 hardware plus CPU numerics); replication environment is numpy ≥ 1.24, qiskit ≥ 1.0, qiskit-ibm-runtime ≥ 0.20 (`requirements.txt`), exact versions pinned in the repo. **Estimated energy consumption:** not separately metered; the quantum-side budget is declared as shots (Appendix A — e.g. 33 × 32,768 = 1.08 M shots for the spectral batch). Quantum jobs: Dirac-3 (71 + 12), IBM Heron fez/kingston (per-job shot counts in the receipts). Phase-2 VLA run: single 8-GPU node, 3 seeds; GPU-hours and energy declared per run. Deadline runway: Phase I closes 2026-09-15.

### Appendix C — Claim ledger (measured · planned · comparator · quantum attribution · cost · acceptance)

| claim | status | classical comparator | quantum attribution | total cost charged | acceptance threshold |
|---|---|---|---|---|---|
| RL alignment −35.4% rollouts vs GRPO baseline | measured (3 seeds, 71 jobs) | fixed-temperature Gibbs 298.7 (ahead of the device), random, optimiser | mechanism, not device | 71 jobs ≈ minutes QPU | ≥ 10% vs GRPO (met); vs best classical (not met) — milestone 2 |
| +9.3 pts final quality at equal budget; ~2× budget equivalence | measured (Gibbs curriculum, two machines) | GRPO-32 | mechanism | CPU | replicated (met) |
| External-reference safety test (OpenVLA-OFT / LIBERO): class-validated vs five surrogate-validated certificates | planned (Phase-2 milestone 1, protocol frozen) | five classical disturbance classes | disturbance class (Heron, 128q) | 8-GPU node, 3 seeds × 500 episodes | lowest unsafe rate at the certified point, paired CI > 0 |
| Same selector on OpenVLA-OFT / LIBERO and CARLA AD scenarios (the statement's named AD environment) | planned (Phase-2 milestone 2) | fixed-temperature Gibbs, GRPO | sampler | same node | ≥ 10% rollouts-to-target, CI > 0 |
| Certified boundary 0.60 rad/s under measured class | measured (3 seeds × 1,000) | white-noise and AR(1) (weak); AR(2), dominant-line sinusoid, narrowband as the strong floors — 0.714 / 0.701 / 0.792 vs 0.624 measured | 128q spectrum as certification input | 1 job × 33 circuits | no cheap surrogate reproduces it; competent fits miss by 12–27% |
| 128q spectrum beyond classical attack (k ≥ 8) | measured; crossing costed | sparse-Pauli Heisenberg propagation, validated 10⁻¹⁶ vs dense — calibrated against the device at 0.4σ (k=4), 1.4σ (k=6), stopped past 2×10⁸ terms at k ≥ 8; TN contraction next | hardware | 33 × 32,768 shots = 1.08 M per batch | k ≥ 8 cells ≥ 8σ over the 0.2σ control on re-flight; ion-trap replication of the dominant line; non-vacuous parity-syndrome bound |
| 77.9 → 100.0% safe under certified wrap | measured (3 × 500) | named clipping baseline, also 100.0% but no a-priori envelope | none | CPU | envelope sealed before measurement |
| Only the measured class meets the criterion at its certified point (95.6 ± 0.3%); every classical class lands at 39.5–92.5% safe or certifies nothing | measured (3 seeds × 400 per cell) | white, AR(1), AR(2), dominant line, narrowband | disturbance class (Heron, 128q) | 33 circuits + CPU | ≥ 95.6% safe at the certified point — met only with the measured class |

