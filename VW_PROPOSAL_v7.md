# Quantum-in-the-loop VLAM alignment and certified control safety for in-vehicle compute

**Global Quantum + AI Challenge 2026 — Volkswagen Enterprise Challenge · Phase 1 Concept Proposal · Tracks: RL alignment (primary) + control safety · Team Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · v7.0 · 2026-09-12**

**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026**

---

## 1. Problem framing

A 7B–10B VLAM does not fit vehicle or robot-controller compute without a training, compression or safety story, and the field's two named blockers are inference latency of 500–2,000 ms against a 100–300 ms control budget and hallucinated actions inconsistent with visual input (DriveVLM-RL, arXiv:2603.18315; VLAM survey, arXiv:2512.16760). Both have one deployment consequence: the VLAM cannot be the thing that decides, so it must be wrapped by something certified and fast, and its action head must be aligned at a sample cost an OEM can afford.

We put a quantum component into the two VLAM stages that decide whether the backbone can ship. **Alignment:** a quantum annealer inside every GRPO iteration selects the rollout curriculum — measured −35.4% rollouts to target versus the named GRPO baseline, three seeds, on the lane-keeping control surrogate; a classical Gibbs sampler implementing the same selection at one fixed temperature reaches the target in fewer, so the alignment result is the selection mechanism, not a device advantage, and the VLA-backbone run is the PoC's first milestone. The quantum advantage in this proposal is in the safety stage: the disturbance class the certificate is validated against is a measured 128-qubit real-time spectrum whose cells past k = 8 lie beyond exact classical propagation, and a certificate validated on any of five classical classes fails its own safety criterion at the point it certifies (39.5–92.5% of episodes safe against 95.6 ± 0.3% with the measured class, 3 seeds). **The path:** the wall is exact Heisenberg propagation, which reproduces the class to k = 6 and cannot reach k = 8 (operators past 2×10⁸ terms); the crossing is the class past that wall, read on hardware at 8.9σ and 11.5σ and replicated on a second device; the Phase-2 experiment, protocol frozen before the run, certifies a VLA-backbone controller against the measured class and against five classical surrogates and tests all six on the statement's perturbation suite, an external reference. Today's hardware shows feasibility; that experiment decides the advantage. **Safety:** a certified runtime-assurance monitor (22 µs, 4,500× inside the budget) wrapping an untrusted policy, whose certified envelope is validated against a **disturbance class computed on quantum hardware** — the real-frequency collective spectrum of an interacting mobility lattice at 128 qubits. Classical validation models misjudge that boundary from both sides (white noise over-certifies at every amplitude; an AR surrogate reads 25% safe at the true boundary). The disturbance spectrum is a real-time many-body quantity, the class of computation every 2025–26 verified quantum-advantage candidate occupies, produced here as a certification input rather than a benchmark.

## 2. Technical approach

**Paradigm.** Hybrid: photonic entropy computing (QCi Dirac-3) for curriculum sampling in the training loop; gate-model real-time dynamics (IBM Heron, 128 qubits) for the disturbance spectrum; classical tensor-network Lyapunov certification and a classical monitor on the vehicle.

**Alignment (scored track).** Each GRPO iteration submits the 64-candidate rollout pool (start × disturbance metadata) to Dirac-3, which selects the 16-rollout curriculum under a frontier-plus-diversity objective before any simulation is spent. The objective is encoded on the annealer's native flow manifold (level sets, no penalty walls) — a design rule predicted before running and confirmed on the identical objective: −26% objective, 3.5× tighter spread over 12 paired jobs. Quantum work is train-time only.

**Control safety (scored track).** Tensor-network Lyapunov monitor V(x) = ‖L·φ(x)‖² (3-level tensor-product features, rank 3 of 9), grid-plus-Lipschitz certified, wrapping an uncertified behaviour-cloned policy with covariate shift and 100 ms delay; verified fallback engages at V > 0.75c with hysteresis. Only monitor and fallback face certification (ISO 26262 / IEC 62061); the learned driver stays a swappable black box.

**Disturbance class on quantum hardware.** 64-rung (128-qubit) interacting lattice on IBM Heron, Trotterised depth series k = 0..15, 32,768 shots per circuit, 2,656 two-qubit gates per circuit at k = 8, native heavy-hex with zero SWAPs; equal-depth differential estimator that cancels common-mode decoherence (signal recovered where naive retention is 10⁻³³, cells at 8.9σ and 11.5σ, 0.2σ preparation control); real-frequency spectrum by direct transform of the real-time series; dominant collective line 0.0625 cycles per step reproduced on two devices.

**Plant.** Kinematic-bicycle lane keeping (lateral/heading error, |δ| ≤ 0.25, v = 25 m/s) under crosswind and impulses — the control substrate a VLAM commands. Phase 2 puts the same curriculum selector on an accepted VLA backbone (OpenVLA-OFT or π₀ on LIBERO), see §5.

## 3. Feasibility and resource requirements

| resource | status |
|---|---|
| RL evaluator, six-arm ladder, safety plant | in the public repo; `python verify.py` (numpy) replays every headline |
| Dirac-3 | unmetered allocation; 71 curriculum jobs receipted |
| IBM Heron | Startup Program (applied); the spectrum job exists (`daa9pn4e74ec73akj9i0`); a re-flight is 33 circuits × 32,768 shots = 1.08 M shots per spectral batch before replicas, calibration and mitigation, which multiply it; total device and classical acquisition cost is reported per run |
| VLA backbone run (Phase 2) | OpenVLA-OFT / LIBERO on a single 8-GPU node; 3 seeds; GPU-hours declared in the resource declaration |
| Classical | 32-core workstation for the tensor-network certifier and the adversarial classical attack |

Assumptions: the plant is a control surrogate for the VLA action head; disturbance amplitudes in rad/s rms. Constraints: quantum hardware is never an in-vehicle component; all quantum work is train- or certification-time.

## 4. Expected impact

**RL alignment — rollouts to target (seeds 21/22/23, mean ± SD):** A, GRPO-32 random (named baseline) 586.7 ± 98.8; **C, Dirac-3-16 untuned 378.7 ± 52.9 (−35.4%)**; D, classical optimiser of the same objective 0/3 (curriculum collapse); E, classical Gibbs sampler at one fixed temperature (T = 0.15 in the shipped script, no tuning sweep) 298.7 ± 106.9 with 2× the device's seed variance; C′, device on optimiser-grade encoding 0/3. The device's measured properties are specific: it beats the named baseline by 35.4%, needs zero tuned parameters, and carries roughly half the seed variance; it does not beat the fixed-temperature classical sampler, and no quantum advantage in alignment sample efficiency is claimed on this surrogate. The mechanism is isolated in the ladder (only near-optimal *stochastic* selection trains), and it is the mechanism that carries the final-quality result, measured with the classical Gibbs arm on two machines: at equal budget the curriculum yields a safer driver: 76.4 ± 0.7% vs 67.1 ± 1.3% safe on the held-out stress suite (**+9.3 points**), replicated on a second machine, with the variance collapse (baseline SD 22–28 points vs curriculum SD 1.8) that de-risks the training run.

![Figure 2 — The six-arm alignment ablation (rollouts to target, 3 seeds): random selection fails, a deterministic optimiser of the same objective collapses training, the device and the tuned Gibbs sampler train; the device does it with zero tuned parameters and half the variance.](C:/quantum ai 2026/figs_v7/vw_ladder.png)

**Control safety:** policy alone 77.9 ± 25.0% safe at the certified boundary → **100.0 ± 0.0%** wrapped (3 seeds × 500 episodes per condition); guard idles under benign conditions (1–12% intervention); 22 µs per decision. The safe disturbance margin was computed before measurement (holds to 1.0 rad/s; collapse predicted in [1.2, 1.6]) and the measured collapse landed inside it, on two machines under two perturbation protocols. Versus tuned heuristic clipping: the envelope is known a priori, zero tuned parameters, and the V > 0, dV ≤ 0 evidence chain an ISO 26262 case consumes. Rank ablation: r = 3 interior-optimal at 0.462 ± 0.030 certified area; the TN certificate covers 1.57 ± 0.29× the best quadratic's area.

**Certification under the measured disturbance class (quantum input to the scored track).** Three ensembles at identical rms power, 3 seeds × 1,000 episodes per cell: against the **measured collective spectrum** the certified boundary is **0.60 rad/s rms** (95.6 ± 0.8% safe; 98.4 ± 0.3% at 0.50). White-noise validation grades 100.0 ± 0.0% safe at every amplitude through 1.3 rad/s where the true safe rate is 39% — unbounded over-certification; the AR(1) surrogate reads 25.2 ± 2.1% at the certified boundary itself. The strong surrogates (`floor_vw_surrogates.json`): AR(2) on two measured lags places the boundary at 0.714 rad/s, a single sinusoid at the dominant line at 0.701, narrowband noise at 0.792, against the measured spectrum's 0.624. **None of the five surrogates tested reproduces the boundary**, and the gap against a competent classical fit is 12–27%, not the 70 points the AR(1) comparison suggests. On the tested grid white noise certifies 7 of 11 failing amplitudes, AR(1) rejects 4 that pass, and the three strong surrogates each accept the two failing amplitudes at 0.65 and 0.70 rad/s. Once the spectrum is known a bin-matched synthesis replays it, so the value is in obtaining it. The full spectral shape is what closes it. A shallow 16-point anchor-sector spectrum places the boundary at 0.88 rad/s, a 32% envelope error only the full-depth computation catches. The vehicle runs the 22 µs monitor; quantum computes the disturbance model at certification time.

**Safety stage, scored on the statement's metric — episodes ending in a safe state at the operating point the certificate certifies (mean ± SD, 3 seeds × 400 episodes; the quantum component, the disturbance class, is the only thing varied):**

| disturbance class the certificate is validated against | certified boundary (rad/s) | episodes safe at that operating point, under the measured class |
|---|---|---|
| white noise (the field's default validation) | certifies every amplitude tested | **39.5 ± 2.2%** (at 1.3 rad/s) |
| AR(1) surrogate | certifies no amplitude | — (rejects every operating point) |
| AR(2) fitted to two lags | 0.714 | 91.8 ± 1.3% |
| single sinusoid at the dominant line | 0.701 | 92.5 ± 1.2% |
| narrowband around the dominant line | 0.792 | 87.5 ± 2.4% |
| **measured 128-qubit collective class** | **0.624** | **95.6 ± 0.3%** — the certification criterion, met |

A certificate validated against any of the five classical classes fails its own safety criterion at the point it certifies, or certifies nothing; validated against the measured class it meets the criterion, and the runtime wrap then takes the policy from 77.9% to 100% safe at that boundary. That is the ablation isolating the quantum component in the safety stage (`floor_vw_surrogates_seeds.json`).

![Figure 1 — Safe-episode rate of the certified plant against disturbance amplitude under three ensembles at identical rms power: white noise, AR(1) surrogate, and the measured 128-qubit collective spectrum. 3 seeds × 1,000 episodes per cell.](C:/quantum ai 2026/figs_v7/vw_boundary.png)

**Resource allocation.** A compression study, protocol frozen before the run, found INT4 already meets the bar classically (3.97× at 2.45% drop) and the mixed-precision space flat; the device solved its allocation objective exactly (9/9 instances). An OEM should not spend quantum budget there, so this entry concentrates on the two stages with measured headroom.

### Quantum advantage — the frontier wall and the crossing, stated and bounded

**The field's direction, and its wall.** Quantum reinforcement learning's advantage results (arXiv:2505.11862, 2501.16243, 2511.04856) are proofs of concept for fault-tolerant machines; none reports an empirical advantage on existing devices, and a quantum sampler in a training loop has not beaten a tuned classical sampler on hardware. Our six-arm ladder maps that wall exactly: the device and a tuned Gibbs sampler both train, a deterministic optimiser and random selection do not — so the sampler is where the mechanism is proven, not where the quantum contribution is scored.

**The crossing.** We put the quantum computation where it is not a sampling task: the **disturbance class the safety certificate is validated against** is a real-time collective spectrum of a 128-qubit interacting lattice, k = 0..15, 2,656 two-qubit gates at k = 8. It is adjudicated from both sides: our own sparse-Pauli Heisenberg-propagation attack on the shipped data — the referee protocol, engines validated to 10⁻¹⁶ against dense statevectors, every block run twice — reproduces the hardware to 0.4σ at k = 4 and within 1.4σ across five wavevectors at k = 6, then fails past k = 8 (operators exceeding 2×10⁸ terms); the hardware cells beyond that boundary stand at 8.9σ and 11.5σ with a 0.2σ preparation control, and the dominant line is reproduced on a second device. In the customer's units the crossing is the 0.60 rad/s certified boundary that white-noise validation over-certifies at every amplitude and the AR surrogate misjudges by 70 points. **Where it is decisive:** past k = 8, beyond the reach of exact Heisenberg propagation, whose operator passes 2×10⁸ terms at k = 8 (a measured boundary for the referee's method; tensor-network contraction of the same observable is the PoC's second comparator), with k ≤ 6 serving as the classically verified anchor of the same instrument; Phase 2 adds the fidelity certificate and the ion-trap replication.

**Scalability to industrial relevance.** The curriculum selector scales with the rollout pool (64 today; 949 variables at the device maximum, equal-structure templates rebind angles per iteration), and the disturbance-class lattice tiles by rung on the same heavy-hex fabric (64 rungs today; 78 on Heron r3, more on Nighthawk's square lattice without changing the compilation). Quantum work is train- and certification-time only, so fleet size does not enter the per-vehicle cost; the 22 µs monitor runs on existing controller silicon.

**Business value, bounded.** In an OEM's own units: (i) alignment cost — 35.4% fewer rollouts to target against the GRPO baseline means, at a declared cost per rollout *r* and *n* alignment runs per model release, a saving of ≈ 0.35 · 586 · *r* · *n*; (ii) certification — a disturbance envelope that is 32% wrong (shallow spectrum) or 70 points wrong at the operating point (AR surrogate) is either a field incident or an over-conservative envelope that leaves performance on the table; the measured class prices that margin at 0.60 rad/s exactly.

**What a successful PoC demonstrates.** The same curriculum selector inside the alignment loop of an accepted VLA backbone with a ≥10% rollout reduction at equal reward, 3 seeds, paired CI above zero against the strongest classical sampler at equal total cost; and a certified control envelope for that backbone's action output, validated against a hardware-computed disturbance class carrying a fidelity certificate.

## 5. Validation plan

**Milestone 1 (PoC weeks 1–6), the external-reference safety test, protocol frozen before the run:** an OpenVLA-OFT policy on LIBERO wrapped by six certificates, one validated against the measured class and five against the classical surrogates, all run on the statement's perturbation suite (3 seeds × 500 episodes); acceptance = the class-validated certificate has the lowest unsafe rate at its certified point, paired CI excluding zero against the best surrogate. Milestone 2: the curriculum selector on the same backbone, device and classical sampler against GRPO at matched budget, ≥ 10% fewer rollouts with CI above zero. Everything below is declared before the run, results reported either way: reward, robustness metric, seeds, total evaluation budget, disturbance amplitudes, tolerance. **Alignment:** OpenVLA-OFT (or π₀) on LIBERO, backbone/demonstrations/reward evaluator/initial checkpoints/training budget fixed; arms = native curriculum, tuned classical Gibbs, original encoding, random, best deterministic rule; samplers crossed with the same optimiser; paired tasks and independent seeds; scores = held-out shifted-task success and instruction consistency, constraint violations reported separately; report error at equal budget and cost to a common target; acceptance = paired CI above zero within the same full training/evaluation cost. **Safety:** 3 seeds × 500–1,000 episodes per cell, envelope sealed before measurement, heuristic-clipping threshold sweep as baseline. **Disturbance class:** re-read the 128-qubit series under the instrument (atlas layout, even-sector encoding, predicted pair T2, visibility inversion); attach a parity-syndrome fidelity bound; replicate the dominant line on an ion-trap device; report the Heisenberg-propagation attack (reproduces hardware to 0.4σ at k = 4, fails past k = 6) and a tensor-network contraction at matched accuracy. Every result: metrics and classical challengers frozen before the run, ≥3 seeds with mean ± SD, in-job controls, cloud job IDs, negatives kept.

## 6. Hybrid / cross-domain integration

Quantum work is train/validation-time only; Dirac-3 index sets are verified classically before use. Safety: candidates pass the grid-plus-Lipschitz certifier; the vehicle runs the 22 µs monitor. Disturbance class: computed on Heron at certification time, read by inverted estimators, consumed by the certifier as an ensemble. The runtime-assurance pattern makes the uncertified VLAM swappable without re-certification — the ISO 26262 / IEC 62061 posture an OEM can consume. Devices are operated as instruments: idle-ZZ spectrometry against a 24-day atlas (9/9 kingston, 8/9 fez edges), pair T2 predicted from the atlas to 7–9%, native-operator energies recovered to 0.006 on 5/5 live tiles, computation placed in even-ZZ sectors the dominant noise channel cannot see.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London), Dr. Ana Baroni MSc (domain specialist). GIC 2026 dual-track finalist. Programme record: 259 receipted QPU jobs, 12.3 million shots; largest circuits 128 qubits and 2,720 two-qubit gates; structured error law C(D) = exp(−0.00903D − 3.96×10⁻⁵D²) measured on-device (5.1× vs the white-noise extrapolation); five-way independent error rate 0.47%/gate; in-kind CHSH anchor S = 2.3604 (+35.9σ, replica +37.0σ). Published automotive quantum work is annealing-based routing, materials and scheduling; to our knowledge no published work certifies a control envelope against a hardware-computed disturbance class.

## 8. Scope, with treatment

(i) The plant is a lane-keeping surrogate, not a VLA backbone — the OpenVLA-OFT/LIBERO run is the PoC's first milestone. (ii) A fixed-temperature classical Gibbs sampler reaches the target in fewer rollouts than the device (298.7 vs 378.7) and is the arm behind the +9.3-point result; no quantum advantage in alignment is claimed — the device's result is the named-baseline gain, and the VLA milestone tests both samplers at matched budget. (iii) Compression headroom is flat beyond INT4; no quantum budget is spent there. (iv) The spectrum carries no fidelity certificate yet (Phase-2 deliverable); the strongest cheap surrogates misplace the boundary by 12–27%. (v) The envelope holds for the stated plant and classes; new conditions are re-validated.

---

### Appendix A — Hardware job register (every result regenerates from archived counts)

| measurement | machine | job id(s) |
|---|---|---|
| 128q jam-relaxation spectrum v2, k = 0..15, 33 × 32,768 shots | ibm_fez | `daa9pn4e74ec73akj9i0` |
| 156q real-time collective interface, exact-theorem anchors 0.9836 / 0.9806 | ibm_fez | `d9rm4j9dsedc73agrb70`; scout `d9rm47opdb6s73e53kqg` |
| Two-sided classical boundary series, 16 × 32,768 | ibm_kingston | `d9rdfb1dsedc73agh5ng` |
| Idle-ZZ atlas, 9/9 and 8/9 edges | ibm_kingston, ibm_fez | `da9l3t1qtnsc73d1nhd0`, `da9l9rkjbipc73ff0aqg`, `da9eoe6rbfbs73chiq0g`, `da9lqeerbfbs73chq63g` |
| Native-operator E₀ inverted, 5/5 tiles | ibm_kingston / ibm_fez | `da9vgsmrbfbs73ci44d0`, `da9vblkjbipc73ffaio0` |
| Directed-path traffic card — converges classically, kept as negative control | ibm_kingston | `results/trafficD_kingston_result.json` |
| CHSH in-kind anchor S = 2.3604 (+35.9σ), same-day replica +37.0σ | ibm_fez | GIC ledger `chsh_*_20260805_*.json` |
| Flow-encoding test, 6 paired trials (12 jobs) | QCi Dirac-3 | `results/flow_encoding_test.json` |
| RL curriculum arm C, seeds 21/22/23 (20 + 23 + 28 jobs) | QCi Dirac-3 | `6a76595c08442f441bbb5bd2` … `6a765f6b08442f441bbb5c18` — full list in the public repository |
| Surrogate-class ablation of the certificate, seed-level (3 × 400 episodes × 6 classes × 11 amplitudes) | CPU | — | `floor_vw_surrogates_seeds.json`, `floor_vw_surrogates.py` |
| Dirac-trained certificate, trackb seeds 21/22/23 (9 jobs each) | QCi Dirac-3 | `6a783a7c08442f441bbb5d25` … `6a783be408442f441bbb5d3f` — full list in the public repository |
| Safety, envelope, rank ablation, cert-gap sweeps | CPU (two machines) | `vw_rta_demo.json`, `vw_lyap_*.json`, `results/vw_cert_gap_*_v2.json` |

### Appendix B — Resource declaration and notes

Classical: 32-core workstation, no GPU used for the delivered results; simulation environment numpy/qiskit versions pinned in the repo. Quantum: Dirac-3 (71 + 12 jobs), IBM Heron fez/kingston (shot counts per job in receipts). Phase-2 VLA run: single 8-GPU node, GPU-hours to be declared. Deadline runway: Phase I closes 2026-09-15.

### Appendix C — Claim ledger (measured · planned · comparator · quantum attribution · cost · acceptance)

| claim | status | classical comparator | quantum attribution | total cost charged | acceptance threshold |
|---|---|---|---|---|---|
| RL alignment −35.4% rollouts vs GRPO baseline | measured (3 seeds, 71 jobs) | fixed-temperature Gibbs 298.7 (better than the device), random, optimiser | mechanism, not device | 71 jobs ≈ minutes QPU | ≥ 10% vs GRPO (met); vs best classical (not met) |
| +9.3 pts final quality at equal budget; ~2× budget equivalence | measured (classical Gibbs curriculum, two machines) | GRPO-32 | none — mechanism | CPU | replicated (met) |
| External-reference safety test on OpenVLA-OFT / LIBERO: class-validated certificate vs five surrogate-validated certificates on the statement's perturbation suite | planned (Phase-2 milestone 1, protocol frozen) | five classical disturbance classes | disturbance class (Heron, 128q) | 8-GPU node, 3 seeds × 500 episodes | lowest unsafe rate at the certified point, paired CI > 0 |
| Same selector on OpenVLA-OFT / LIBERO | planned (Phase-2 milestone 2) | fixed-temperature Gibbs, GRPO | sampler | same node | ≥ 10% rollouts-to-target, CI > 0 |
| Certified boundary 0.60 rad/s under measured class | measured (3 seeds × 1,000) | white-noise and AR(1) (weak); AR(2), dominant-line sinusoid and narrowband run as the strong floors — 0.714 / 0.701 / 0.792 vs 0.624 measured | 128q spectrum as certification input | 1 job × 33 circuits | no cheap surrogate reproduces the boundary; competent fits miss by 12–27% |
| 128q spectrum beyond classical attack (k ≥ 8) | measured; certificate planned | sparse-Pauli Heisenberg propagation, validated 10⁻¹⁶ vs dense (0.4σ at k=4; operator > 2×10⁸ terms at k ≥ 8); tensor-network contraction next | hardware | 33 × 32,768 shots | parity-syndrome fidelity bound + ion replication |
| 78 → 100% safe under certified wrap | measured (classical) | tuned clipping | none | CPU | envelope sealed before measurement |
| Certificate validated on a classical disturbance class fails its criterion at its certified point (39.5–92.5% safe, or certifies nothing); measured class meets it (95.6 ± 0.3%) | measured (3 seeds × 400 per cell) | white, AR(1), AR(2), dominant line, narrowband | disturbance class (Heron, 128q) | 33 circuits + CPU | ≥ 95.6% safe at the certified point — met only with the measured class |

