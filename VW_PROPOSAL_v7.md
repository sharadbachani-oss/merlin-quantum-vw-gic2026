# Quantum-in-the-loop VLAM alignment and certified control safety for in-vehicle compute

**Global Quantum + AI Challenge 2026 — Volkswagen Enterprise Challenge · Phase 1 Concept Proposal · Tracks: RL alignment (primary) + control safety · Team Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · v7.0 · 2026-09-12**

**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026**

---

## 1. Problem framing

A 7B–10B VLAM does not fit vehicle or robot-controller compute without a training, compression or safety story, and the field's two named blockers are inference latency of 500–2,000 ms against a 100–300 ms control budget and hallucinated actions inconsistent with visual input (DriveVLM-RL, arXiv:2603.18315; VLAM survey, arXiv:2512.16760). Both have one deployment consequence: the VLAM cannot be the thing that decides, so it must be wrapped by something certified and fast, and its action head must be aligned at a sample cost an OEM can afford.

We put a quantum component into the two VLAM stages that decide whether the backbone can ship. **Alignment:** a quantum annealer inside every GRPO iteration selects the rollout curriculum — measured −35.4% rollouts to target versus the named GRPO baseline, three seeds. **Safety:** a certified runtime-assurance monitor (22 µs, 4,500× inside the budget) wrapping an untrusted policy, whose certified envelope is validated against a **disturbance class computed on quantum hardware** — the real-frequency collective spectrum of an interacting mobility lattice at 128 qubits. Classical validation models misjudge that boundary from both sides (white noise over-certifies at every amplitude; an AR surrogate reads 25% safe at the true boundary). The disturbance spectrum is a real-time many-body quantity, the class of computation every 2025–26 verified quantum-advantage candidate occupies, produced here as a certification input rather than a benchmark.

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
| IBM Heron | Startup Program (applied); the spectrum job exists (`daa9pn4e74ec73akj9i0`); re-flights ≤ 33 circuits × 32,768 shots |
| VLA backbone run (Phase 2) | OpenVLA-OFT / LIBERO on a single 8-GPU node; 3 seeds; GPU-hours declared in the resource declaration |
| Classical | 32-core workstation for the tensor-network certifier and the adversarial classical attack |

Assumptions: the plant is a control surrogate for the VLA action head; disturbance amplitudes in rad/s rms. Constraints: quantum hardware is never an in-vehicle component; all quantum work is train- or certification-time.

## 4. Expected impact

**RL alignment — rollouts to target (seeds 21/22/23, mean ± SD):** A, GRPO-32 random (named baseline) 586.7 ± 98.8; **C, Dirac-3-16 untuned 378.7 ± 52.9 (−35.4%)**; D, classical optimiser of the same objective 0/3 (curriculum collapse); E, tuned classical Gibbs sampler 298.7 ± 106.9 — stronger on raw rollouts, at 2× the device's variance and with per-task temperature tuning the device does not need; C′, device on optimiser-grade encoding 0/3. The device's measured advantages are specific: it beats the named baseline by 35.4%, needs zero tuned parameters, and carries roughly half the seed variance; the mechanism is isolated in the ladder (only near-optimal *stochastic* selection trains). At equal budget the curriculum yields a safer driver: 76.4 ± 0.7% vs 67.1 ± 1.3% safe on the held-out stress suite (**+9.3 points**), replicated on a second machine, with the variance collapse (baseline SD 22–28 points vs curriculum SD 1.8) that de-risks the training run.

**Control safety:** policy alone 77.9 ± 25.0% safe at the certified boundary → **100.0 ± 0.0%** wrapped (3 seeds × 500 episodes per condition); guard idles under benign conditions (1–12% intervention); 22 µs per decision. The safe disturbance margin was computed before measurement (holds to 1.0 rad/s; collapse predicted in [1.2, 1.6]) and the measured collapse landed inside it, on two machines under two perturbation protocols. Versus tuned heuristic clipping: the envelope is known a priori, zero tuned parameters, and the V > 0, dV ≤ 0 evidence chain an ISO 26262 case consumes. Rank ablation: r = 3 interior-optimal at 0.462 ± 0.030 certified area; the TN certificate covers 1.57 ± 0.29× the best quadratic's area.

**Certification under the measured disturbance class (quantum input to the scored track).** Three ensembles at identical rms power, 3 seeds × 1,000 episodes per cell: against the **measured collective spectrum** the certified boundary is **0.60 rad/s rms** (95.6 ± 0.8% safe; 98.4 ± 0.3% at 0.50). White-noise validation grades 100.0 ± 0.0% safe at every amplitude through 1.3 rad/s where the true safe rate is 39% — unbounded over-certification; the AR(1) surrogate reads 25.2 ± 2.1% at the certified boundary itself. A shallow 16-point anchor-sector spectrum places the boundary at 0.88 rad/s, a 32% envelope error only the full-depth computation catches. The vehicle runs the 22 µs monitor; quantum computes the disturbance model at certification time.

![Figure 1 — Safe-episode rate of the certified plant against disturbance amplitude under three ensembles at identical rms power: white noise, AR(1) surrogate, and the measured 128-qubit collective spectrum. 3 seeds × 1,000 episodes per cell.](C:/quantum ai 2026/figs_v7/vw_boundary.png)

**Resource allocation.** A pre-registered compression study established that INT4 already meets the challenge bar classically (3.97× at 2.45% accuracy drop vs ≥2× at ≤5%) and that the remaining mixed-precision space is flat — the device solved its allocation objective exactly (DP gap 0.0, 9/9 instances) and beat random 5–25×; the finding tells an OEM where not to spend quantum budget, which is why this entry concentrates on the two stages with measured headroom.

**What a successful PoC demonstrates.** The same curriculum selector inside the alignment loop of an accepted VLA backbone with a ≥10% rollout reduction at equal reward, 3 seeds, paired CI above zero against the strongest classical sampler at equal total cost; and a certified control envelope for that backbone's action output, validated against a hardware-computed disturbance class carrying a fidelity certificate.

## 5. Validation plan

Pre-registered: reward, robustness metric, seeds, total evaluation budget, disturbance amplitudes, tolerance. **Alignment:** OpenVLA-OFT (or π₀) on LIBERO, backbone/demonstrations/reward evaluator/initial checkpoints/training budget fixed; arms = native curriculum, tuned classical Gibbs, original encoding, random, best deterministic rule; samplers crossed with the same optimiser; paired tasks and independent seeds; scores = held-out shifted-task success and instruction consistency, constraint violations reported separately; report error at equal budget and cost to a common target; acceptance = paired CI above zero within the same full training/evaluation cost. **Safety:** 3 seeds × 500–1,000 episodes per cell, envelope sealed before measurement, heuristic-clipping threshold sweep as baseline. **Disturbance class:** re-read the 128-qubit series under the instrument (atlas layout, even-sector encoding, predicted pair T2, visibility inversion); attach a parity-syndrome fidelity bound; replicate the dominant line on an ion-trap device; report the commissioned tensor-network attack (reproduces hardware to 0.4σ at k = 4, fails past k = 6) and a Pauli-path run at matched accuracy. Every result: frozen pre-registration, ≥3 seeds with mean ± SD, in-job controls, cloud job IDs, negatives kept.

## 6. Hybrid / cross-domain integration

Quantum work is train/validation-time only. Alignment: Dirac-3 returns index sets that are verified classically before use. Safety: certificate candidates pass the deterministic grid-plus-Lipschitz certifier; the vehicle runs the 22 µs monitor and fallback. Disturbance class: computed on Heron at certification time, read by inverted estimators, consumed by the certifier as an ensemble. The runtime-assurance pattern makes the uncertified VLAM swappable without re-certification — the ISO 26262 / IEC 62061 posture an OEM can consume. Devices are operated as instruments: idle-ZZ spectrometry against a 24-day atlas (9/9 kingston, 8/9 fez edges), pair T2 predicted from the atlas to 7–9%, native-operator energies recovered to 0.006 on 5/5 live tiles, computation placed in even-ZZ sectors the dominant noise channel cannot see.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London), Dr. Ana Baroni MSc (domain specialist). GIC 2026 dual-track finalist. Programme record: 259 receipted QPU jobs, 12.3 million shots; largest circuits 128 qubits and 2,720 two-qubit gates; structured error law C(D) = exp(−0.00903D − 3.96×10⁻⁵D²) measured on-device (5.1× vs the white-noise extrapolation); five-way independent error rate 0.47%/gate; in-kind CHSH anchor S = 2.3604 (+35.9σ, replica +37.0σ). Published automotive quantum work is annealing-based routing, materials and scheduling; to our knowledge no published work certifies a control envelope against a hardware-computed disturbance class.

---

### Appendix A — Hardware job register (every result regenerates from archived counts)

| measurement | machine | job id(s) | receipt |
|---|---|---|---|
| RL curriculum, arm C, 3 seeds (71 jobs) | QCi Dirac-3 | 71 ids in `vw_rl_C_s{21,22,23}.json` | `vw_rl_{A..E}_s*.json` |
| Dirac-trained certificate seeds | QCi Dirac-3 | 27 ids in `results/trackb_seed{21,22,23}.json` | same |
| Flow-encoding test, 6 paired trials | QCi Dirac-3 | 12 paired ids | `results/flow_encoding_test.json` |
| 128q jam-relaxation spectrum v2, k = 0..15, 33 × 32,768 shots | ibm_fez | `daa9pn4e74ec73akj9i0` | `results/vw_jam_relaxation_spectrum_v2.json` |
| 156q real-time collective interface, exact-theorem anchors 0.9836 / 0.9806 | ibm_fez | `d9rm4j9dsedc73agrb70`; scout `d9rm47opdb6s73e53kqg` | `results/a1p_result_20260808_200224.json` |
| Two-sided classical boundary series, 16 × 32,768 | ibm_kingston | `d9rdfb1dsedc73agh5ng` | `A1_FINAL_VERDICT.md` |
| Idle-ZZ atlas, 9/9 edges | ibm_kingston | `da9l3t1qtnsc73d1nhd0`, `da9l9rkjbipc73ff0aqg` | `results/npoint2_result_20260830_014105.json` |
| Idle-ZZ atlas, 8/9 edges | ibm_fez | `da9eoe6rbfbs73chiq0g`, `da9lqeerbfbs73chq63g` | `results/npoint2_result_20260829_182700.json` |
| Native-operator E₀ inverted, 5/5 tiles | ibm_kingston / ibm_fez | `da9vgsmrbfbs73ci44d0`, `da9vblkjbipc73ffaio0` | `results/ncomp_regrade_*.json`, `results/nc1_energy_inverted.json` |
| Directed-path traffic card — converges classically, kept as negative control | ibm_kingston | receipt in file | `results/trafficD_kingston_result.json` |
| CHSH in-kind anchor S = 2.3604 (+35.9σ), same-day replica +37.0σ | ibm_fez | GIC ledger `chsh_*_20260805_*.json` | — |
| Safety, envelope, rank ablation, cert-gap sweeps | CPU (two machines) | — | `vw_rta_demo.json`, `vw_lyap_*.json`, `results/vw_cert_gap_*_v2.json` |

### Appendix B — Resource declaration and notes

Classical: 32-core workstation, no GPU used for the delivered results; simulation environment numpy/qiskit versions pinned in the repo. Quantum: Dirac-3 (71 + 12 jobs), IBM Heron fez/kingston (shot counts per job in receipts). Phase-2 VLA run: single 8-GPU node, GPU-hours to be declared. Deadline runway: Phase I closes 2026-09-15.
