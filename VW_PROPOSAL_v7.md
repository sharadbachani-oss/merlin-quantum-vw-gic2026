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

**Scope, stated with its treatment.** (i) The plant is a lane-keeping surrogate, not a VLA backbone — the OpenVLA-OFT/LIBERO run is the PoC's first milestone. (ii) The tuned classical Gibbs arm beats the device on raw rollouts — the device claim is the named baseline, zero tuning and half the variance. (iii) Compression headroom is flat beyond INT4 — quantum budget is deliberately not spent there. (iv) The disturbance spectrum carries no fidelity certificate yet — the parity-syndrome bound and ion-trap replication are Phase-2 deliverables. (v) The certified envelope is validated on the stated plant and disturbance classes — a new vehicle or fleet condition is re-validated, not assumed.

Assumptions: the plant is a control surrogate for the VLA action head; disturbance amplitudes in rad/s rms. Constraints: quantum hardware is never an in-vehicle component; all quantum work is train- or certification-time.

## 4. Expected impact

**RL alignment — rollouts to target (seeds 21/22/23, mean ± SD):** A, GRPO-32 random (named baseline) 586.7 ± 98.8; **C, Dirac-3-16 untuned 378.7 ± 52.9 (−35.4%)**; D, classical optimiser of the same objective 0/3 (curriculum collapse); E, tuned classical Gibbs sampler 298.7 ± 106.9 — stronger on raw rollouts, at 2× the device's variance and with per-task temperature tuning the device does not need; C′, device on optimiser-grade encoding 0/3. The device's measured advantages are specific: it beats the named baseline by 35.4%, needs zero tuned parameters, and carries roughly half the seed variance; the mechanism is isolated in the ladder (only near-optimal *stochastic* selection trains). At equal budget the curriculum yields a safer driver: 76.4 ± 0.7% vs 67.1 ± 1.3% safe on the held-out stress suite (**+9.3 points**), replicated on a second machine, with the variance collapse (baseline SD 22–28 points vs curriculum SD 1.8) that de-risks the training run.

![Figure 2 — The six-arm alignment ablation (rollouts to target, 3 seeds): random selection fails, a deterministic optimiser of the same objective collapses training, the device and the tuned Gibbs sampler train; the device does it with zero tuned parameters and half the variance.](C:/quantum ai 2026/figs_v7/vw_ladder.png)

**Control safety:** policy alone 77.9 ± 25.0% safe at the certified boundary → **100.0 ± 0.0%** wrapped (3 seeds × 500 episodes per condition); guard idles under benign conditions (1–12% intervention); 22 µs per decision. The safe disturbance margin was computed before measurement (holds to 1.0 rad/s; collapse predicted in [1.2, 1.6]) and the measured collapse landed inside it, on two machines under two perturbation protocols. Versus tuned heuristic clipping: the envelope is known a priori, zero tuned parameters, and the V > 0, dV ≤ 0 evidence chain an ISO 26262 case consumes. Rank ablation: r = 3 interior-optimal at 0.462 ± 0.030 certified area; the TN certificate covers 1.57 ± 0.29× the best quadratic's area.

**Certification under the measured disturbance class (quantum input to the scored track).** Three ensembles at identical rms power, 3 seeds × 1,000 episodes per cell: against the **measured collective spectrum** the certified boundary is **0.60 rad/s rms** (95.6 ± 0.8% safe; 98.4 ± 0.3% at 0.50). White-noise validation grades 100.0 ± 0.0% safe at every amplitude through 1.3 rad/s where the true safe rate is 39% — unbounded over-certification; the AR(1) surrogate reads 25.2 ± 2.1% at the certified boundary itself. A shallow 16-point anchor-sector spectrum places the boundary at 0.88 rad/s, a 32% envelope error only the full-depth computation catches. The vehicle runs the 22 µs monitor; quantum computes the disturbance model at certification time.

![Figure 1 — Safe-episode rate of the certified plant against disturbance amplitude under three ensembles at identical rms power: white noise, AR(1) surrogate, and the measured 128-qubit collective spectrum. 3 seeds × 1,000 episodes per cell.](C:/quantum ai 2026/figs_v7/vw_boundary.png)

**Resource allocation.** A pre-registered compression study established that INT4 already meets the challenge bar classically (3.97× at 2.45% accuracy drop vs ≥2× at ≤5%) and that the remaining mixed-precision space is flat — the device solved its allocation objective exactly (DP gap 0.0, 9/9 instances) and beat random 5–25×; the finding tells an OEM where not to spend quantum budget, which is why this entry concentrates on the two stages with measured headroom.

### Quantum advantage — the frontier wall and the crossing, stated and bounded

**The field's direction, and its wall.** Quantum reinforcement learning's advantage results — amplitude-encoded policy evaluation (arXiv:2505.11862), quantum natural policy gradient at O(ε⁻¹·⁵) (arXiv:2501.16243), Boltzmann-machine sampling (arXiv:2511.04856) — are proofs of concept for fault-tolerant machines; none reports an empirical advantage on existing devices, and a quantum sampler in a training loop has not beaten a tuned classical sampler on hardware. Our six-arm ladder measures the same wall: the tuned Gibbs arm wins raw rollouts. Sampling is a thermal task, and thermal tasks are where classical methods keep pace.

**The crossing.** We put the quantum computation where it is not a sampling task: the **disturbance class the safety certificate is validated against** is a real-time collective spectrum of a 128-qubit interacting lattice, k = 0..15, 2,656 two-qubit gates at k = 8. It is adjudicated from both sides: our own tensor-network attack on the shipped data reproduces the hardware to 0.4σ at k = 4 and within 1.4σ across five wavevectors at k = 6, then fails past k = 8 (operators exceeding 2×10⁸ terms); the hardware cells beyond that boundary stand at 8.9σ and 11.5σ with a 0.2σ preparation control, and the dominant line is reproduced on a second device. In the customer's units the crossing is the 0.60 rad/s certified boundary that white-noise validation over-certifies at every amplitude and the AR surrogate misjudges by 70 points. **Bounded:** dormant where classical real-time methods still reach (k ≤ 6), decisive past k = 8; the fidelity certificate and ion-trap replication are Phase-2 gates; the sampler claim remains the named baseline, zero tuning and half the variance.

**What a successful PoC demonstrates.** The same curriculum selector inside the alignment loop of an accepted VLA backbone with a ≥10% rollout reduction at equal reward, 3 seeds, paired CI above zero against the strongest classical sampler at equal total cost; and a certified control envelope for that backbone's action output, validated against a hardware-computed disturbance class carrying a fidelity certificate.

## 5. Validation plan

Pre-registered: reward, robustness metric, seeds, total evaluation budget, disturbance amplitudes, tolerance. **Alignment:** OpenVLA-OFT (or π₀) on LIBERO, backbone/demonstrations/reward evaluator/initial checkpoints/training budget fixed; arms = native curriculum, tuned classical Gibbs, original encoding, random, best deterministic rule; samplers crossed with the same optimiser; paired tasks and independent seeds; scores = held-out shifted-task success and instruction consistency, constraint violations reported separately; report error at equal budget and cost to a common target; acceptance = paired CI above zero within the same full training/evaluation cost. **Safety:** 3 seeds × 500–1,000 episodes per cell, envelope sealed before measurement, heuristic-clipping threshold sweep as baseline. **Disturbance class:** re-read the 128-qubit series under the instrument (atlas layout, even-sector encoding, predicted pair T2, visibility inversion); attach a parity-syndrome fidelity bound; replicate the dominant line on an ion-trap device; report the commissioned tensor-network attack (reproduces hardware to 0.4σ at k = 4, fails past k = 6) and a Pauli-path run at matched accuracy. Every result: frozen pre-registration, ≥3 seeds with mean ± SD, in-job controls, cloud job IDs, negatives kept.

## 6. Hybrid / cross-domain integration

Quantum work is train/validation-time only. Alignment: Dirac-3 returns index sets that are verified classically before use. Safety: certificate candidates pass the deterministic grid-plus-Lipschitz certifier; the vehicle runs the 22 µs monitor and fallback. Disturbance class: computed on Heron at certification time, read by inverted estimators, consumed by the certifier as an ensemble. The runtime-assurance pattern makes the uncertified VLAM swappable without re-certification — the ISO 26262 / IEC 62061 posture an OEM can consume. Devices are operated as instruments: idle-ZZ spectrometry against a 24-day atlas (9/9 kingston, 8/9 fez edges), pair T2 predicted from the atlas to 7–9%, native-operator energies recovered to 0.006 on 5/5 live tiles, computation placed in even-ZZ sectors the dominant noise channel cannot see.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London), Dr. Ana Baroni MSc (domain specialist). GIC 2026 dual-track finalist. Programme record: 259 receipted QPU jobs, 12.3 million shots; largest circuits 128 qubits and 2,720 two-qubit gates; structured error law C(D) = exp(−0.00903D − 3.96×10⁻⁵D²) measured on-device (5.1× vs the white-noise extrapolation); five-way independent error rate 0.47%/gate; in-kind CHSH anchor S = 2.3604 (+35.9σ, replica +37.0σ). Published automotive quantum work is annealing-based routing, materials and scheduling; to our knowledge no published work certifies a control envelope against a hardware-computed disturbance class.

---

### Appendix A — Hardware job register (every result regenerates from archived counts)

| measurement | machine | job id(s) |
|---|---|---|
| 128q jam-relaxation spectrum v2, k = 0..15, 33 × 32,768 shots | ibm_fez | `daa9pn4e74ec73akj9i0` |
| 156q real-time collective interface, exact-theorem anchors 0.9836 / 0.9806 | ibm_fez | `d9rm4j9dsedc73agrb70`; scout `d9rm47opdb6s73e53kqg` |
| Two-sided classical boundary series, 16 × 32,768 | ibm_kingston | `d9rdfb1dsedc73agh5ng` |
| Idle-ZZ atlas, 9/9 edges | ibm_kingston | `da9l3t1qtnsc73d1nhd0`, `da9l9rkjbipc73ff0aqg` |
| Idle-ZZ atlas, 8/9 edges | ibm_fez | `da9eoe6rbfbs73chiq0g`, `da9lqeerbfbs73chq63g` |
| Native-operator E₀ inverted, 5/5 tiles | ibm_kingston / ibm_fez | `da9vgsmrbfbs73ci44d0`, `da9vblkjbipc73ffaio0` |
| Directed-path traffic card — converges classically, kept as negative control | ibm_kingston | `results/trafficD_kingston_result.json` |
| CHSH in-kind anchor S = 2.3604 (+35.9σ), same-day replica +37.0σ | ibm_fez | GIC ledger `chsh_*_20260805_*.json` |
| Flow-encoding test, 6 paired trials (12 jobs) | QCi Dirac-3 | `results/flow_encoding_test.json` |
| RL curriculum arm C, vw_rl_C_s21.json (20 jobs) | QCi Dirac-3 | `6a76595c08442f441bbb5bd2`, `6a76597408442f441bbb5bd3`, `6a76599008442f441bbb5bd4`, `6a7659a608442f441bbb5bd5`, `6a7659bc08442f441bbb5bd6`, `6a7659de08442f441bbb5bd7`, `6a7659fc08442f441bbb5bd8`, `6a765a1a08442f441bbb5bd9`, `6a765a2e08442f441bbb5bda`, `6a765a4608442f441bbb5bdb`, `6a765a6208442f441bbb5bdc`, `6a765a7708442f441bbb5bdd`, `6a765a8d08442f441bbb5bde`, `6a765aa008442f441bbb5bdf`, `6a765ab508442f441bbb5be0`, `6a765acb08442f441bbb5be1`, `6a765ae208442f441bbb5be2`, `6a765af708442f441bbb5be3`, `6a765b0f08442f441bbb5be4`, `6a765b2208442f441bbb5be5` |
| RL curriculum arm C, vw_rl_C_s22.json (23 jobs) | QCi Dirac-3 | `6a765b8708442f441bbb5be6`, `6a765b9d08442f441bbb5be7`, `6a765bb208442f441bbb5be8`, `6a765bc608442f441bbb5be9`, `6a765bda08442f441bbb5bea`, `6a765bee08442f441bbb5beb`, `6a765c0408442f441bbb5bec`, `6a765c1608442f441bbb5bed`, `6a765c2b08442f441bbb5bee`, `6a765c3e08442f441bbb5bef`, `6a765c5408442f441bbb5bf0`, `6a765c6808442f441bbb5bf1`, `6a765c7c08442f441bbb5bf2`, `6a765c8e08442f441bbb5bf3`, `6a765ca108442f441bbb5bf4`, `6a765cba08442f441bbb5bf5`, `6a765ccd08442f441bbb5bf6`, `6a765ce208442f441bbb5bf7`, `6a765cf508442f441bbb5bf8`, `6a765d0908442f441bbb5bf9`, `6a765d1d08442f441bbb5bfa`, `6a765d3108442f441bbb5bfb`, `6a765d4508442f441bbb5bfc` |
| RL curriculum arm C, vw_rl_C_s23.json (28 jobs) | QCi Dirac-3 | `6a765d5b08442f441bbb5bfd`, `6a765d6f08442f441bbb5bfe`, `6a765d8308442f441bbb5bff`, `6a765d9608442f441bbb5c00`, `6a765da908442f441bbb5c01`, `6a765dbc08442f441bbb5c02`, `6a765dd508442f441bbb5c03`, `6a765dee08442f441bbb5c04`, `6a765e0208442f441bbb5c05`, `6a765e1608442f441bbb5c06`, `6a765e2908442f441bbb5c07`, `6a765e3d08442f441bbb5c08`, `6a765e5008442f441bbb5c09`, `6a765e6208442f441bbb5c0a`, `6a765e7508442f441bbb5c0b`, `6a765e8708442f441bbb5c0c`, `6a765e9a08442f441bbb5c0d`, `6a765eac08442f441bbb5c0e`, `6a765ebf08442f441bbb5c0f`, `6a765ed308442f441bbb5c10`, `6a765ee708442f441bbb5c11`, `6a765efa08442f441bbb5c12`, `6a765f0e08442f441bbb5c13`, `6a765f2108442f441bbb5c14`, `6a765f3408442f441bbb5c15`, `6a765f4608442f441bbb5c16`, `6a765f5908442f441bbb5c17`, `6a765f6b08442f441bbb5c18` |
| Dirac-trained certificate, trackb_seed21.json (9 jobs) | QCi Dirac-3 | `6a783b7f08442f441bbb5d37`, `6a783b8b08442f441bbb5d38`, `6a783b9708442f441bbb5d39`, `6a783ba408442f441bbb5d3a`, `6a783bb108442f441bbb5d3b`, `6a783bbd08442f441bbb5d3c`, `6a783bca08442f441bbb5d3d`, `6a783bd708442f441bbb5d3e`, `6a783be408442f441bbb5d3f` |
| Dirac-trained certificate, trackb_seed22.json (9 jobs) | QCi Dirac-3 | `6a783a7c08442f441bbb5d25`, `6a783a8708442f441bbb5d26`, `6a783a9108442f441bbb5d27`, `6a783a9c08442f441bbb5d28`, `6a783aa708442f441bbb5d29`, `6a783ab108442f441bbb5d2a`, `6a783abc08442f441bbb5d2b`, `6a783ac608442f441bbb5d2c`, `6a783ad108442f441bbb5d2d` |
| Dirac-trained certificate, trackb_seed23.json (9 jobs) | QCi Dirac-3 | `6a783adf08442f441bbb5d2e`, `6a783aea08442f441bbb5d2f`, `6a783af508442f441bbb5d30`, `6a783b0008442f441bbb5d31`, `6a783b0b08442f441bbb5d32`, `6a783b1608442f441bbb5d33`, `6a783b2108442f441bbb5d34`, `6a783b2c08442f441bbb5d35`, `6a783b3608442f441bbb5d36` |
| Safety, envelope, rank ablation, cert-gap sweeps | CPU (two machines) | `vw_rta_demo.json`, `vw_lyap_*.json`, `results/vw_cert_gap_*_v2.json` |

### Appendix B — Resource declaration and notes

Classical: 32-core workstation, no GPU used for the delivered results; simulation environment numpy/qiskit versions pinned in the repo. Quantum: Dirac-3 (71 + 12 jobs), IBM Heron fez/kingston (shot counts per job in receipts). Phase-2 VLA run: single 8-GPU node, GPU-hours to be declared. Deadline runway: Phase I closes 2026-09-15.
