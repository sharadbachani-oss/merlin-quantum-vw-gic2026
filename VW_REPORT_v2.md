# Quantum-in-the-Loop VLAM Alignment and Certified Control Safety for In-Vehicle Compute

**Global Quantum + AI Challenge 2026 — Enterprise Challenge (Volkswagen)**
**Tracks: RL Alignment (primary scored) + Safety (secondary scored) · Context: VLAM-grade perception/reasoning on vehicle and robot controllers**
**Team: Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · Report v2.2, 2026-08-30**
**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026**

---

## 0. Challenge demand → this package

VLAMs at 7B–10B are the planned backbone of autonomous driving and robotics,
and they do not fit in-vehicle or robot-controller compute without a
training, compression, or safety story. This entry puts a
**quantum-enhanced / quantum-inspired component into two VLAM stages**
that decide whether that backbone can ship: **RL alignment of the action
policy** (sample cost) and **formal assurance of the control output**
(ISO 26262 / IEC 62061). The 7B weights stay a swappable black box —
that is the OEM requirement, not a dodge.

| Challenge demand | This package |
|---|---|
| VLAM stage (required) | (1) post-training **RL alignment** of the action head; (2) **runtime safety** on the control-output stage |
| Bottleneck — model footprint 7B–10B | Targeted; kill-switch fired. Classical GPTQ-INT4 already clears ≥2× @ ≤5%. Not scored (§7) |
| Bottleneck — attention O(N²) | Not targeted. Disclosed. |
| Bottleneck — RL alignment cost | **−35.4%** rollouts vs named GRPO baseline (bar ≥10%); **+9.3 pt** safer at equal budget |
| Bottleneck — control safety | Uncertified driver **77.9% → 100%** safe; envelope sealed *before* measurement; **22 µs** (4,500× under 100 ms) |
| ≥3 runs, mean ± SD; ablation isolating Q/QI | 3 seeds everywhere; six-arm RL ladder + rank ablation (§2–§4) |
| 4–8 page report; public repo + README | This document + https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026 |

| Claim (§) | Evidence artifact | Receipt |
|---|---|---|
| RL: 35.4% under GRPO budget, 3/3 seeds (§2) | `vw_rl_{A,B,C,D,E}_s{21,22,23}.json` | 71 Dirac-3 job IDs in JSONs |
| RL: +9.3-pt robustness at equal budget; ~2× budget-equivalence, two-box (§2) | `rl_finalquality_classical.json` + `RL_FINALQUALITY_BOX.md` | replicated, independent machine |
| Safety: 78→100% under RTA; envelope predicted then confirmed (§3) | `vw_rta_demo.json`, `vw_lyap_bias_sweep.json` | 3 seeds × 500 episodes ×2 machines |
| Flow encoding of the RL quantum stage: −26%, 3.5× tighter (§5) | `flow_encoding_test.json` | 12 paired Dirac-3 jobs, frozen prediction |
| Device-layer control of that quantum stage: ζ, T2, inverted E0 (§5) | `ncomp_regrade_*`, `t2_bridge_result.json`, `npoint2_result_*` | fez + kingston, banked |
| Supporting: S(q,ω) jam-relaxation, two-sided classical boundary (annex) | `vw_jam_relaxation_spectrum.json`, `A1_FINAL_VERDICT.md` | fez `d9rm4j9dsedc73agrb70`; kingston `d9rdfb1dsedc73agh5ng` |

Every negative result carries the same receipt class (§7).

## 1. VLAM integration — the component, not a cloud VLAM

A VLAM is perception → reasoning → **action**. The 7B–10B backbone is
what OEMs cannot train or host on the vehicle. Two stages around that
backbone are where a quantum / QI component can change the product
without moving inference to a GPU cluster:

1. **Alignment (scored).** After (or instead of) expensive preference
   fine-tuning, the action policy is aligned by RL under domain
   randomization. We put a QCi Dirac-3 annealer *inside every GRPO
   iteration* as the curriculum selector. Quantum work is
   **train-time only**.
2. **Control safety (scored).** The VLAM's action is an uncertified
   command. A tensor-network Lyapunov monitor + verified fallback wraps
   that command at **22 µs**, 4,500× inside the **≤100 ms** inference
   budget. Only monitor + fallback are certified; the VLAM stays
   replaceable without re-certification — the ISO 26262 / IEC 62061
   posture an OEM can consume.

Plant for both tracks: kinematic-bicycle lane-keeping (lateral / heading
error, |δ| ≤ 0.25, v = 25 m/s) under crosswind and impulses — the
**control substrate a VLAM commands**, not a 7B forward pass. Scope is
declared: we do not claim a compressed 7B checkpoint or an O(N²)
attention replacement. We claim the two stages that make a VLAM
deployable on a vehicle or robot controller.

## 2. RL-alignment track (scored): quantum-in-the-loop training

**Method.** Each GRPO iteration submits the 64-candidate rollout pool
(start × disturbance metadata) to a QCi Dirac-3 quantum annealer, which
selects the 16-rollout curriculum (frontier + diversity objective) before
any simulation is spent. Task: lane-keeping alignment under domain
randomization (kinematic bicycle, v=25 m/s regime, crosswind ±1.2 rad/s,
impulses); linear-Gaussian policy from zero initialization; target = within
5% of nominal-controller evaluation reward.

**Budget result (rollouts-to-target, seeds 21/22/23):**

| arm | selection | mean ± SD |
|---|---|---|
| A — GRPO-32, random (**named baseline**) | 586.7 ± 98.8 |
| B — random-16 (size control) | unreliable (1/3) |
| **C — Dirac-3-16, untuned** | **378.7 ± 52.9 (−35.4%)** |
| D — classical optimizer, same objective | fails 0/3 (curriculum collapse) |
| E — classical Gibbs sampler, tuned T | 298.7 ± 106.9 (parity, disclosed) |
| C′ — device on optimizer-grade encoding | fails 0/3 (confirms D's mechanism) |

**Product result (the stronger claim): at *equal* budget the curriculum
produces a measurably safer driver.** Held-out stress suite (bias
0.9–1.35 rad/s + impulses): curriculum-trained policies 76.4 ± 0.7% safe
vs baseline 67.1 ± 1.3% (**+9.3 points**). Independently replicated on a
second machine (+7.2 pts seeds 21–23; +18.4 pts seeds 21–25), with the
budget–quality Pareto: **curriculum at half the budget matches or beats the
baseline at every rung** (E@320 ≥ A@640; E@640 > A@1280 ≈ 2× budget
equivalence), and a variance collapse (baseline SD 22–28 pts — seeds that
never learn — vs curriculum SD 1.8): **curriculum selection de-risks the
training run.** Robust across stress windows (+16 to +21 pts).

## 3. Safety track (scored): certified runtime assurance

**Method.** Tensor-network Lyapunov monitor V(x)=‖L·φ(x)‖² (3-level
tensor-product features, rank 3 of 9), grid+Lipschitz certified, wrapping
an uncertified learned policy (behavior-cloned MLP with covariate shift and
100 ms delay): policy drives; verified fallback engages at V > 0.75c with
hysteresis. Only monitor + fallback face certification — the learned
driver (stand-in for the VLAM action head) stays a black box, swappable
without re-certification. The same wrap applies to a robot controller
under IEC 62061: certified monitor, uncertified policy, deterministic
fallback.

**Results (3 seeds × 500 episodes/condition):** policy alone
**77.9 ± 25.0%** safe at the certified boundary → **100.0 ± 0.0%** wrapped;
benign conditions: guard idles (1–12% intervention). Monitor overhead
**22 µs/decision** (4,500× under the 100 ms budget). **Sealed-prediction
envelope:** the safe disturbance margin was computed *before* measurement
(holds to 1.0 rad/s; collapse predicted in [1.2, 1.6]) and the measured
collapse landed inside the predicted interval — on two machines, under two
perturbation protocols.

**Named-baseline comparison, disclosed in full:** well-tuned heuristic
clipping ties the certificate on raw %-safe at every disturbance level we
measured (both collapse where the shared fallback saturates; threshold
sweep included). The certificate's measured wins: the envelope is known
*a priori* (clipping's margin is discoverable only by crashing); zero
tuned parameters (trigger = computed c); and the formal V>0, dV≤0 evidence
chain — the artifact an ISO 26262 / IEC 62061 case consumes — which no
heuristic produces at any tuning.

**Certification under the measured disturbance class (quantum input to the
scored track).** A certified envelope is only as good as the disturbance
model it is validated against. We evaluated the certified plant under
three disturbance ensembles at *identical rms power*: white noise, an
AR(1) surrogate matched to variance and autocorrelation (the standard
classical validation models), and the **measured collective disturbance
spectrum** — the real-frequency jam-relaxation object computed on quantum
hardware from receipted flights (annex; dominant collective line 0.0625
cycles/step). Result (3 seeds × 1,000 episodes per cell): the two
classical validation models misjudge safety by **25–70 points in opposite
directions** at matched power. White-noise validation grades 100.0 ± 0.0%
safe at every amplitude — including 1.3 rad/s rms, where the true safe
rate is 34% (unbounded over-certification). The AR surrogate never
reaches 95% safe even at 0.7 rad/s (41.1 ± 2.5%), so surrogate-based
certification surrenders **≥3× of the legitimate amplitude envelope
(≥9× in power)**. Against the measured spectrum the certified boundary
is **0.88 rad/s rms** (98.7 ± 0.0% at 0.80; 93.4 ± 0.5% at 0.90) —
recovered operating envelope at equal certified safety, robust across
the step-to-seconds mapping (swept 1–2 s/step). The vehicle-side story
is unchanged: quantum computes the disturbance model at
certification time; the car runs the 22 µs monitor. Receipt:
`results/vw_cert_gap_score.json`.

## 4. Mandatory ablation — isolating the quantum / QI component

**RL (six-arm ladder).** Random fails. A classical *optimizer* of the
identical objective breaks training (0/3). A tuned classical *sampler*
reaches parity (disclosed, sized honestly). The device delivers the
winning ingredient — stochastic near-optimal curriculum sampling —
natively, with zero tuned parameters and the tightest seed variance,
executing in every iteration of the scored run (71 receipted jobs). Arm
C′ shows that *improving* the device's optimization quality reproduces
the classical optimizer's failure — measured proof that the metric
rewards sampling, which bounds every entrant, classical or quantum, at
sampling-parity on this track.

**Safety (rank axis).** r=2 certifies nothing (expressivity floor);
**r=3 interior-optimal, 0.462 ± 0.030 certified area**; full rank
unstable (± 0.110 with a collapsed seed); in the saturation-bound
regime the TN certificate covers **1.57 ± 0.29×** the best quadratic's
area (all seeds ≥ 1.39). Scope: the r=3 optimum is a measured d=2
result; at d=3 the useful rank grows (disclosed, §7). Removing the
low-rank tensor structure costs certified area *and* reliability.

## 5. Why the VLAM-loop quantum component is a controlled instrument

The RL stage is not a black-box annealer call. The same framework that
placed as a GIC 2026 dual-track finalist predicted, before running, that
encoding the curriculum QUBO on the annealer's native flow manifold (no
penalty walls) beats naive binary encoding. Measured on the identical
selection objective, 6 paired trials, 12 receipted jobs: **flow encoding
−26% better objective, 3.5× tighter spread** — frozen prediction
confirmed. That is the design rule behind the 71-job curriculum and
the method any OEM optimization on analog quantum hardware should use.

The same devices that will carry Phase-II mobility spectra are now
**operated as an instrument, not a hope.** Idle ZZ (ζ) is a spectrometer
(two independent readouts; 9/9 kingston and 8/9 fez edges vs a 24-day
atlas). Encoded T2 is predicted from that atlas (Γ_Bell = Γ_a+Γ_b;
median error 7% / 9% on 18 edges) — a jam-relaxation or curriculum
tile's coherence budget is a number *before* the first shot. Energy and
spectra are visibility-inverted, not raw-summed: the native 64-cell
operator recovers E0 on **5/5 live tiles** (best −2.523 vs −2.518,
**−0.006**); filler edges with L0 < 0.90 were a layout error and are
no longer graded. Computation sits in even-ZZ sectors the noise cannot
see (logical X_L 9/9; GHZ-4 protection passed).

**What this changes for Volkswagen.** The scored RL number does not
move — it is already receipted. Its *provenance* is now named: put the
VLAM-loop problem on the machine's native manifold. The scored safety
envelope stays the ISO artifact; Phase II feeds it a *derived*
disturbance model (collective S(q,ω) of interacting traffic, read with
inverted estimators) instead of a grid heuristic on a 2-state plant.
That is full framework capability applied to the challenge's two VLAM
stages, not a second supremacy claim.

Receipts (banked, no copy flights): `npoint2_result_20260830_014105.json`,
`npoint2_result_20260829_182700.json`, `t2_bridge_result.json`,
`ncomp_regrade_20260830_122917.json`, `ncomp_regrade_20260830_123936.json`,
`nc1_energy_inverted.json`.

## 6. Deployment architecture

Quantum work is train/validation-time only — vehicle inference latency
is the classical monitor: **22 µs**, 4,500× under the **≤100 ms**
requirement. The runtime-assurance pattern makes the uncertified VLAM
swappable without re-certification. Every quantum output is classically
verified before use: selected curricula are index sets; certificate
candidates pass the deterministic grid+Lipschitz certifier. Quantum
hardware is never a trusted in-vehicle component — the correct posture
for an ISO 26262 / IEC 62061 pipeline.

## 7. Negative results and limitations (disclosed with the same receipts)

- **Compression / footprint bottleneck:** abandoned by pre-registered
  kill-switch — GPTQ-INT4 clears the ≥2× @ ≤5% bar *classically alone*
  (3.97× at 2.45% drop); the {4,8} allocation is flat and the {2,4}
  frontier collapses immediately past 4× (15 accuracy points for +0.09×)
  — super-additivity unharvestable by any allocator, quantum or
  classical. The device solved its cycle-1 linear objective exactly
  (DP gap 0.0, 9/9) and beat random 5–25× on real accuracy; the honest
  verdict is that the problem, not the solver, is empty. Not a scored
  claim. Attention O(N²) was not opened.
- **Directed-traffic Path B:** a hardware-measured directed jam-clearing
  signal (tilt-driven, growing to k=8, clean control) whose hostile
  hardness attack **converged** (Schmidt rank ≤512 through k=8, exact) —
  device validated, but classically reproducible; reported as a directed-
  transport demonstration, not a second advantage.
- **RL sampler parity (arm E):** disclosed; the device claim is sized to
  its measured margin (native, untuned, tightest variance), not overstated.
- **Rank universality:** a pre-registered prediction that the r=3 optimum
  is dimension-independent was tested at d=3 and **refuted**; the ablation
  is scoped to d=2.
- **Validation-ensemble null:** quantum-sampled disturbance fields beat
  white noise but tied a Markov surrogate on the single-vehicle plant
  (the vehicle low-passes the quantum-hard structure); reported, mechanism
  understood, flagged as future work for long-memory multi-agent systems.
- **Scope:** 2-state control substrate and small learned policies. The
  7B VLAM backbone is not trained or compressed here; it is the
  swappable box the two integrated stages are built to serve.

## 8. Reproducibility

Every result: a frozen pre-registration written before execution, raw data
stored verbatim, ≥3 seeds with mean ± SD, in-job controls, and cloud job
IDs. Key results replicated on a second, independent machine. All scripts
and JSON receipts accompany this submission (§0 map); every hardware figure
regenerates from archived counts with no credentials; every miss (§7) is a
receipt of the same class as every pass. Deadline: GIC 2026 Phase I closes
2026-09-15; this package is submitted with runway.

---

## Annex. Supporting hardware record (not a scored track)

The two scored tracks are the challenge answer. This annex is why the
mobility *disturbance class* those tracks will consume in Phase II is a
real-frequency object, not an ill-posed continuation.

For interacting mobility systems the industrially decisive quantity is
**S(q,ω), the jam-relaxation spectrum**. Imaginary-time plus analytic
continuation is mathematically ill-posed at any size. Quantum hardware
evolves on the real-time axis; the spectrum is a Fourier transform with
**no continuation step**. Shipped: X(q,t) of a 156-qubit congestion
interface on IBM Heron and `vw_jam_relaxation_spectrum.json` (dominant
collective line 0.0625 cycles/step across q = 0.5–2.0).

**Two-sided adjudication.** Hostile classical attack on our own data
(engines 1e-16 vs dense; joins 1e-15; all blocks twice): **k ≤ 6** fully
reproduced (0.4σ at k=4; five wavevectors within 1.4σ at k=6); **k ≥ 8**
the attack died (operators past 2×10⁸ terms) and the referee's verdict
stands: *"k≥8 cells stand contested with measured receipts."* Hardware
cells there: **8.9σ and 11.5σ**, clean k=0 control. Exact-theorem
anchors 0.9836 / 0.9806; derived 18-line spectrum beats a 20%-detuned
control (R² 0.87 vs 0.71). In-kind CHSH anchor **S = 2.3604, +35.9σ**,
same-day replica +37.0σ.

These cells will not be re-flown as a copy. Phase II *re-reads* them
under the §5 instrument (atlas layout, even-sector encoding, predicted
T2, L0 invert) and couples the spectrum to the §3 envelope and
Volkswagen's traffic stack.

---
*Team Merlin Digital. One framework, one instrument: a quantum component
in the VLAM alignment loop and a certified wrap on the VLAM action,
sized to the challenge bars, with every miss on the same receipt class
as every pass.*
