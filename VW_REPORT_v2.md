# Quantum Advantage of Route for Mobility Response — with Quantum-in-the-Loop Training and Certified Runtime Assurance

**Global Quantum + AI Challenge 2026 — Enterprise Challenge (Volkswagen)**
**Tracks: RL Alignment (primary scored) + Safety (secondary scored) · Context: Autonomous Driving / Mobility**
**Team: Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · Report v2, 2026-08-26**
**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-vw-gic2026**

---

## 0. Verify the headline claims (each row: claim → artifact → receipt)

| Claim (§) | Evidence artifact | Receipt |
|---|---|---|
| Advantage of route: S(q,ω) with no continuation step (§1) | `vw_jam_relaxation_spectrum.json` | fez job `d9rm4j9dsedc73agrb70` |
| Classical boundary, two-sided adjudication (§1, §5) | `A1_FINAL_VERDICT.md` (hostile referee) | kingston `d9rdfb1dsedc73agh5ng`; attack receipts |
| RL: 35.4% under GRPO budget, 3/3 seeds (§2) | `vw_rl_{A,B,C,D,E}_s{21,22,23}.json` | 71 Dirac-3 job IDs in JSONs |
| RL: +9.3-pt robustness at equal budget; ~2× budget-equivalence, two-box (§2) | `rl_finalquality_classical.json` + `RL_FINALQUALITY_BOX.md` | replicated, independent machine |
| Safety: 78→100% under RTA; envelope predicted then confirmed (§3) | `vw_rta_demo.json`, `vw_lyap_bias_sweep.json` | 3 seeds × 500 episodes ×2 machines |
| Framework-led device physics: flow encoding +26%, 3.5× tighter (§4) | `flow_encoding_test.json` | 12 paired Dirac-3 jobs, frozen prediction |
| CHSH in-kind anchor +35.9σ, replicated (§5) | `chsh_*_20260805*.json` | fez, same-day independent replica |
| Exact-theorem anchors 0.98; derived spectrum beats detuned (§5) | `a1p_result_20260808_200224.json` | two-engine pending, 4/4 gates |

Every negative result carries the same receipt class (§7).

## 1. The advantage claim — an advantage of route, stated and bounded

For interacting mobility systems the industrially decisive quantity is the
**real-frequency collective response** — how congestion forms, relaxes, and
responds to control: **S(q,ω), the jam-relaxation spectrum**. The scalable
classical route to real-frequency response functions is imaginary-time
simulation plus analytic continuation, which is **mathematically ill-posed
at any system size, independent of entanglement or coupling**: distinct
spectra reproduce the same imaginary-time data within any finite error bar
(counterexample pair exhibited in our GIC 2026 finalist repository; this
route argument has been through independent judging — dual-track finalist).
Quantum hardware evolves on the real-time axis natively; the spectrum
follows by direct Fourier transform with **no continuation step**. This
package ships that object: the collective fluctuation series X(q,t) of a
156-qubit congestion interface, measured in real time on IBM Heron, and its
spectrum `vw_jam_relaxation_spectrum.json` (dominant collective relaxation
line at 0.0625 cycles/step across q = 0.5–2.0).

**The activation boundary is measured — and adversarially adjudicated from
both sides.** We commissioned a hostile classical attack on our own flown
data (validated engines: 1e-16 vs dense; joins unit-tested 1e-15; all
blocks computed twice):
- **Dormant side (k ≤ 6):** the attack fully reproduced the hardware —
  0.4σ at k=4 face-value; all five wavevectors within 1.4σ at k=6 through a
  single damping constant; truncation-robust to 9×10⁻⁶. The instrument is
  proven by its adversary.
- **Active side (k ≥ 8):** the attack measurably died — operators past
  2×10⁸ terms (≈11 GB each), pair joins beyond machine and method — and
  the referee's shipped verdict states: *"k≥8 cells stand contested with
  measured receipts — the classical frontier measured rather than
  assumed."* The hardware's pre-registered cells there: **8.9σ and 11.5σ**,
  with a clean k=0 preparation control.

Scope stated plainly: flown instances are graded wherever verification is
possible (exact-theorem anchor waveforms at correlation 0.9836/0.9806; the
two-sided depth receipts); the beyond-boundary sector is claimed per the
referee's verdict. The demonstrated capability class — real-time collective
response of interacting lattice media beyond classical reach — is the
computation class of fleet-scale interaction, congestion control, and
long-memory mobility physics. Phase II scales the demonstrated generator to
network topologies coupled to VW's traffic stack.

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

**Mandatory quantum ablation — the six-arm map, complete.** Random fails;
a classical *optimizer* of the identical objective breaks training (0/3);
a tuned classical *sampler* reaches parity (disclosed, sized honestly);
the device delivers the winning ingredient — stochastic near-optimal
curriculum sampling — natively, with zero tuned parameters and the
tightest seed variance, executing in every iteration of the scored run
(71 receipted jobs). We additionally show (C′) that *improving* the
device's optimization quality reproduces the classical optimizer's
failure — measured proof that the metric rewards sampling, which bounds
every entrant, classical or quantum, at sampling-parity on this track.

## 3. Safety track (scored): certified runtime assurance

**Method.** Tensor-network Lyapunov monitor V(x)=‖L·φ(x)‖² (3-level
tensor-product features, rank 3 of 9), grid+Lipschitz certified, wrapping
an uncertified learned policy (behavior-cloned MLP with covariate shift and
100 ms delay): policy drives; verified fallback engages at V > 0.75c with
hysteresis. Only monitor + fallback face certification — the learned
driver stays a black box, swappable without re-certification.

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
chain — the artifact an ISO 26262 case consumes — which no heuristic
produces at any tuning.

**Quantum-component ablation (rank axis):** r=2 certifies nothing
(expressivity floor); **r=3 interior-optimal, 0.462 ± 0.030 certified
area**; full rank unstable (± 0.110 with a collapsed seed); in the
saturation-bound regime the TN certificate covers **1.57 ± 0.29×** the
best quadratic's area (all seeds ≥ 1.39). Scope: the r=3 optimum is a
measured d=2 result; at d=3 the useful rank grows (disclosed, §7). The
low-rank tensor structure is genuine regularization — removing it costs
certified area *and* reliability.

## 4. Framework-led device physics (supporting, not scored)

The framework's reading of the annealer's dissipative dynamics predicted,
before running, that encoding on the machine's native flow manifold (no
penalty walls) beats the naive binary encoding. Measured on the identical
selection objective, 6 paired trials, 12 receipted jobs: **flow encoding
−26% better objective, 3.5× tighter spread** — a frozen prediction
confirmed. This is the design rule behind the RL curriculum encoding and
generalizes to any optimization VW places on analog quantum hardware; it
is reported as method provenance, not a scored claim.

## 5. Hardware annex — the measured record behind the advantage

- **Adjudicated crossing (§1):** IBM Heron kingston `d9rdfb1dsedc73agh5ng`
  (16 × 32,768 shots, 64-site interface on 156 qubits) + fez
  `d9rm4j9dsedc73agrb70` (36 × 8,192, real-time spectroscopy series);
  hostile referee verdict `A1_FINAL_VERDICT.md`.
- **Exact-theorem anchors:** the local sector's evolution is a theorem
  (240-dimensional closed operator orbit, two engines agreeing to 1e-9,
  step map unitary to 1.0000000000). Hardware reproduces these exact
  waveforms at correlation **0.9836 / 0.9806**, and the derived 18-line
  spectrum beats a 20%-detuned control (R² 0.87 vs 0.71); a detuned-
  dynamics control arm moved as derived (6/6).
- **In-kind anchor (theorem-level, no classical erosion path):** the
  framework vacuum violates the CHSH bound at **S = 2.3604, +35.9σ**,
  independently replicated same-day (+37.0σ), derived null behaving.
- **Provenance chain:** two months, five instrument classes, two vendors,
  21 adjudicated frozen-prediction rows (GIC 2026 finalist ledger) plus this
  submission's ~90 receipted jobs; the same discipline that placed
  a dual-track finalist placement in GIC 2026.

## 6. Deployment architecture

Quantum work is train/validation-time only — inference latency untouched
(22 µs monitor, 4,500× under budget). The runtime-assurance pattern makes
the uncertified VLAM swappable without re-certification. Every quantum
output is classically verified before use: selected curricula are index
sets; certificate candidates pass the deterministic grid+Lipschitz
certifier; spectral cells are graded against the two-sided referee.
Quantum hardware is never a trusted component — the correct posture for a
safety pipeline.

## 7. Negative results and limitations (disclosed with the same receipts)

- **Compression track:** abandoned by pre-registered kill-switch — GPTQ-INT4
  clears the ≥2× @ ≤5% bar *classically alone* (3.97× at 2.45% drop); the
  {4,8} allocation is flat and the {2,4} frontier collapses immediately
  past 4× (15 accuracy points for +0.09×) — super-additivity unharvestable
  by any allocator, quantum or classical. The device solved its cycle-1
  linear objective exactly (DP gap 0.0, 9/9) and beat random 5–25× on real
  accuracy; the honest verdict is that the problem, not the solver, is
  empty. Ships as a methods section, not a scored claim.
- **Directed-traffic Path B:** a hardware-measured directed jam-clearing
  signal (tilt-driven, growing to k=8, clean control) whose hostile
  hardness attack **converged** (Schmidt rank ≤512 through k=8, exact) —
  device validated, but classically reproducible; reported as a directed-
  transport demonstration, not a second advantage. Structural lesson:
  beyond-classical needs the 2D interface geometry (the §1 result) or
  depth k≥16–20, not a 1D chain.
- **RL sampler parity (arm E):** disclosed; the device claim is sized to
  its measured margin (native, untuned, tightest variance), not overstated.
- **Rank universality:** a pre-registered prediction that the r=3 optimum
  is dimension-independent was tested at d=3 and **refuted**; the ablation
  is scoped to d=2.
- **Validation-ensemble null:** quantum-sampled disturbance fields beat
  white noise but tied a Markov surrogate on the single-vehicle plant
  (the vehicle low-passes the quantum-hard structure); reported, mechanism
  understood, flagged as future work for long-memory multi-agent systems.
- **Scope:** 2-state control substrate and small learned policies; the
  reference VLAM is architecturally exempted (§6), not evaluated end-to-end.

## 8. Reproducibility

Every result: a frozen pre-registration written before execution, raw data
stored verbatim, ≥3 seeds with mean ± SD, in-job controls, and cloud job
IDs. Key results replicated on a second, independent machine. All scripts
and JSON receipts accompany this submission (§0 map); every hardware figure
regenerates from archived counts with no credentials; every miss (§7) is a
receipt of the same class as every pass. Deadline: GIC 2026 Phase I closes
2026-09-15; this package is submitted with runway.

---
*Team Merlin Digital. One framework, one instrument discipline, from a
GIC-finalist materials result to this mobility submission: the classical
route to real-frequency response is broken by mathematics; the processor
travels the route natively; the boundary is measured, adjudicated, and
published.*
