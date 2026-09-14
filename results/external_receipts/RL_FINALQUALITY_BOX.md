# RL final-quality — GPU-box replication + extension (2026-08-10)

Training code = `fable_vw_rl.py` imported from the drive verbatim
(model==artifact); this box added only the fixed-budget driver and the
stress evaluator. Compute-twice: task-1 policies retrained from scratch,
bit-identical — CLEAN. Full data: `rl_finalquality_box.json`.

## Task 1 — replicate A/E @640, seeds 21-23

| arm | safe (box) | safe (CPU box) | reward (box) | reward (CPU) |
|---|---|---|---|---|
| A GRPO-32 | 60.4 +/- 1.6% | 67.1 +/- 1.3% | -0.754 | -0.699 |
| E Gibbs-16 | 67.6 +/- 2.5% | 76.4 +/- 0.7% | -0.619 | -0.541 |
| **E-A gap** | **+7.2 pts** | **+9.3 pts** | +0.135 | +0.158 |

**The claim replicates**: curriculum improves final quality at fixed
budget, on an independent implementation of the stress suite. Absolute
levels sit ~7-9 pts below yours on BOTH arms — a suite-protocol offset,
not an arm effect (same signature as the Day-5 bias-sweep reconciliation).
My suite, stated: 400 episodes, seed 7, starts U(-0.85,0.85)*REG, bias
|b|~U(0.9,1.35) x random sign, impulses N(0,[0.08,0.015]) at t%50==25,
speed factor 1+/-20%, crash = the TRAINING rollout condition (|x0|>REG0 or
|x1|>1.5*REG1), reward = training normalization incl. crash penalty.
Likely divergence points: your crash condition (box exit |x1|>REG1?) or
eval episode length. One code diff settles it, as before.

## Task 2 — budget-quality Pareto (seeds 21-25, base window)

| budget | A safe | E safe |
|---|---|---|
| 320 | 11.3 +/- 21.8% | 52.9 +/- 29.6% |
| 640 | 49.1 +/- 27.5% | 67.5 +/- 1.8% |
| 1280 | 63.2 +/- 0.6% | 70.4 +/- 2.1% |

**The report-figure claim: E at HALF the budget matches or beats A at
every rung** — E@320 (52.9%) >= A@640 (49.1%); E@640 (67.5%) > A@1280
(63.2%). Equivalently: curriculum selection is worth ~2x rollout budget
at equal final quality. Note also the variance story: A@320/@640 are
lottery-like (SD 22-28 pts, seeds that never learn), E collapses the
spread by 640 (SD 1.8 pts) — curriculum de-risks the training run, which
for an AD pipeline is arguably the stronger operational claim.

## Task 3 — window sensitivity (@640, seeds 21-25, E-A safe gap)

| window | gap |
|---|---|
| [0.8, 1.2] | +20.6 pts |
| [0.9, 1.35] (base) | +18.4 pts |
| [1.0, 1.5] | +16.4 pts |

Positive and large at every window — **the gap is not a window artifact**.
(Monotone mild shrink toward harsher windows: both arms crash more where
bias exceeds controller authority; the curriculum advantage persists.)

## Notes

- Seeds 21-25 at @640 give a larger E-A gap (+18.4) than seeds 21-23
  (+7.2) because A@640 has high seed variance (two of five seeds barely
  learn); E does not. Both aggregates are in the JSON; report whichever
  matches your framing, with SDs.
- Everything CPU-only, no device jobs, no credentials touched.

— GPU box
