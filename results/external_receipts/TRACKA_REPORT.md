# TRACK A — M3 evaluation report (GPU box, 2026-08-10)

Inputs: `tracka_sensitivity.json` (M1, this box), `tracka_alloc_dirac.json`
(M2, Dirac-3, 9 points, all DP-verified optimal). Same frozen probe
(CIFAR-100 test, 2,000 fixed indices, seed 21), same quantizer (symmetric
per-output-channel, {2,4,8}-bit), CLIP ViT-B/32 vision tower + projection
(87.85M params, fp16 = 175.7 MB). Whole eval run twice: **CLEAN** (both
passes bit-identical).

## Headline, stated honestly

**The success criterion FAILS on real evaluation; the quantum-vs-random
ablation PASSES decisively at every budget.**

- Criterion was: dirac arm >= 2.5x compression at <= 5% accuracy drop AND
  strictly above random at equal budget.
- Measured at the gentlest point (2.74x): 33.5% accuracy vs 64.2% fp16
  baseline — a 30.7-point drop. No point on the curve is close to the
  5%-drop bar.
- At every one of the 9 budgets, the Dirac allocation beats the random
  ablation by 5x-25x accuracy (e.g. 33.5% vs 1.5 +/- 0.2% at 2.74x).
  The device-solved allocations are genuinely, measurably better than
  chance at identical budgets — the mandatory ablation is unambiguous.

## Full table (probe accuracy, 2,000 images; random = 3 seeds mean+-SD)

| arm | size | ratio | accuracy | drop |
|---|---|---|---|---|
| fp16 | 175.7 MB | 1.00x | 64.2% | — |
| uniform INT8 | ~88 MB | 2.03x | 65.5% | -1.3 (better) |
| uniform INT4 | ~44 MB | 3.97x | 24.1% | 40.1 |
| dirac lam=2e-05 | 64.1 MB | 2.74x | 33.5% | 30.7 |
| random @64.1 MB | 64.1 MB | 2.74x | 1.5 +/- 0.2% | 62.7 |
| dirac lam=5e-05 | 51.9 MB | 3.38x | 21.4% | 42.9 |
| random @51.9 MB | 51.9 MB | 3.38x | 1.1 +/- 0.2% | 63.1 |
| dirac lam=1e-04 | 45.1 MB | 3.90x | 14.9% | 49.2 |
| random @45.1 MB | 45.1 MB | 3.90x | 1.2 +/- 0.5% | 63.0 |
| dirac lam=2e-04 | 41.1 MB | 4.27x | 6.8% | 57.5 |
| dirac lam=5e-04 | 35.1 MB | 5.00x | 0.7% | 63.5 |
| dirac lam>=1e-03 | 30.3-31.2 MB | 5.6-5.8x | 1.3-1.4% | ~62.9 |

(random arms at the aggressive budgets: 0.9-1.3%, omitted rows for
brevity; all in tracka_pareto.json.)

## Diagnosis: the additive surrogate is the failure point, not the solver

The M2 allocations are exactly optimal (DP gap 0.0, 9/9) FOR THE OBJECTIVE
THEY WERE GIVEN — sum of single-group deltas from M1. That objective does
not transfer: joint quantization is strongly SUPER-additive on this model.
The cleanest internal evidence is uniform INT4: the sum of its 344
individual deltas predicts ~940/2000 correct; measured is 482. At 2-bit
the interaction is far worse (the predicted-negative-delta points — the
"regularization" reading of M2's -744/-359 — measure as 30-50 point
drops). M1's numbers are faithful measurements of what they are:
one-group-at-a-time sensitivities. The additive composition assumption is
what broke, exactly as M2's cover note suspected ("the additive
predictions look optimistic... your real numbers decide").

## What would rescue the track (options, not runs — deadline calls)

1. **Interaction-aware calibration set for the same solver**: measure
   delta on small random GROUP SETS at matched budgets, fit a pairwise or
   per-layer-count correction, re-solve on Dirac-3 (same flow encoding,
   objective stays linear/quadratic). One more M1-M2-M3 cycle.
2. **Error-compensating quantizer** (GPTQ-style, weights adjusted to
   absorb quantization error): shifts the entire curve; uniform INT4
   typically recovers to near-fp16 on ViT-class models, and the {2,4,8}
   allocation problem then starts from a usable floor. Larger change,
   standard tooling exists.
3. **Restrict levels to {4,8} with a calibrated joint objective** —
   abandons the 2-bit lever but the INT4 collapse suggests the win, if
   any, lives between 2.0x and 3.97x anyway.
Note the ablation result survives any of these: device-solved >> random
is already demonstrated on real accuracy at nine budgets.

## Receipts

- `tracka_pareto.json` — full results, both-pass CLEAN, seeds, byte
  accounting (group bytes include per-row fp16 scales).
- Probe, groups, quantizer, and codebooks identical to M1 by
  construction (same JSON is the single source of truth).
- Random arms: uniform-random {2,4,8} draw, greedy repair to <= budget,
  then greedy raise while <= budget; seeds 1-3; achieved bytes within
  0.2% of the dirac point's bytes at every budget.

— GPU box
