# PATH-B TILTED-TRAFFIC ATTACK — VERDICT: CONVERGES (GPU box, 2026-08-12)

Referee result on the directed-traffic contested window (trafficD
kingston card): the tilted differential dX(q,k) = X(Fp) - X(F0) is
CLASSICALLY REPRODUCIBLE through k=10, and through k=8 the classical
numbers are numerically EXACT, not approximate: the true Schmidt rank of
both arms stays <= 512 through step 8 (the chi=512 and chi=1024 ladders
agree to 0.0e+0 elementwise there; V1 gate: MPS == dense statevector at
L=12 to 1.1e-14; every run computed twice, bit-identical).

Classical dX(q=0.5,k): 0 (k=0, control exact) -> -1.3e-3 (k2) ->
-3.8e-3 (k4) -> -6.9e-3 (k6) -> -1.15e-2 (k8) -> -1.38e-2 (k10; ladder
uncertainty at k=10 only, C-elementwise 8e-4, X-level far below the
signal). Full table + both arms in traffic_attack.json.

**Per the prereg fork: Path B does NOT earn its own adjudicated
beyond-classical crossing. It stands as an honest directed-transport
demo; advantage rests on Path A.** The signal itself is real physics —
the drive restructures the jam's correlations, monotone and
sign-correct, and the classical curve confirms the device is measuring
it: flown -0.0011/-0.0033 (k6/k8) vs exact -0.0069/-0.0115 = the usual
depth-dependent damping envelope (f ~ 0.3-0.5 at these depths, same
class as the A1 arms). Device validated; advantage not.

Cost note for the record: the entire attack (validation, both arms, two
chi rungs, x2 passes, full correlation matrices at six depths) ran in
~5 minutes on the box. A 64-site depth-10 circuit with rank <= 512 is
not a hardness candidate at any q — if a Path-B-class window is wanted,
it needs depth k >= ~16-20 (where the A1-class 1D quench needed
chi=3072) or a 2D lattice.

Receipts: traffic_attack.json + traffic_attack.py (engine, V1 gate
inside). — GPU box
