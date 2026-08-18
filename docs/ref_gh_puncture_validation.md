# Reference-frame FO-GH puncture validation report

## Gate decision

**REFERENCE-GH FORMULATION NOT ESTABLISHED.**

Flat-reference algebra/source, exact-Minkowski, linear-wave, and robust-
Minkowski checks passed. A nonperiodic exact-trumpet `n=16`, `t=0.1` CPU/CUDA
smoke remained small (`field Linf 7.40e-11`, constraint Linf `7.69e-11`). These
bounded results do not override the required stationary convergence gate.

The intended `dx=1/16,1/24,1/32` initial-RHS ladder reverses with resolution:
`4.786e-9, 8.442e-8, 1.357e-7`. Corresponding reference-Ricci residuals are
`1.372e-7, 7.210e-7, 2.316e-6`; maxima move inward and the RHS maximum is
`Pi00`. The exact stationary state therefore fails the controlling gate.

Two evidence-rejected experiments were reverted: transformed-background source
subtraction produced `t=0.1` field errors `3.66e-10, 4.35e-9, 3.03e-8`; a
coordinate-wave identity increased the finer initial RHS to `1.109e-7` and
`3.049e-7`. No mask, floor, excision, reset, clipping, or weakened threshold was
introduced.

No stationary `t=20` ladder, wormhole transition, or puncture stability run was
launched, so no puncture-stability claim exists. Perlmutter allocation 57239999
was explicitly relinquished. A CUDA-aware MPI build succeeded but was not
runtime-qualified because the formulation gate failed first.

See `stationary_gpu_gate.tsv` and `perlmutter_stationary_gate_provenance.txt` in
`docs/fo_gh_artifacts/reference_frame_20260818/`. Large outputs are excluded.
