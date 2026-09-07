# Uniform decomposition and snapshot diagnostic checkpoint

## Communication and restart tests

`check_intrinsic_decomposition.py` holds the 8-per-direction global grid fixed
and compares one block with 4 blocks in 2D or 8 blocks in 3D. Blocks have four
active cells and four ghost layers in each active direction. It checks every
50-field global cell and every stored ghost face/edge/corner at cycles 0,1,2,3,
with dt=1e-4, KO=0.3, eta=2, kappa=1 and lambda=alpha*gamma_R (gamma_R=1).
The states are the non-Einstein smooth integration fixture, not physical initial
data. All tolerances are 2e-12 and were recorded before the tests.

The double-precision reader follows the actual restart writer's native POD
stream and saved logical locations. It checks layout/version/count, payload
length, uniform levels, periodic boundaries and exactly one owner per global
cell. It does not infer block ordering. It is restricted to these intrinsic-only
uniform fixtures, not a general restart conversion tool.

CPU serial: six FD2/4/6 x 2D/3D cases pass (18 processes). CPU two-rank MPI:
the same six cases pass (24 processes). MPI rank count is checked in the actual
log and by distinct rank health files. The MPI executable was rebuilt with
Kokkos bounds checking. Maximum discrepancy from single-block serial evolution
is 1.681e-18, including all ghost values. This tiny difference already exists
in initial data from floating coordinate evaluation across decompositions.
The original fixture has a shared symmetric phase. A stronger follow-up assigns
an independent integer oblique mode to every field using explicitly labeled
synthetic intrinsic restarts; this prevents axis/index permutations from being
hidden by the shared phase. It is an operator fixture, never physical initial
or contaminated legacy data. All six asymmetric two-rank comparisons are bitwise
identical to their serial reference, including every ghost and rank-changing
continuation.

That follow-up exposed a repeat-restart parser failure: the reader appends
`restart_tracker_state=false`, which the intrinsic constructor previously rejected
on a subsequent restart. The constructor now accepts false metadata but still
rejects true tracker state. The failed attempt is preserved; the repaired
asymmetric suite passes. Evolution equations and operators did not change.

Restarting at cycle 1 and continuing to cycle 3 is bitwise identical, both with
the same rank count and after changing from two MPI ranks to serial.

The isolated CUDA build uses a verified 6886b6af source snapshot and the existing
pinned Kokkos dependency, in a new scratch directory. Its test controller waits
for that specific build, verifies source/test hashes and available GPU memory,
then runs the nonlinear oracle and serial/two-rank decomposition checks. Its
original source is 6886b6af, before the repeat-restart parser correction. A separate
follow-up package is prepared for the correction and asymmetric tests; these
must be tested before claiming the latest parser is CUDA-qualified.
**CUDA build/tests are still pending at this checkpoint; no CUDA PASS is claimed.**
See the recorded controller status for the live process identities. Do not launch
a duplicate build or test merely because the controller has not finished.

## Independent snapshot diagnostics

`intrinsic_diagnostics.py` reconstructs only primary g,A from the chart and
Ahat for physical constraints. It differentiates materialized primary fields,
builds their ordinary Christoffel symbols, differentiates those symbols and
forms the conformal Ricci tensor. It does not use the evolution's Rstar, C/Z,
or independent auxiliaries to form physical H/M:

    H = 2K^2/3 - A_ij A^ij + w^2 R[g] + 4w Delta_g w - 6 |Dw|_g^2
    M_i = partial_j A^j_i + Gamma^j_jk A^k_i - Gamma^k_ji A^j_k
          - 2 partial_i K/3 - 3 A^j_i partial_j w/w

The alpha-weighted momentum is retained separately using rho in its final term.
The diagnostics also include C, Z, all 30 G-D_h(phi) residual components, all
30 intrinsic curl components and 18 independent raw Q-curl components. The
lapse potential is the materialized product rho*w at each grid point.

These are **global periodic analysis operators**, not yet production diagnostic
tasks. Repeated derivatives have a wider stencil; assembling the complete global
snapshot supplies valid data. This does not establish validity of evaluating
the same expressions from an unsynchronized four-layer local halo. Nonconforming
meshes and physical boundaries are rejected by this reader.

The analytic test uses the volume-preserving shear x'=x+0.15 sin(y), whose
metric has gxx=1, gxy=0.15 cos(y), gyy=1+(0.15 cos(y))^2 and gzz=1. With
w=1+0.1 cos(y), K=0.03 sin(y), Ahat=diag(0.02,-0.02,0), its physical constraints
are known analytically. They are nontrivial even though the conformal metric
is flat. Independent H and M maximum errors converge at orders:

| stencil | H observed orders | M observed orders |
| --- | --- | --- |
| FD2 | 1.962, 1.991 | 1.981, 1.998 |
| FD4 | 3.969, 3.992 | 3.970, 3.995 |
| FD6 | 5.950, 5.987 | 5.957, 5.992 |

Alterations to all GH/auxiliary fields leave physical H/M bitwise unchanged.
All ten auxiliary families are checked with seeded directional residuals/curls.
Raw Q-curl is recorded but does not yet have a separate snapshot-operator oracle.

`analyze_intrinsic_restart.py` applies the diagnostics to actual synchronized
MPI snapshots. It records per-component coordinate-volume L1/L2 integrals, RMS,
maxima, signed values at maxima and global k,j,i locations. The full active
volume is 2.21 in this fixture; ghosts never dilute the norms. Arrays and hashes
are retained. No decay, stability, physical-field convergence or puncture claim
is inferred from the four short integration snapshots.

## Reproduction

Run `check_intrinsic_decomposition.py` with `--binary`, `--reference-binary` and
`--output`; for MPI also specify `--launcher '/absolute/mpiexec -n 2' --ranks 2`.
The reference binary runs serially on one block. Run
`check_intrinsic_diagnostics.py --output /new/directory`, then
`analyze_intrinsic_restart.py --input /path/to/rst --output /new/directory`.
The plot script consumes the diagnostic check's results.json.

Next: finish the already-running CUDA checks and compare their float64 outputs
to CPU, then integrate independent diagnostics into the production task graph
with a valid stencil strategy. Nonconforming intrinsic transfer and all physical
qualification gates remain open.
