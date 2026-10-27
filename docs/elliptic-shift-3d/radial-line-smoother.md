# Global radial-line smoother for the thin-shell scalar preconditioner

The standalone CPU prototype improves the exact seven-shell N8 scalar test
from115 to11 FGMRES iterations, with the full scalar Poisson operator,
restriction, prolongation, coarse solve, restart24 and1e-9 residual target held
fixed. Solve-only cost falls from4.36 to.50 seconds on this Mac. This verifies
one scalar preconditioner case; it does not establish vector-shift, native
AthenaK, MPI/GPU, coupled evolution or scalable performance results.

`src/multigrid/composite_radial_line.hpp` is a new helper and does not edit
`CompositePoisson`, `CompositeShiftStage`, the pilot or the final elliptic
operator. The test-only hierarchy in `test_radial_line.cpp` copies the existing
cycle/transfer structure solely to substitute this smoother. Production
integration remains a separate parent-owned step.

## Block equation and interface treatment

For the inverse-map derivatives J[A][a]=dqA/dxa, define
G[AB]=sum_a J[A][a]J[B][a] and H[A]=sum_a H[A][a][a]. The full scalar operator is

    A u = -sum_AB G[AB] D_AB u - sum_A H[A] D_A u.

Each line fixes a shell face, its angular block bounds and two angular cell
indices. It spans **all radial layers**, sorting active points by physical
radius. The cached dense block is

    B = -(G[00] D_00 + H[0] D_0)
        + 2 N² (G[11]+G[22]) I,

using the same second-order logical derivatives as the scalar preconditioner.
The radial center is2 N² G[00] and radial neighbors have coefficients
-N² G[00] minus/plus N H[0]/2. Radial ghost samples are expanded through the
actual active-only interpolation plans, retaining every same-line donor,
including donors across multiple layers. The anchor-form floating point
partition correction in `Fill` is retained too. Inhomogeneous physical data
are absent because these are homogeneous correction equations.

Other-line, angular, mixed-derivative, Cartesian/shell and seam couplings stay
in the **full residual**. Each simultaneous damped sweep applies

    u <- u + .65 B^-1 (rhs - A u)

to shell lines, and weighted point-Jacobi with the same damping to Cartesian
points. No neighbor data are fixed independently of the global residual.
No physical source or boundary condition is changed. The test constructs
these factors independently on N8,N4,N2 levels; the existing global direct
coarse solve is retained on N2. Geometric volume-weighted restriction and
trilinear prolongation are unchanged.

Dense partial-pivoted LU factors are cached per global line, with a default
maximum length128. The helper rejects missing matching angular blocks,
radial gaps/overlaps, stale topology revisions or ghost-stencil modes,
unsupported derivative order, malformed vectors and invalid damping. It
requires its referenced grid to retain its address/lifetime; moving a grid
owner after cache construction requires rebuilding the cache. A changed
geometry/regrid must rebuild every level. Shell hanging angular interfaces
are not silently approximated as separate lines.

## Exact seven-shell scalar comparison

The mesh matches the failed thin-shell pilot: Cartesian halfwidth.25,
2x2x2 interior blocks, radii[.45,.55,.60,.65,.70,.75,1], one block per shell
face, N8,25600 active cells, radial interface stencils enabled. All processes
had180-second timeouts and stayed below50000 active cells.

The manufactured scalar is

    u=(1-r²)(1+.2x-.1y+.15z²+.05xy).

It has homogeneous Dirichlet correction data at r=1. The discrete-MMS RHS
`A u_exact` tests linear algebra alone. A separate continuum-MMS RHS `-Delta
u_exact` is computed from analytic Cartesian jets and retains spatial
truncation error. This scalar preconditioner verification is separate from
the frozen physical alpha*A-source comparisons; it makes no claim that
`A beta_recorded` is an acceptable physical shift source.

| Case | Iterations | Original residual RMS | Solution RMS error | Solve seconds |
|---|---:|---:|---:|---:|
|Point-Jacobi cycles, discrete MMS|115|8.84372e-10|5.33285e-12|4.35929|
|Global radial-line cycles, discrete MMS|11|1.53029e-10|6.29520e-13|.50224|
|Global radial-line cycles, continuum MMS|11|1.54886e-10|1.53152e-4|.49111|

The original residual is recomputed by unchanged `CompositePoisson::Operator`
at every FGMRES restart and before success. The test's cycle operator agrees
bit-for-bit with that operator on a separate dense-valued vector. Every
active cell contributes to the volume-weighted RMS; no region is excluded.
The same small setup creates both hierarchies for comparison and takes
approximately.74–.79 seconds, so it is not a measurement of either production
constructor alone. Complete process wall times were5.71,1.31,1.35 seconds.

The helper has384 global lines of length56 on N8,96 of length28 on N4, and24
of length14 on N2. At the coarse level24 nonadjacent-layer donor entries are
retained; a block-local tridiagonal substitute would miss them. Minimum
pivot/max-block-entry ratio is.0164 on N8. That ratio is a diagnostic, not a
condition number or proof for arbitrary configurations.

Independent `First`/`Second` plus actual ghost-fill evaluations check six
representative complete lines per level, covering all six faces. Maximum
normalized block-assembly error is2.18e-16. Direct matrix multiplication
checks the LU solve on **every** line, with maximum normalized residual
1.07e-16. The matrix and LU are both retained in the helper for these
optional audits; production use could later reduce diagnostic storage.

## Refinement, subdivision and rejected configurations

Refining one Cartesian leaf gives29184 N8 cells. The same line cycle converges
in13 iterations. This checks the existing coarse-fine/global transfer path
with the new smoother; it is not a regridding-during-evolution test.

A full N4 hierarchy with two angular subdivisions per face reaches1408
coarse N2 cells and fails the **inherited**1024-cell bounded direct-solve cap.
The attempt's exit2 and exact error are preserved in `controls.json`; the cap
was not relaxed. This is a hierarchy resource/configuration limit, not an
observed line-matrix instability.

A separate helper-only N4 test on those same subdivided faces has11264 cells,
384 global lines of length28, block-assembly error2.03e-16 and LU residual
8.09e-17. Zero correction is preserved exactly. Seven negative contract tests
confirm explicit errors for zero damping, shape mismatch, insufficient line
cap, changed ghost-stencil mode, changed topology revision, fourth-order
preconditioner grids and unmatched angular-layer metadata. These checks
exercise the helper without circumventing the existing coarse-solve limit.

## Evidence and commands

`evidence/radial-line-smoother-20261027/manifest.json` records compile commands,
all actual compiler dependency hashes, a compiled-source archive, unchanged
source/binary checks, per-process bounds and complete residual histories.
`controls.json` retains the refined success and subdivision failure.
`contracts.json` and its compiler command record the helper-only tests.

```sh
python3 tst/elliptic_shift_3d/run_radial_line.py \
  --output build-shift-3d/new-radial-line-evidence
# The separately compiled helper-contract command is recorded in
# evidence/radial-line-smoother-20261027/contract-build-command.json.
```

The original CompositePoisson/Stage/Pilot files were not edited by this
subtask. Remaining gates are parent-owned vector-preconditioner integration,
original-vector residual/cost validation on the actual thin-shell pilot,
regridding/restart cache rebuilding, and portable device/MPI implementation
if this host prototype is adopted.
