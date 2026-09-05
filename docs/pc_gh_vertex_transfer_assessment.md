# Vertex collapse branch assessment for PC-GH

Assessment date: 2026-09-05. Classification: promising transfer improvement;
no new PC-GH evolution qualification or proof of discrete constraint preservation.

## Branch and scope

After fetching the fork, the latest relevant remote branch is
`origin/codex/z4c-vc-performance-aurora-20260829`, at
`6bfa5c11c3b10775294f5f2d95a196d84d6718cb` (2026-09-01 17:13:29 -0400,
`z4c: archive N1024 and Brill extrema evidence`). It contains the native
vertex-centered Z4c gravitational-wave/Brill collapse work and later performance
improvements. It is not a native-vertex PC-GH implementation.

The comparison pins current PC-GH source at `be93dfd0`; the full resolved hash and
SHA256 hashes of extracted sources are in the accompanying JSON. No branch was
merged and no production equations or transfers were changed. The numerical work
here is local matrix/polynomial analysis, not an evolution run. Future evolution
comparisons remain assigned to Della with the CUDA backend.

## What is better in that branch

`src/mesh/vertex_amr.hpp` uses coincident coarse/even-fine vertices, injection
restriction, exact copying at coincident prolongation sites, and symmetric
Lagrange interpolation at odd sites. Consequently restriction after prolongation
is exactly the identity, absent intervening evolution. Automatic interpolation
orders are q4/q6/q8 for spatial orders p2/p4/p6. The latest collapse workload is
O4 with q6. Halo sizing accounts for the interpolation radius; q8 with four fine
ghosts needs five coarse ghosts.

The synchronization design gives coincident copies one canonical value using
the finest-level contributors, and reconstructs hanging nodes. The latest sparse
MPI implementation preserves this reconciliation while avoiding global payload
replication. These are useful ingredients for controlling inconsistent copies
and repeated AMR injection.

Current PC-GH high-order cell-centered prolongation is selected by ghost width:
three points for ng=2 and five points for ng=4. Those are interpolation orders
q3 and q5 respectively. Thus current sixth-order differencing with ng=4 consumes
q5 transfers. The finite-difference utility source is byte-identical between the
two pinned branches: changing centering does not itself change its coefficients.

## Exact nesting is not exact derivative commutation

Let P prolong a base field and its independently stored derivative, and let
D_c and D_f denote coarse and fine differentiation. For reduction residual
R_c = v_c - D_c u_c, componentwise prolongation gives

    R_f = P R_c + (P D_c - D_f P) u_c.

The second term is a newly injected residual even when R_c vanishes. With
vertex q4 interpolation and centered FD2, cubic data are reproduced exactly,
but an exact polynomial calculation gives

    (D_f P - P D_c) x^3 = -3 H^2 / 4,     h = H/2.

Therefore the branch does not supply a commuting derivative transfer. Its
polynomial and round-trip properties do not establish compatible transfer for
PC-GH derivative/curl constraints. The branch tests and derivation inspected
here establish nesting, interpolation moments, halo/topology properties, and
convergence; they do not establish that missing commutation identity.

## Independent controlled operator experiment

Reproduce from this worktree, choosing a fresh output directory:

```sh
.venv-bbh-plots/bin/python analysis/pc_gh_regular_extension/compare_vertex_transfer.py --output qualification-runs-20260904/regular-extension/vertex-transfer-audit-verified
```

The retained authoritative invocation redirected stdout/stderr to
`qualification-runs-20260904/regular-extension/vertex-transfer-audit-verified.log`
and exited zero. The output directory contains `comparison.json` and pinned
source snapshots. Earlier `vertex-transfer-audit*` experiments are retained;
the initial run's BLAS warnings are superseded by explicit tensor contractions
in the verified run. Production weights are extracted and verified in exact
arithmetic. The hypothetical control's polynomial moments are also verified.

The experiment uses a unit periodic 1D grid, N=16/32/64/128 coarse points and
twice as many fine points. It compares actual CC and VC tensor factors with a
third, hypothetical CC Lagrange transfer at the VC interpolation order. This
control is not implemented in AthenaK. A stationary seam at x=1/2 consumes
interpolated left ghost values and exact right active values for
sin(2 pi x + 0.37). The table measures only the additional first-derivative
error induced by those ghosts, at N=64:

| FD order | Current CC | VC automatic order | CC elevated-order control |
|---:|---:|---:|---:|
| 2 | 2.2391e-3 (q3) | 4.3919e-5 (q4) | 3.6247e-5 (q4) |
| 4 | 6.3753e-6 (q5) | 1.1750e-7 (q6) | 8.7221e-8 (q6) |
| 6 | 7.7971e-6 (q5) | 2.8284e-10 (q8) | 1.9705e-10 (q8) |

This smooth experiment supports a large reduction in interface derivative
contamination, but attributes much of it to interpolation order. It does not
show that vertex centering is essential, or that these factors predict puncture
error reductions. Asymptotic one-sided ghost contributions generically scale
like H^(q-1) for first derivatives and H^(q-2) for second derivatives. Whole-grid
tests can hide this order loss through stencil cancellation.

For the full smooth-field commutator at N=128, CC/VC errors are approximately
1.893e-3/1.892e-3 for p2 and 1.140e-6/1.140e-6 for p4: bulk coarse/fine
differencing mismatch remains. For p6 they are 4.404e-8/6.180e-10; increasing
the interpolation order removes the lower-order transfer bottleneck. The
elevated-order CC control gives essentially the same p6 commutator improvement.

The all-mode, uniform-quadrature operator norm H ||D_f P - P D_c|| is still
nonzero: for p6 it is 2.4521 (current CC), 2.0742 (VC), and 1.7775 (CC control).
This is not a time-evolution amplification factor or an energy-stability proof.
The analysis omits full AMR restriction, nonlinear projection, synchronization,
RK stages, physical boundaries, moving interfaces, and multidimensional curls.

## Existing branch evolution evidence

All following paths refer to the pinned VC commit above.

* `docs/investigations/z4c_vc_brill_transfer_qualification_20260823/TRANSFER_SELECTION.md`:
  on the same VC geometry, raising q4 to q6 restored approximately fourth-order
  dynamic-AMR wave convergence from approximately second order. This supports
  the interpolation-order mechanism independently of the present static model.
* `docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/REPORT.md`:
  the latest N1024 result matches the published first curvature peak, with fine
  resolution triples compatible with O4 central fields through that peak.
  Peak C/H/M squared integrals fall roughly two orders of magnitude from N512.
  Nonetheless the strict constraint gate is exceeded near the peak, and late
  rebound/minimum three-level convergence is unavailable. The earlier broad
  failure labels must not obscure this later improvement. Conversely survival
  and peak agreement must not be reported as complete constraint qualification.
* `docs/investigations/z4c_vc_history_extrema_perlmutter_20260901/REPORT.md`:
  later dynamical Brill runs are partial, with block-cap/wall-time limitations;
  they do not establish converged black-hole formation.

These Z4c collapse tests do not qualify PC-GH reduction/curl propagation.

## Implications and next discriminating tests

The branch is a useful source of higher-order transfer, explicit halo contracts,
and consistent vertex ownership. It does not remove continual numerical sources
from the constraint subsystem. Even damped transport under persistent forcing
can retain an error of source strength divided by damping rate; transport does
not guarantee that the forcing, reflected modes, or unresolved wavelengths are
controlled.

The most focused next comparison is to raise CC transfer order with matching
halo support, then repeat the PC-GH p/Q/L/B pulse and interface tests with the
same PDE, damping, time step, and hierarchy. Measure the actual full-system
reduction and curl jumps around every transfer and RK stage; compare uniform
and fixed-AMR runs before the single-puncture convergence gate. Vertex layout
can then be compared separately if the transfer-order correction is insufficient.
Neither operator results nor short survival authorize skipping the original
single-puncture and merger qualification gates.

A full native-vertex PC-GH port also changes puncture sampling. Current isotropic
PC-GH initialization places punctures on cell faces and evaluates expressions
containing 1/r and x/r only away from zero. An origin-aligned vertex grid samples
r=0. The VC branch explicitly tests rejection of singular true-vertex Kerr
puncture data (`tst/unit/z4c/z4c_vc_kerr_puncture_rejection_test.py`). That test is
not proof that regular PC-GH variables cannot be initialized there, but a direct
layout switch requires separately justified analytic limits and evolution
treatment at r=0, plus boundary/restart/diagnostic changes. It is not a runtime
toggle for the existing PC-GH solver.
