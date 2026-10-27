# Optional affine-preserving vector derivative: coupled gate failed

The default remains the existing mapped fourth-order vector discretization.
The new optional host derivative is an experimental fine operator. It preserves
Cartesian constant and linear shift modes without dividing beta by a puncture
power, pinning its value, projecting evolved fields, or changing the reference
connection. It is not qualified for wormhole evolution.

For Cartesian coordinate-volume weights, let P be the fixed global least-squares
projection onto the twelve affine vector coefficients. Its covariance is built
from geometry once; coefficients are fitted linearly at each operator evaluation.
For every first/second derivative use

    D_new beta = D_analytic(P beta) + D_mapped(beta - P beta).

The active beta values remain verbatim. Boundary remainder values are the actual
vector Dirichlet data minus P beta. The full metric-dependent L_g(P beta) is
retained; it does not generally vanish on a nonflat metric. The declared fine
residual uses this operator, not the previous mapped derivative. Old-operator
residuals on candidate solutions are recorded as separate controls. Both are
fourth-order consistent continuum discretizations; a small new residual alone
cannot validate the evolution gauge.

The stage adapter reconstructs the same derivatives for volume metric evolution
and subtracts/restores the affine mode in the scri Hermite reconstruction. The
reference beta=-x remains analytic. The scalar multigrid preconditioner is
unchanged. The option requires regular tensor scri, defaults off, and is not
exposed in the native AthenaK input. Checkpoint v17 binds it; v2--16 decode off.
Accepted regrid rebuilds the geometry projection rather than copying stale plans.

Manufactured full variable-metric tests retain all metric corrections. In the
N6/two-shell affine test, operator max error falls from 0.12884 to 1.09e-13 while
the continuum source remains nonzero (max0.157119). For a nonlinear vector MMS,
solution RMS error changes from 0.00143279 to 1.53151e-5 at N6 and from
4.7906e-5 to 1.01754e-6 at N10; both methods take20 iterations. Original declared
residual RMS is below1e-10 and point maximum below1e-7. The previous operator's
N6 nonlinear residual on the new solution is about0.10375: it is not reported as
passing that different discretization.

Production-engine controls verify equality with the independently qualified
prototype, linearity, invalid/stale-geometry rejection, physical beta values,
body derivatives and actual surface affine metric response. The default-off
nonzero angular pulse produces volume and sphere CSVs bitwise identical to the
prior method. Four final engine/regrid/checkpoint tests pass in23.96s, including
preservation of the option through accepted regrid and v16 backward decoding.

The genuine coupled research RK4 wormhole pilot uses the unchanged M=1/3
isotropic initial data, Minkowski reference, selected lapse/telegraph sources,
N6 eight-block interior, seven shells, supported58 positive derivatives and
Hermite scri. All10,800 cells and scri nodes remain in diagnostics. Two steps of
1e-4 take21.18s; restart is bitwise. The declared vector RMS residual is
6.74101e-11 (point1.9593e-9), but the final accepted tensor scri pole is
8.92509202e-5, versus baseline1.07350145e-6 and target1e-7. Hamiltonian RMS
remains0.163545. This is a substantial coupled failure, not an improvement.
No larger-resolution affine coupled pilot is justified before localization.
The independent selected radial N2049 comparison also worsens: radial beta RMS
0.00102225237 becomes0.00107266954; full Cartesian beta error RMS
0.00113637420 becomes0.00119232859. Raw angular scatter, every active-cell error,
interface/endpoint region splits and N1025-versus-N2049 reference controls are
retained in `affine-coupled-radial-comparison-20261027`. No radial cell is excluded.

A21.06s IO-only rerun saves the exact final accepted geometry and freshly solved
beta. Both output tables are bitwise equal to the first failed run; process exit1
records the scientific failure. The ordinary midpoint checkpoint contains a
last-stage warm start, so it must not substitute for this accepted-state audit.
The next diagnostic compares the actual discrete Frechet derivative of
metric-built V under the full metric RHS with L beta-S. It separates product-rule,
composed-first/direct-second derivative, log-chain, halo and surface effects.
A composed centered-first-derivative operator may have checkerboard null modes;
no such replacement has been adopted.

Evidence under `evidence/`:

- `affine-mapped-operator-20261027`, `affine-mapped-solve-20261027`: bounded commands,
  source capsules and full manufactured controls.
- `affine-engine-build-20261027`: build/test logs and accepted-capture byte proof.
- `affine-default-equivalence-20261027`: unchanged default coupled output proof.
- `affine-coupled-n6-20261027`, `affine-coupled-n6-accepted-20261027`: failed coupled
  runs, complete source revisions, initializers, all-cell/surface output and
  accepted checkpoint. Raw originals remain local beside lossless gzip copies.

Reproduce the small failed gate with a fresh output directory:

    cmake --build build-shift-3d --target elliptic_shift_3d_coupled_pilot -j2
    python3 tst/elliptic_shift_3d/run_affine_coupled_control.py --n 6 --output NEW_DIRECTORY

The runner has a180s process bound and50,000-cell budget. It deliberately retains
exit1.
Native production integration, radial collapse and changing puncture-regularity
safety are not established by these controls.
