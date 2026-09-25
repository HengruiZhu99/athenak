# Stationary Kerr trumpet reference

The candidate extends the existing `R0=M` Schwarzschild trumpet to Kerr using
Dennison, Baumgarte and Montero,
[Trumpet Slices in Kerr Spacetimes, 1409.1887v2](https://arxiv.org/pdf/1409.1887v2).
Choose `R0=M`, spin length `a` with `|a|<M`, and `r=R-M`. The background is
stationary because its coefficients have no time dependence. The lapse is the
stationary ADM lapse, not a precollapsed replacement. This does not assert that
the ordinary unsubtracted 1+log equation leaves this reference stationary;
the implemented residual gauge retains explicit background subtraction.

## Reslicing from Boyer–Lindquist coordinates

Let `c=sqrt(M^2-a^2)`, `Delta=R^2-2MR+a^2`. The transformation is

\[
 dt_{\rm BL}=dt-\frac{(R^2+a^2)c}{\Delta(R-M)}\,dR,\qquad
 d\phi_{\rm BL}=d\phi-\frac{ac}{\Delta(R-M)}\,dR.
\]

Then set `r=R-M` and use ordinary spherical-to-Cartesian spatial coordinates.
Both time and azimuth change. The resulting metric is regular at the outer
horizon even though conversion coefficients from the singular BL chart diverge
there. A radial relabeling of Liu's wormhole chart alone cannot produce this
interior: that chart covers two exterior sheets with `R>=R+`. The new slice
instead enters the future horizon and approaches its trumpet at `R=M`.

## Cartesian fields and positivity

Define `n=x/r`, `w=(-n_y,n_x,0)`, `v=(-y,x,0)`,
`Sigma=R^2+a^2 n_z^2`, and `X=(R^2+a^2)^2-a^2(x^2+y^2)`. Direct transformation
gives

\[
 \alpha=r\sqrt{\Sigma/X},\quad
 \beta^i=\frac{c(R^2+a^2)x^i-a(2Mr+M^2+a^2)v^i}{X},
\]
\[
 \gamma_{ij}=\frac{\Sigma\delta_{ij}
  +a^2(1+2MR/\Sigma)w_iw_j-ac(n_iw_j+w_in_j)}{r^2},
 \qquad \det\gamma=\frac{\Sigma X}{r^6}.
\]

`X-r^2 Sigma=(R^2+a^2)(2Mr+M^2+a^2)>0` proves `0<alpha<1` for `r>0`.
In an orthonormal radial/polar/azimuthal basis, the radial-azimuthal block has
positive diagonal `Sigma` and positive numerator determinant `X`; the polar
component is also positive. Thus the spatial metric is positive definite off
the puncture. No absolute value, lapse floor, or clipping establishes these
properties. Stationarity determines

\[
 K_{ij}=\frac{\mathcal L_\beta\gamma_{ij}}{2\alpha}.
\]

The provider differentiates this identity using analytic second derivatives
of the ADM fields, returning `K` and its first derivatives. It does not supply
a fabricated `K` Hessian. At `a=0`, these expressions reduce to
`alpha=r/(r+M)`, `beta=M*x/(r+M)^2`,
`gamma=((r+M)/r)^2 delta`, and `K=M(delta-2nn)/r^2`.
The existing zero-spin implementation remains available for bitwise matching.

## Relation to the other references

[Liu, Etienne and Shapiro, 1001.4077](https://arxiv.org/pdf/1001.4077) construct
wormhole puncture data with a larger coordinate horizon at high spin. Their
moving-puncture evolution allows the slice to relax. Holding its two-sheet
stationary reference instead requires signed lapse; its negative sheet is
therefore a different gauge issue from the positive-lapse trumpet considered
here. Subtracting a stationary reference RHS does not remove negative
coefficients from the principal part of a residual perturbation equation.

[1806.08364](https://arxiv.org/pdf/1806.08364) constructs boosted, nonspinning
Schwarzschild trumpet data through a Kerr–Schild intermediate chart. The
boosted gauge only approximates its eventual equilibrium. It is useful guidance
for a later moving reference, but it is not an explicit stationary spinning
solution. A laboratory-frame boost introduces dependence on position minus
velocity times time; that requires a time-dependent reference implementation.

## Standalone verification and limits

`tst/unit/kerr_trumpet/validate.py` compiles the provider independently. Its
200 samples cover spins `0,0.5,+0.9,-0.9`, axes, horizons, interiors and radii
`0.03125M` through `2048M`. Independent physical ADM contractions gave maximum
Hamiltonian `1.15e-14`, momentum norm `6.93e-14`, stationary metric RHS
`4.58e-16`, and stationary curvature RHS `5.78e-14`. The zero-spin field
comparison is within `3.66e-15`. Value-only finite differences exhibit
sixth-order convergence before the finest Hamiltonian cancellation floor.
Surface angular momentum is `0.9` within `2e-15`; extrapolated ADM mass is
`0.9999999947` for the nominal unit mass.

At spin `0.9`, the horizon coordinate radius is `0.435889894M`. Therefore
`dx=0.125M` resolves its diameter with only about seven cells, compared with
16 for the original Schwarzschild trumpet. `dx=0.0625M` restores about 14.
The geometry has directional conformal-field limits at the puncture, and the
paper specifically cautions about numerical suitability. Geometry correctness
and exact zero preservation do not establish perturbation stability.

A separate reproducible plot and survey are stored under
`/Users/hz0693/research/TDE/kerr-trumpet-20260922/plot_geometry.py` with PNG/PDF
and JSON outputs. Over 18,876 sampled points per spin (`r=0.01..2048M`), the
maximum axis-wise physical speed estimate is `0.999237` for spin `0.9`, versus
`0.999512` for Schwarzschild. The lapse-gauge proxy is `1.413391` versus
`1.413666`, respectively. These estimates include shift advection but do not
analyze the full coupled Gamma-driver/Z4c characteristic system or prove a CFL
condition. The spin increases no sampled speed above the Schwarzschild maximum;
extra resolution, rather than this speed proxy, can still reduce the timestep.

## Evolution integration and current validation scope

The `z4c_tov_ks` problem accepts `problem/bh_background=kerr_trumpet`,
`problem/bh_spin=0.9`, and matching `coord/a=0.9`. It requires `bh_mass=1`,
`chi_psi_power=-4`, a direct analytic background, no coordinate excision or
interior sponge, and conserved-variable mesh prolongation. Set
`mesh_refinement/prolong_primitives=false`. The zero-spin option uses the old
Schwarzschild implementation unchanged. Diagnostic/AMR radial masks use the
spherical coordinate radius, not the Kerr-Schild oblate radius. The existing
background-adapted residual 1+log equations are unchanged.

Correctness regressions (NumPy and a separately built MPI executable required):

```sh
python3 tst/regression/z4c_kerr_trumpet.py --exe /absolute/build/src/athena --output /absolute/stage-results
python3 tst/regression/z4c_kerr_trumpet_guards.py --exe /absolute/build/src/athena --output /absolute/guard-results
```

The initial local tests passed twelve cases: zero vacuum, lapse perturbation,
and atmosphere with matter feedback, on uniform/refined meshes with one/four
MPI ranks. Zero remains exactly zero at audited RK, projection, exchange,
boundary and refinement operations. Nonzero perturbations have nonzero geometric
responses and identical active-cell hashes across MPI partitionings. Independent
background values agree through ghost cells within `6.7e-16`. Full stored/ghost
metrics remain positive. Eleven unsupported-configuration guards pass, and
spin-zero vacuum/lapse/atmosphere snapshots including ghosts are byte-identical
to the original Schwarzschild branch.

Matched local sixth-order pulse controls in the small domain `[-2,2]^3`,
`dx=.125M`, both reached `20M`. They used a `1e-8` lapse pulse, the same residual
1+log driver, `kappa1=kappa2=0`, lapse damping `.1`, shift eta `.02`, KO `.5`,
linear residual ghosts and characteristic CPBC; no interior treatment.
Kerr/Schwarzschild final maximum Theta was `1.546e-10`/`6.556e-11`;
exterior RMS was `2.770e-11`/`7.883e-12`. The fitted RMS slopes over `15–20M`
were `-.0802/M`/`-.1105/M`. Each RMS excludes that background's own horizon;
these proper volumes differ. A common `r>1M` region in the final equatorial
slice had maxima `1.272e-10`/`6.555e-11`. Neither comparison proves long-term
or stellar stability. At spin `.9`, this coarse grid only has seven cells across
the coordinate horizon; the refined/wider-domain controls are separate tests.

The raw Kerr exterior Hamiltonian maximum at initialization changes from
`1.21005` at `dx=.125M` to `.00446946` at `.0625M`; this is background truncation,
not a newly growing residual. Do not compare this diagnostic with Theta growth
or identify it as the first injection site.

`analysis/kerr_trumpet/check_checkpoint.py` is a read-only diagnostic for the
new background. It validates saved uniform/static per-rank payloads and raw
metric positivity using independent Kerr geometry. It deliberately does not
use the Schwarzschild campaign's geometry checks and does not authorize a
restart or certify constraints. The old production checkpoint controller is
not suitable for this new background.

## Integration into project/tde (September 2026)

This branch includes the validated stationary R0=M Kerr geometry construction,
its independent geometry and MPI regressions, the mesh-only owner-pointer
initialization fix, and the stationary-background cache (default on for explicitly stationary providers). See
[STATIONARY_CACHE.md](../../tst/unit/kerr_trumpet/STATIONARY_CACHE.md) for cache
invalidation, reproducible on/off comparisons and measured performance scope.
Fisheye mapping, experimental source corrections, derivative variants and gauge
or constraint-damping experiments are not part of this integration. Existing
input files do not switch to Kerr. Eligible analytic backgrounds now cache by
default; `problem/cache_stationary_background=false` preserves the uncached path.

Construction validation establishes positive lapse off the puncture, positive
spatial metrics, continuum ADM identities/stationarity and consistent residual
initialization. It does **not** establish long-time perturbation stability:
residual subtraction removes F_h(background), not its linearized action on a
perturbation. Several tested configurations, including finer meshes, exhibit
late coherent inner gauge/constraint growth. Neither this source integration
nor successful finite checkpoint audits clears a full stellar campaign.
