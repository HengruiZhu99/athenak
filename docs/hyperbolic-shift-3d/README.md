# Native hyperbolic shift: admission work, not a completed 3D evolution

**The requested coupled hyperboloidal evolution is not implemented on this branch.**
It contains native C++ point kernels, a full 3D frozen-symbol audit, mapped
operator tests and bounded reproduction of the unchanged spherical donor.
The new kernels are not called by the evolution task graph. In particular, this
is not a wormhole-to-trumpet result, a scri boundary well-posedness result, or a
replacement for the preserved elliptic experiment.

## Sources and preserved infrastructure

* Infrastructure parent: `f5d99045e0168e844f87fc0b6189593aedd18da9`.
* Formulation/test donor: `c0ae7472a7aecc92cc72312c33d4cdc417a26291`.
  Both the connector and a subsequent `git ls-remote` found that exact head on
  `codex/hyperboloidal-hyperbolic-shift`; no later update was present.
* Kokkos: `6739bc623081648af9e752b616d9671527922cbf`.
* The separate `codex/hyperboloidal-elliptic-shift-3d-multigrid` head was
  `6d54b2ef7154102325d83ef4faa3917e52b28df6`; it was not checked out or changed.

The required donor Python modules, spherical C++ adapter and frozen coefficient-four
input were copied verbatim (41 files). The small standalone CMake harness is
adapted to build only the needed library and four audits. These are reproduction
tools, not a new Python evolution implementation.
No donor geometric source directory was merged: the native tensor headers at
the infrastructure parent already match the donor. The global Cartesian field
components, complete J/H derivative transformation, mesh framework, RK coupling
and Cartesian AMR implementation remain intact. The only extension.cpp change
rejects `cubed_sphere/hyperbolic_shift=true` explicitly, so that requesting this
unfinished mode cannot silently run the regular finite-radius equations.

## Implemented point equations

`src/z4c/hyperboloidal/hyperbolic_shift.hpp` implements the donor coefficients,
fixed Cartesian telegrapher vector, shift rates, auxiliary transport, compact
Z4 damping and scri normal identities. The default **and test selection** of
the unsuppressed bulk coefficient is four. No Omega floor, metric floor,
constraint projection, elliptic solve, or metric-built replacement of the
independently evolved connection is introduced.

For the unit fixed Minkowski reference, let
`Omega=(1-r²)/2`, `A=(1+r²)/2`, `u=alpha-A`, and
`D_t=partial_t-beta^i partial_i`. The 3D coupling is

```
D_t alpha = r² + S_alpha/Omega + eta chi partial_i(w B^i)
tau D_t B^i = kappa w delta^{ij} partial_j u - B^i
D_t beta^i = -x^i + mu D^i - 4 C (beta^i+x^i)/Omega
D_t D^i = D_t Lambda^i - D^i
```

Here `kappa=.005`, `tau=.02`, `epsilon=1e-4`,
`eta=(1+epsilon)alpha/(alpha+epsilon A)`, `F=alpha²+4 Omega² alpha`,
`xi=1+2(1-r²)^4`, and

```
S_alpha = -F(p+3) + x_i(alpha beta^i+A x^i)
          -2 xi alpha u +16 Omega^4 r² u/A.
```

**Source discrepancy:** the donor report prints `r(alpha beta+r)` in its
lapse equation. Its validated native `UnfactoredReferenceGauge` code instead
gives `r(alpha beta+A r)`. The latter cancels at the Minkowski reference and
is what is ported here. The regression compares against that native donor
kernel, not against a duplicated transcription of the report.

The flat-reference divergence and componentwise advection are intentional
coordinate gauge choices. They are invariant under global Cartesian rotations;
they are not a proposal to transform the contracted conformal connection as
an ordinary vector between patch coordinate frames. All components stay in
the shared Cartesian frame. The window is the donor C7 profile, one through
`r=.56`, zero at and beyond `.8`; the shift collar C starts at `.8` and reaches
one at `.9`. Consequently `w*C=0` exactly, and the donor radial metric term
in its separated coefficient never contributes to the blend:

```
mu = (1-C)*4 + C*(3/2)*chi*F.
```

This is an exact direction-independent extension of the blended coefficient,
not a replacement with a new bulk coefficient. Compact damping uses the finite
product `(1-w)*4*alpha+w*8*A` with `kappa2=-.99`, and the donor's disclosed
Appendix-C scalar-Theta source adjustment. It never divides by alpha.

## Principal symbol and the obstruction

The test library extracts the geometric principal coefficients from the native
`ConformalRHS` on the determinant-one/tracefree algebraic manifold. Its 26
reduction variables are the normal derivatives of chi, five metric components,
alpha, three shift and three B components, together with five A components,
`p/Omega`, `Theta/Omega`, three evolved Lambda components and three D components.
The physical-trace reduction is invertible only at positive Omega. Its limit
at scri is **not** a proof about the singular boundary PDE.

For covector n, define `Q=gtilde^{ij} n_i n_j` and `E=delta^{ij} n_i n_j`.
The sampled native symbols agree with

```
cP² = alpha² chi Q
cL² = chi F Q + eta chi kappa w² E/tau
cS² = 4 mu Q/3
cT² = mu Q
det(zI-P) = z^6 (z²-cP²)^6 (z²-cL²)(z²-cS²)(z²-cT²)^2.
```

The test audits eigenspace dimensions, not just real eigenvalues. It checks
100 positive-metric oblique samples, scalar restriction against the donor,
and selected transverse/light/lapse collisions. These finite tests do not
constitute a uniform symmetrizer proof.

A concrete counterexample to extrapolating the radial gap to all directions is
at `r=.4`, `alpha=.5`, `chi=1`, `F=.6028`, `eta=.9999840018557847`, `w=1`,
`mu=4`, with

```
gtilde = diag(0.0027928390440952237,18.92243605719428,18.92243605719428)
radial direction = (1,0,0); tested direction = (0,1,0).
```

The metric is positive and has determinant one. The donor radial gap times g
is `4.72983513474237`, yet the tangential longitudinal shift and lapse speeds
coincide at `+/-0.5308976962588428`. Each eigenvalue has algebraic multiplicity
two and geometric multiplicity one. The complete matrix and operands are saved
in `evidence/native/oblique-collision.npz`. This is an admissible-state
counterexample, **not evidence that a reproduced donor trajectory reaches it**.
The donor's exact scalar collision-minor proof is also reproduced in
`evidence/exact-donor-scalar.json`; positive alpha, chi and F with lapse speed
above light speed make the relevant 9-by-9 minor nonzero at the collision.

The implemented all-direction gap diagnostic is

```
Gmin = 4 mu/3 - chi F
       - eta chi kappa w² lambda_max(gtilde)/tau.
```

It is the minimum of `(cS²-cL²)/Q` over directions. A radial-only monitor
misses the counterexample. Staying in a compact positive-geometry region with
strict positive Gmin is a possible restricted admission condition; positivity
of alpha and chi alone is insufficient. The exactly collapsed puncture corner
still has the donor's defective limiting scalar symbol. A lapse floor does
not establish a uniform estimate there.

An explicitly separate control sets
`mu=(3/2)chi(F+eta kappa w² lambda_max(gtilde)/tau)`. It uses mu=8 at the
counterexample and restores complete eigenspaces there and in the 100 sampled
oblique states. **It is not adopted**: it changes constant bulk four, has no
nonlinear evolution validation, and does not fix the collapsed corner or prove
boundary stability.

## Scri identities and missing boundary conditions

At actual Omega=0, with `n_i=-partial_i Omega`, the kernel computes
`s_i=n_i/sqrt(gbar^{jk}n_j n_k)` from the live regular metric. It reconstructs
`beta^i=beta_trial^i-(s_j beta_trial^j+alpha)s^i`. For a fixed analytic n,

```
partial_t s_i = (1/2)s_i s^k s^l partial_t gbar_kl
s_i beta_t^i = -alpha_t - beta^i partial_t s_i.
```

Writing the driver as `beta_t^i=a^i+mu D^i`, where a includes the *regular
limit* of the restoring term and the same advection stencil, the dependent
trace is `s_i D^i=(-alpha_t-beta^i s_i,t-s_i a^i)/mu`. The kernel reconstructs
that normal component while preserving the supplied tangential components.
It does not prescribe a second independent normal datum. It rejects a
positive-Omega call rather than treating a finite-radius cell as scri.

At normalized Minkowski scri, the limiting 26-variable symbol has **three
incoming, sixteen outgoing and seven tangent** modes. The incoming modes are
one longitudinal and two transverse shift modes. Thus normal nullness alone
leaves two tangential incoming gauge fields. A candidate fixed-reference
tangential shift condition must be accompanied by its differentiated auxiliary
compatibility, including the moving projector. This has **not** been admitted
as a full boundary condition here. Full vector restoring regularity also
requires the corresponding `beta+x` numerator to vanish in the chosen
asymptotic phase. Nullness alone does not enforce that or the geometric
numerators, shear, Z4 conditions or derivative hierarchy.

No endpoint storage has been added to the cell-centered shell. No claim is
made that these point identities are applied after RK stages, ghost filling,
interpolation or AMR transfers. Those integration hooks, the full geometric
scri limit and tangential treatment, RK4/seven-point/KO8 support, complete
shell/gauge outputs and checkpoint/restart are still required. Existing
unsupported-configuration and restart guards remain in force.

## Results and limits

See `evidence/assessment.json`, `evidence/native/audit.json`, the saved test
logs, and `evidence/donor-convergence.png`.

| Check | Result and scope |
| --- | --- |
| Native radial gauge vs donor | 105 point cases, maximum difference 3.42e-13; scalar native symbol differences below 5e-16 |
| Live-normal scri identities | 200 random metrics; nullness error below 1.3e-15, moving-normal finite-difference check; separate 54 face/edge/corner point checks |
| Mapped modes | Three spacings on all six patches; fourth-order J/H derivatives; constant/linear Cartesian modes about an off-center origin; no moving-hole evolution |
| Regular donor reproduction | Equilibrium N65 and pulses N65/129/257 reach t=1; ten-field full-domain endpoint order 2.353 |
| Independent donor timestep test | Fixed N65, t=.15, dt/h=.05/.025/.0125; full ten-field order 3.997 |
| Short donor puncture reproduction | N257/513/1025 reach coordinate t=.05=0.15M; full ten-field endpoint order 3.648; no lapse-floor calls |
| Archived donor 100M campaign | Reviewed, not rerun; exterior convergence survives but puncture endpoint order -0.0549 and growing local constraint peaks remain failures; gauge still evolving |
| Existing finite-radius scalar transport / failed-stage capture | Both pass, 118.83 / 38.83 seconds |
| Existing regular-Z4c/AMR smoke suite | **Timeout at 300.10 seconds** during corner AMR cases; retained as incomplete, not a pass |
| Smaller uniform finite-radius smoke | Minkowski and harmonic wave each complete three native steps; this does not replace the timed-out AMR suite |
| New 3D nonlinear scri evolution | **Not run / not implemented** |

The native mapped operator test is not seam-crossing evolution. The ordinary
finite-radius scalar/regular-Z4c regressions do not test this new gauge.
The short puncture result does not address the late puncture defect. Its r=0
slot stores asymptotic coefficients; exported constraint norms explicitly
describe all positive-radius nodes and do not present an undefined physical
r=0 constraint as a passing Cartesian-cell diagnostic. No positive-radius node
is dropped. The full-domain field comparisons include all ten fields and no
field rescaling; regional norms use their own weights. Time and space tests
have different final times, so no temporal/spatial error ratio is asserted.

Off-center puncture evolution, moving punctures, horizon extraction on the
3D patches, substantial nonspherical disturbances, MPI, GPU and hyperboloidal
AMR/restart remain untested. No Aurora jobs or HPC allocations were used.

## Reproduce the bounded CPU work

```
git submodule update --init kokkos
cmake -S . -B build-scri -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_CUBED_SPHERE_TESTS=ON \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON
cmake --build build-scri -j4
ctest --test-dir build-scri -R 'hyperbolic_shift_3d|cubed_sphere_map' --output-on-failure
ctest --test-dir build-scri -R 'cubed_sphere_regular_z4c|cubed_sphere_scalar_transport|cubed_sphere_failure_capture' --output-on-failure

cmake -S tst/elliptic_shift_1d -B build-donor -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-donor --target spherical_z4c -j2
ctest --test-dir build-donor -R 'hyperbolic_shift_(boundary|moving|gamma_inner|inner_lapse)' --output-on-failure
python tst/hyperbolic_shift_1d/matrix.py \
  --protocol inputs/hyperbolic_shift_1d/gamma4-inner-damping-v1.json \
  --library "$PWD/build-donor/libspherical_z4c.so" \
  --output "$PWD/build-donor/gamma4-reproduction"
python tst/hyperbolic_shift_1d/matrix.py \
  --protocol inputs/hyperbolic_shift_3d/donor-time.json \
  --library "$PWD/build-donor/libspherical_z4c.so" \
  --output "$PWD/build-donor/gamma4-time"
python tst/hyperbolic_shift_1d/matrix.py \
  --protocol inputs/hyperbolic_shift_3d/donor-puncture-short.json \
  --library "$PWD/build-donor/libspherical_z4c.so" \
  --output "$PWD/build-donor/gamma4-puncture-short"
python tst/hyperbolic_shift_3d/assess.py --build build-donor --output evidence-new
python tst/hyperbolic_shift_3d/audit.py \
  --library build-scri/tst/hyperbolic_shift_3d/libhyperbolic_shift_3d_native.so \
  --output evidence-new/native
python tst/hyperbolic_shift_1d/exact_gamma_driver.py evidence-new/exact-donor-scalar.json
build-scri/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mapped > evidence-new/mapped.log
python tst/hyperbolic_shift_3d/plot_audit.py --evidence evidence-new
python tst/hyperbolic_shift_3d/bounded_patch_smoke.py \
  --binary build-scri/src/athena --output evidence-new/small-patch-smoke
```

Use fresh output directories for donor runs. Each run has its own wall cap;
the harness returning zero does not mean a scientific run passed: inspect the
manifest status and failed-stage archive. The expected-rejection input is
`inputs/hyperbolic_shift_3d/guard.athinput`. Run it with `build-scri/src/athena`
to verify the explicit admission guard. Do not remove that guard and infer
that the remaining regular-Z4c runtime implements these point equations.
