# Physical reference wave-map gauge: native preflights

All eight prescribed native reference/short angular-pulse preflights pass, including readback of all 64 saved binary64 restart arrays. The stationary Minkowski reference remains at floating-point residual levels. Large-pulse constraint norms decrease across the three grids at t=.02, with most spatial-Z error concentrated in the outer collar. These short controls admit the fixed t2 comparison; they do not establish long-time stability or a regular exact-scri formulation.

The branch is `z4c_hyperboloidal_layer`. Production `src/` and root CMake remain byte-identical to implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`. Private native builds were prepared from HEAD `a0f8fc8464665db3104fdbdaf142661259a6a399`; the actual-array probe and evolutions were launched from `e654ceb0602c5fc47d8e0c600aedda660a96c92f`. The preceding [consistency audit](hyperboloidal-wave-map-consistency-audit.md) records the constrained harmonic principal system and raw nonlinear manufactured RHS gate.

The eventual acceptance target remains a substantial angular gauge disturbance on Minkowski, followed by a black hole surviving the inner wormhole-to-trumpet transition with the Minkowski hyperboloidal reference throughout. This globally harmonic diagnostic gauge does not supply the required inner puncture gauge.

## Actual native binding and builds

The private Cartesian overlay replaces only the four lapse/shift RHS rows, using the unchanged physical-reference wave-map helper SHA256 `56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28`. Coordinates are taken from the same grid expression as the reference evaluation. `ReferenceConnection`, `Gauge` and `rwm::Assemble` include each lapse and shift pole exactly once. The geometric equations retain physical `P=K-2Theta`, C0, kappa input 10 with damping argument `10/alpha`, and kappa2 zero.

The native C0 patch subtracts its analytic Minkowski geometric floating-point RHS residual. The earlier 48-point manufactured gate used the raw geometric RHS without that subtraction. The present reference runs explicitly check the integrated subtraction path; the wave-map gauge itself uses factored analytic stationarity.

Exactly six CartesianPatch-consuming translation units are rebuilt per recipe. The old forced spatial-norm injection is removed from every selected compile command; 176 other objects and four libraries are reused with recorded hashes. The matched C0 recipe uses the public Cartesian header byte-for-byte. All three CPU Serial/double, arm64, C++17 Release builds compile and link successfully, with empty stderr. Independent review checks the dependency closure, inverse Cartesian patch, half-step-only source diff and all 17 prepared input files.

| Private executable | SHA256 |
|---|---|
| wave-map | `829d590ee02e7865744ee9ea8e2bb00e036ee0dd9b7c2609da60740c15138b8c` |
| matched C0 | `9297e0389dd48e65a5c650b5f7aa3ffb0d4b4c6c358fdd6e31044318ac62bade` |
| wave-map, half cap | `256527de2e7476392f15d1b7800e8a1372d1834f2e91a46d20b5a1b696869bee` |

Fourth-order centered/mixed derivatives, native upwind advection, sixth-order KO with coefficient .1, RK3, final-stage-only active-cell algebraic projection, and symmetric degree-2 spherical continuation remain unchanged. Ghost metrics receive no projector or characteristic/constraint compatibility enforcement. Existing uniform single-3D-block, ng3, CPU Serial/double, vacuum and no-floor restrictions remain active; MPI, AMR, GPU and matter are unsupported.

The unchanged hyperboloidal timestep cap is `min(.025*h/max_speed,.03*Omega_min)`. Input CFL cancels in this branch, so changing that input alone would not halve the step. The separate half-cap build multiplies the result by .5 inside the hyperboloidal branch, halving both caps. Its gauge equations are identical. Endpoint clipping and full-precision history/restart dt are retained; the six-digit console and a recomputed live-state cap are insufficient dt comparators.

## Compiled array seam and independent readback

The compiled probe calls actual CartesianPatch RHS on loaded arrays at six declared N24 cells, for reference, small and large inputs: 18 cases. Its comparator explicitly assembles regular plus pole/Omega gauge rows, geometric reference subtraction, single upwind correction and KO across all 22 rates. Every seam difference and the reference RHS is zero, against fixed `2e-12` and `1e-10` gates. Omitting a shift pole produces sensitivity .3039394; duplicating the lapse pole produces .1172157, both above `1e-8`. The standalone seam array initializer copies the production formula; later actual Mesh initializer output is checked separately.

The probe executable SHA256 is `584e74bc257e7661310fff684af6d5ccf12c18dd24886a7e0ae9941c87161a54`; its successful seam receipt is `54a7f3f955c294bcc45307986ec33f4d7a2fb5e10d6626c89df94a68061a2b73`. The final snapshot analyzer is `c1b9487717e85e920b274b7dcb429290eed6b0d96b3b6cbd8e00d6ea352f7c72`. Independent review covers all-axis region sizes, layout, successful same-source probe receipt and before/after source, executable, dependency, input, restart and history hashes. Earlier guard versions and the mechanical review hash race are preserved.

For all 64 saved arrays, native Prepare/derivative/Constraints calls produce physical H and Theta and M/Z norms contracted with the Penrose inverse spatial metric `barGammaInv=chi*gtildeInv`. Eigenvalue diagnostics separately record the unit-determinant Z4c metric `gtilde` and Penrose metric `barGamma=gtilde/chi`. Native/history RMS agreement is within `1.735e-18`. A second reader uses an independent manual struct/layout parser, leading-principal-minor SPD checks, cofactor inverse, determinant and trace calculations. It passes all eight cases without importing the original reader or calling the kernel. All actual t0 initializer fields agree exactly with the prescribed 25-field state.

The production pole history covers geometric C0 poles only. Separately labeled snapshot gauge-pole numerators are values-only diagnostics; they do not test full differentiated gauge sources or prove pole cancellation.

## Fixed configuration and short results

All runs have mass zero, S=1, a=.5, geometric transition .05--.95, gauge parameters .45--.85, and cube `[-1.1,1.1]^3` with N16/N24/N32. The wave-map gauge ignores the old source flags; the matched C0 comparator uses physical-trace lapse and preferred source off. The production initializer uses

```
s=(1-r^2)^4 exp(-r^2/.35^2),
delta alpha=A*s*(1+.2*x+.3*y*z),
delta beta=B*s*(1+.3*y*z,.2*x,.1*x*y).
```

Large amplitudes are A=.2, B=.1; small amplitudes are .02, .01. These are nonradial pulses with fourth-order vanishing at scri, rather than compact support strictly inside it. The reference has A=B=0.

There is no exact Cauchy-core cell at these resolutions: minimum radius is .1190785, .0793857 and .0595392, respectively, above r0=.05. Separate local core checks remain the evidence there. Minimum Omega is .0026953125, .002170138889 and .003876953125, so it is not monotone under this spherical-mask refinement. The corresponding pole caps are `8.0859375e-5`, `6.51041667e-5` and `1.1630859375e-4`. These sampling differences preclude inferring a convergence order from three short norms.

Four references run to t=.05 with 11 saved arrays each; the three large and one small pulse run to t=.02 with five each. Every native process exits zero. Reference gates require drift <=`1e-10`, all H/M/Z/Theta RMS <=`1e-9`, and determinant/trace residuals <=`1e-11`. Pulse gates require completion, finite fields/diagnostics, positive lapse/chi, SPD metric and determinant/trace residuals <=`1e-10`; they impose no reference-drift threshold on the pulse.

| Reference | Maximum full-field drift | Maximum H RMS |
|---|---:|---:|
| wave-map N16 | `5.718e-15` | `2.840e-14` |
| wave-map N24 | `2.382e-14` | `7.731e-14` |
| wave-map N32 | `2.915e-14` | `1.520e-13` |
| matched C0 N24 | `2.382e-14` | `7.731e-14` |

All native reference determinant and trace residuals are below `1.10e-15`; the independent cofactor calculation also passes the fixed algebraic-normal gates. The reference lapse minimum is about .981 and metric eigenvalues remain positive.

| Pulse at t=.02 | H RMS | M conformal RMS | Z conformal RMS | Theta physical RMS |
|---|---:|---:|---:|---:|
| large N16 | `.0153498` | `.0287690` | `.00731711` | `.000265354` |
| large N24 | `.00574662` | `.00734875` | `.00213195` | `.000113489` |
| large N32 | `.00308322` | `.00246441` | `.000772351` | `.0000457479` |
| small N24 | `.000579585` | `.000735245` | `.000213721` | `.0000112405` |

Across saved large-pulse samples, lapse remains >=.999518, chi >=.761074 and the minimum eigenvalue of `gtilde` >=.761105. Algebraic normals remain around `1e-15`. The small N24 pulse has lapse >=.983276 and positive metric. These are finite-duration field guards, distinct from relative improvement or stability acceptance.

At the large-pulse endpoint, H maxima occur at radii .2280, .1998 and .1786. The fractions of squared Z norm at r>=.95 are .8587, .8817 and .9388. Thus bulk H and outer Z errors need separate assessment. Finite-radius null-condition deviations are also recorded; they do not establish an exact-scri compatibility law or identify a unique source of error.

## Continuing controls and evidence

The nine fixed t2 runs are launched separately: large wave-map and matched C0 at N16/N24/N32, wave-map N24 half cap, small N24, and a wave-map N24 reference. During this checkpoint the wave-map N16 run failed its native positive-ADM-state guard at t=`1.3671453700579306`, cycle 16910, near r=.98, with negative lapse. Other controls remain ongoing or queued. The failed run and partial outputs are preserved in their separate working prefix; this compact preflight archive contains prepared long-input/source copies but no t2 evolution data. No t6 or t12 continuation is admitted here, and short-preflight success does not override that longer-run failure.

The predeclared N24 useful-improvement gate requires every final H/M/Z <= matched C0 and at least two reduced by 20%. Continuation to t6 additionally requires N32 final H/M/Z <=.8 times N24, half-cap N24 differences <=10%, and no H/M/Z/Theta RMS more than doubling between t1 and t2. Zero-denominator comparisons use absolute `1e-10`. A later t12 continuation requires the t6 last-unit-window maxima <=1.25 times the preceding unit window, retained resolution/timestep controls and review. Passing any finite-time gate would still leave long-time, continuum and exact-scri questions open.

The [compact evidence archive](validation/hyperboloidal-reference-wave-map-native-preflights-20261009/archive-README.md) has 740 files, 24,095,462 bytes, and 304 finite JSON files. Catalog SHA256 is `fc9de674ff19ce3bd9e6cf51b4ccd489f5ff5820748aa09e0b7f1a8ccd372349`. It retains sources, commands, logs, readbacks and failures byte-for-byte. All 278 arrays/compiled payloads, including 64 restart and 192 visualization files, remain metadata-only; no copied file exceeds 1 MiB. All 1,013 original inventory entries and 1,675 dependency/executable identities pass before/after rehash. Compact verification does not replay omitted arrays or rebuild executables.

Production equations have not changed, so previously passing production regressions were not repeated. No lower-order wave-map generator or boundary stability certificate is established by these preflights. Exact-scri null/shear/Z4/Theta closure remains unresolved. The black-hole stage still requires constraint-consistent initial slicing and a justified inner gauge through wormhole-to-trumpet adjustment, retaining the Minkowski reference without a black-hole fixed-point RHS subtraction.
