# Fixed rho enrichment: robustness gates and rejected alternatives

This is a host reconstruction experiment, not a change to the gauge, physical
RHS, puncture boundary data, or default evolution algorithm. The current actual
mesh qualifies for a further **optional static-centered radial-first** coupled
probe. Generic moving-puncture qualification remains false. No coupled evolution
has yet been run with the supported58 candidate described here.

## Operator and fixed guards

The original local basis is all 56 Cartesian monomials of total degree at most
five plus log(rho). The candidate adds only rho, giving 58 columns. Unknown log
powers are coefficients of log(rho), not evolved/fitted exponents or universal
wormhole powers. Plans are field-independent. They preserve the original active
scalar value and replace only its spatial first/second derivatives. No field
projection, parity prescription, beta pin, background RHS subtraction, or
Minkowski-reference modification is introduced.

Column-scaled, pivoted, twice-reorthogonalized QR must satisfy full rank, condition
estimate <=1e9 and full-basis reproduction <=1e-6. The independently audited
functional L1 limits remain **value 16, h-gradient 32, h²-Hessian 64**. The raw
log-coefficient diagnostic is distinct from those derivative functionals. Every
failed attempt is recorded. No lower-degree reconstruction or silent numerical
fallback is allowed.

The supported58 variant tries the ordered nearest-active Cartesian prefixes
125, 150, 175, 256 and keeps the first fully qualified plan. Prefix ordering is
(distance squared, active ordinal). All 58 columns, including rho, remain exact
within the reproduction certificate. It differs from the bounded blend below.

## Why the earlier candidates were not adopted

The strict 125-donor rho candidate qualified 25/64 translated manufactured
footprints, rejecting 35 on Hessian and four on gradient amplification. Its raw
uncapped diagnostic qualified all 64 algebraically, but reached gradient L1
44.06 and Hessian L1 256.61. The full odd rho+rho³+rho⁵ basis reached Hessian L1
1453.97 and also had 221 actual-mesh reproduction failures. Neither an improved
radial answer nor survival of a run justifies discarding these guards.

The separate geometry-only convex derivative blend kept the same 16/32/64 caps,
with one theta per scalar jet. It qualified all 64 footprints and all 2728 actual
rows, but loses rho exactness when theta<1. This is a decisive changing-regularity
falsifier. For q=p log(rho)-0.7rho+degree-five polynomial, origin eight-query
maximum physical Hamiltonian error at h=.04,.02,.01,.005 was:

| p | h=.04 | h=.02 | h=.01 | h=.005 |
|---|---:|---:|---:|---:|
| 4 | 1.405e-6 | 1.778e-7 | 2.235e-8 | 2.800e-9 |
| 2 | 9.584e-4 | 4.852e-4 | 2.440e-4 | 1.223e-4 |
| 1.09 | .018381 | .017489 | .016525 | .015568 |
| .5 | .123946 | .177535 | .252525 | .358103 |

The p=.5 error diverges approximately as h^(-.5). One angular zz, exponent1.1,
h=.02 paired case increased from 2.339e-7 to 4.974e-5. The blend must **not** be
integrated as the generic puncture derivative fix. These failures do not prove
that every other fixed-basis reconstruction must fail.

Frozen strict/raw/blend evidence, complete powers, angular paired comparisons,
noise and a plot are linked from
[evidence/rho-log-tradeoff-report-20261027/summary.json](evidence/rho-log-tradeoff-report-20261027/summary.json).
The first actual raw evidence has meaningless alpha-row `HChi(alpha)` output;
those columns must be ignored. Its all-domain/core H norms always used chi and
are unaffected. The subsequent diagnostic source prints H only for chi.

## Supported58 manufactured qualification and explicit failures

The 64 near-puncture footprints use h=.04,.02,.01,.005, eight bracketing cells,
and puncture positions (0,0,0) and (.037,.013,-.021). Fields include six arbitrary
log powers, fixed rho+nonradial polynomial, radial fractional and three angular
fractional families. Deterministic independent sample noise is bounded by 1e-12.
A true 256-nearest catalog is taken from a 9³ Cartesian donor lattice.

Supported58 qualifies 59/64: 25 with125 donors, 32 with175 and two with256. All
five remaining translated footprints are hard failures even at256 donors:

| h | octant | rho/h | rank | condition | gradient L1 | Hessian L1 |
|---|---:|---:|---:|---:|---:|---:|
| .04 | 6 | .460299 | 58 | 1676.8 | 30.357 | 136.435 |
| .04 | 7 | .601560 | 58 | 1555.8 | 13.606 | 73.098 |
| .02 | 0 | .589491 | 58 | 2185.2 | 23.592 | 88.996 |
| .01 | 2 | .489898 | 58 | 1808.0 | 26.963 | 90.410 |
| .005 | 1 | .331662 | 58 | 1285.5 | 32.444 | 164.321 |

All are full-rank and violate functional limits, not rank limits. Every translated
resolution group is incomplete. Its complete eight-query maximum and physical
volume RMS are therefore **null**, with missing octants recorded. Supported-only
subset norms are labeled as such and are not acceptance norms.

For complete origin eight-query groups, physical-volume weights are
h³ exp(-1.5q), with q=log(chi). The initial radial wormhole H maximum errors are:

| h | original125 | supported58 | baseline at matched donor count |
|---|---:|---:|---:|
| .04 | 2.072807 | .456412 | 2.196835 |
| .02 | .778981 | .087930 | .836059 |
| .01 | .177996 | .007871 | .192652 |
| .005 | .030267 | .000420 | .032898 |

Supported orders for these four related resolutions are 2.376, 3.482, 4.229.
This is a local initial-data diagnostic, not full 3D evolution convergence.
For p=1.09 and p=.5 plus rho, maximum H errors on all origin queries stay below
1.7e-11; the large changing-power errors from the blend disappear when rho
exactness is retained.

The three angular modes remain regularity-limited: measured orders track their
fractional exponents .3,1.1,2.4, not uniform fourth order. At h=.005, angular_xy
physical-volume RMS for exponents(.3,1.1,2.4) is respectively
(.0074168,5.8471e-5,2.2713e-8), versus original125
(.0196709,.00016916,7.3954e-8). Matching donor support matters: angular_zz,
exponent1.1 has RMS1.0592e-6 for supported58, versus original2.7274e-6 but
matched-support baseline3.3742e-7. Thus the rho basis has a measurable angular
bias even when its original125 comparison improves.

All complete/incomplete group maxima and physical-volume RMS, arbitrary powers,
matched-support controls, unsupported footprints and provenance are in
[evidence/supported-rho-log-report-20261027/summary.json](evidence/supported-rho-log-report-20261027/summary.json).
The [convergence/control plot](evidence/supported-rho-log-report-20261027/supported-rho-controls.png)
was visually checked. It deliberately plots only complete origin8/8 groups and
labels the five translated failures.

## Actual mesh and all-cell initial Hamiltonian diagnostic

Actual N6, seven shell layers {.45,.55,.60,.65,.70,.75,1}, bulk2 with three
puncture refinements has 47088 active cells. The radius.24 value-only Cartesian
coarse/fine halo cache is unchanged. The existing local derivative selection
rho<.1 and rho/h<=8 selects2728 cells. Core reporting rho<.12 includes18368
cells; every full-domain cell remains in acceptance norms.

All2728 actual candidate rows qualify:2720 at125 donors and eight at175.
Sixteen failed125/150 Hessian attempts are retained. Max condition1.37168e7,
full58 reproduction6.608e-10, gradient L1 17.2997, Hessian L1 63.0194.

| derivative | all-domain H RMS | all-domain H max | core H RMS | core H max |
|---|---:|---:|---:|---:|
| original125 | .02961996 | .04782254 | .03202869 | .04782254 |
| baseline at matched support | .03202719 | .04782254 | .03465678 | .04782254 |
| supported58 | .00299250 | .04362334 | .000542174 | .008024483 |

Norm weights are the original mapped physical cell volume divided by chi^(3/2).
The full maximum after reconstruction is unchanged shell point126. The core
maximum is unchanged ordinary-FD point39340 at rho/h=8.411, outside the current
override. It is not an accepted candidate row or a rejected row. The old maximum
point34919 at rho/h2.598 is accepted and improves locally from -.04782254 to
-.000735134. No cells are dropped. Initial conformally flat H is evaluated with
the actual khat/Arr curvature terms; the scalar identity was independently
checked against the tensor constraint kernel in the preceding halo diagnostic.
It is not a conformal-flat restriction on later tensor evolution.

The complete actual run took42.15s including29.30s value-halo construction.
This source/operator differentiation diagnostic does not establish a passing
elliptic solve, coupled evolution, moving puncture, or changed scientific target.

## Actual-field noise sensitivity

All2728 changed rows, including the eight enlarged rows, are audited with
independent deterministic chi/lapse sample noise <=1e-12. Unchanged rows retain
the old operator. Max actual H perturbation is1.0720e-9 versus baseline1.0752e-9;
physical-volume H-noise RMS2.1659e-11 versus2.1539e-11. Max principal Penrose
lapse-Laplacian perturbation is2.7090e-10 versus2.6457e-10. This is the principal
lapse contribution, not the complete evolution RHS.

Actual initial A is zero on all selected rows, so the actual momentum response
to this chi-gradient perturbation is exactly zero. The conditional unit-A
momentum amplification gain instead rises from1802.1 to4302.9; a later nonzero-A
run must measure its real effect. It would be misleading to infer generic
momentum-noise robustness from the initial zero alone.

Source-noise raw data and manifests are frozen in
[evidence/supported-rho-log-source-noise-20261027](evidence/supported-rho-log-source-noise-20261027/manifest.json).
The supported deterministic scaled derivative noise maximum is4.4091e-12.
No functional cap, scientific tolerance, reference, source or active value was
changed to obtain these numbers.

## Reproduction commands and status

Run from the isolated repository, using its configured build and unchanged
bounded Serial CPU libraries. Every build and diagnostic process has a180s
limit and at most50000 active cells. Use fresh evidence directories:

```sh
python3 tst/elliptic_shift_3d/run_rho_log_diagnostic.py --build build-shift-3d --supported --output /tmp/supported-rho-mms
python3 tst/elliptic_shift_3d/run_rho_log_diagnostic.py --build build-shift-3d --supported-actual --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat --output /tmp/supported-rho-actual
python3 tst/elliptic_shift_3d/run_rho_log_diagnostic.py --build build-shift-3d --supported-source-noise --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat --output /tmp/supported-rho-noise
```

Each evidence directory contains frozen transitively included source, source and
library hashes, exact compile/run commands, exit status, timings and raw results.
Successful diagnostic exit means the experiment completed; scientific acceptance
remains false. Generic translations, dynamic regularity in coupled evolution,
nonzero-A noise propagation and the optional coupled/restart probe remain open.
Original1D jobs and worktrees were not touched.
