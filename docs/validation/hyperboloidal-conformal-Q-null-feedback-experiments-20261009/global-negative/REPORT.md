# Conformal-Q/null-feedback actual full-tensor global screen: negative

The independently gated candidate passes the actual source/Jv/native-stage
implementation checks, but its N16 projected-continuous t2 gauge and shell
histories are catastrophic relative to the matched original C0 spatialnorm
control. This is a negative exploratory screen of one finite-Omega discrete
operator. It is not a certified eigenvalue finding, continuum instability
proof, physical energy bound, or native long-time evolution result. No t6
extension or additional native evolution is justified by this result.

## Explicit system and source identity

The grid is the original admitted centered Cartesian N16/span2.2 sphere:
1640 active points, h=.1375, Omega_min=.0026953124999994555, S=1, a=.5,
reference transition .05–.95, symmetric degree2 strict-interior nonrecursive
ghost continuation, native centered/mixed/upwind stencils and KO=.1. C0,
kappa_input=10 and kappa2=0 are unchanged. The original point-major k/j/i
free20 gauge/shell seeds are byte-identical to the frozen baseline seeds.

The input explicitly sets physical_trace_lapse=false, preferred_source=true,
scri_lapse_damping=2, gauge cutoff .45–.85. Both actual native RHS and cached
local Point call the exact frozen
`qnf::Gauge(p,u,g,{.85,.95,5,true})` and `qnf::Assemble` helper. Thus only alpha
is blended between physical-P and Q lapse formulas; the shift also changes
to the preferred-Q regular extension plus sigma5 null-residue beta pole.
The old spatialnorm beta-restoring source is removed. There is no implicit
parameter toggle or combination with C1, kappa2 profiles, or prior lapse/trace
controls. P storage and every geometric equation remain C0.

The candidate replaces only the copied forced-include gauge wrapper. The
production CartesianPatch header is unmodified and byte-verified against
implementation 27c19d20696ea6dd4704032c51dfd026218f64f2. Its analytic geometric
reference subtraction, background reconstruction, ghost filling and projection
lifecycle are preserved. The gauge helper internally compares live fields
with analytic reference values; no additional gauge RHS subtraction is added.
The beta pole is assembled once by qnf::Assemble.

Core gate index dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96
and independent review a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650
were verified before binding/compilation. Shared helper/base hashes are
f3f3acfe16ba3ce36d0b687225aa1e3b33c159781e46d05e1441ccea7c57e049 and
216bb80c18ee33e553de8defd027e6d3e0394eab38a19a56f66699583e0eca7a.
The final recheck covers all 73 core files, 17 review files, 382 gate inputs,
2310 compiler dependency references, both diagnostic callbacks and all 71
original global frozen small files. Full commands, compiler/flags, link
archives, executable and source hashes are retained. No tracked files changed.

## Actual full22, reference and sparse attribution checks

The six inherited implementation-consistency gates pass:

| Check | Observed | Threshold |
|---|---:|---:|
| Prepared reference change | exactly 0 | exactly 0 |
| Maximum projected reference RHS | 6.084e-12 | 1e-10 |
| Raw CSR versus cached stencil matvec relative error | 1.106e-15 | 2e-14 |
| Native22 RHS centered-amplitude sweep, worst best-vector error | 8.952e-10 | 2e-8 |
| Actual final-only RK3 derivative, worst best-vector/time error | 1.903e-10 | 1e-8 |
| Reference restriction times lift identity maximum error | 2.995e-16 | 1e-13 |

The reference H/M/Z RMS are 3.239e-14/1.331e-15/6.643e-16. The small-Omega
amplified reference RHS roundoff is retained above. All 63576 donor references
are strictly interior and nonrecursive; ghost weight sums err by <=1.332e-15.
All servers exit zero. The projector/Prepared amplitude controls retain the
expected quadratic projector convergence and actual ghost derivative checks.

An independent Python expansion uses analytic reference jets, not fitted
matrix entries. It differentiates delta(chi ginv), the weighted null residue,
the preferred regular shift projection, the alpha physical-P/Q blend and
removal of the original norm pole. Only local chi/g/alpha/beta value columns
in alpha/beta rows change; every derivative-slot coefficient is unchanged as
a mathematical function. Every geometry/P/Theta/A/Lambda matrix row is bitwise
unchanged. The raw22 and projected20 sparse differences agree with the
independent formula to 2.252e-9/2.175e-9 absolute, versus maximum predicted
coefficient 11214.48. The many additional tiny sparse differences from cached
finite-difference coefficient roundoff are retained; no coefficient dropping
or replacement stencil is used. Initial native constraint and C_h J_h
component RMS match the original control exactly for all four validation
seeds, consistent with a gauge-row-only change.

Raw22 matrix SHA:
fd16cf31a31533ff5d7d135bdc9f9dfdca5a05fc7db9927c155b65d3250397c4,
14900481 nonzeros. Projected20 matrix SHA:
dbec9872290d912d43ee7c6464bfd7a223eb0880729c3fc460f6e05add29a4ca,
15111151 nonzeros. Exact reference lift/restrict matrices are saved.

## Native finite-stage relationship and action accuracy

The exact cached final-only map is P_ref R3(dt J22) Lift. It is independently
compared with the actual nonlinear native one-step centered derivative over
multiple amplitudes and dt factors 4,2,1,1/2,1/4. It is distinct from
R3(dt P_ref J22 Lift). At nominal dt=.03 Omega_min=8.085937499998367e-5,
their state-unit differences are 1.479e-9 for the gauge seed and 1.539e-5 for
the shell seed. The complete halving table and normal-stage fractions remain
in the full22 validation receipt. The subsequent histories evolve the
continuous projected generator P_ref J22 Lift, not the finite native map.

Independent canonical Taylor actions at t=.025 and .05 use an exact constant
configuration 1/h similarity, undone before comparison. Their maximum
relative state discrepancy with Arnoldi is 2.874e-14; the canonical work costs
95.923s. This similarity changes no spectrum, initial seed or physical norm.

The exploratory t2 action costs 69.167s, 6560 matrix-vector products plus 242
direct curve-residual products. Maximum local 50/80 truncation-pair error is
9.695e-11 and maximum direct curve defect divided by state norm is
1.3254e-10 per unit time. Its prefix agrees with the separately checked short
run to 1.101e-14. These are empirical action controls, not a rigorous
nonnormal forward-error bound. No long independent canonical action is run.
Every saved state is finite, and the 1e12 Euclidean guard is not triggered.
BLAS/scipy floating-status RuntimeWarnings are retained in the original pilot
and long logs; no cause is assigned. The finite-state/receipt checks, direct
residuals and independent short canonical comparisons are recorded separately.

## Matched native diagnostics at t2

H is native physical Hamiltonian RMS. M/Z are native conformal covectors
contracted with the reference Penrose spatial inverse and use unweighted cell
RMS. The field-unit norm is the reference-volume component norm combining
configuration values/first derivatives with momentum values, with native
upper tensor components counted once. It is not an invariant energy or a
proved symmetrizer norm. Both systems use exactly the same definitions.

| Seed | C0 H/M/Z | Candidate H/M/Z | Candidate/C0 ratios |
|---|---|---|---|
| Gauge | 2.192661 / 1.220403 / .369710 | 10866.019 / 31087.375 / 5713.473 | 4955.6 / 25473.0 / 15453.9 |
| Shell | .00250440 / .00228468 / .000996051 | 1580.822 / 4535.785 / 841.236 | 631216.8 / 1985300.7 / 844570.5 |

Gauge field-unit amplification is 423893.1 versus C0 35.7290; shell 18730.0
versus .0169943. Free20 Euclidean amplifications are 675035.97 and 100091.27.
The field-unit norm and all three constraint RMS attain their sampled maxima
at t2 for both seeds. This is not merely an increase from a zero constraint
seed or a change in primitive field units: the constraint-bearing shell
also grows, and matched native constraints become orders of magnitude larger.
It still does not identify a generator eigenvalue or a continuum mechanism.

At t2 the r>=.9 squared H/M/Z fractions are .0564/.7449/.9258 for the gauge
seed and .0562/.7432/.9258 for the shell. H/M/Z peak radii are approximately
.85593/.91981/.99865 in both. Thus H is primarily below the outer shell while
M/Z concentrate near the boundary. Centered diagnostic amplitude controls at
t0/1/2 differ by <=8.208e-10 relative; the native diagnostic operator itself
is unchanged. Full sampled histories, component groups, peaks and localization
are retained in summary.json and diagnostic NPZ metadata.

Root independently owns native preflights; their results are not reclassified
as this continuous generator's trajectories. Pole/principal local PASS does
not rescue this global negative screen. Finite-Fourier transition behavior,
R0/Taylor closure and any altered feedback weight belong to separate gates;
none is substituted here. The present helper, sources and outputs are frozen
unchanged before any future candidate.
