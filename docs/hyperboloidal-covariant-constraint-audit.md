# Covariant constraint terms: finite-Omega experiments

Production remains implementation27c19d20, with the validated evidence through
branch commit aef47b0a. These experiments retain the Minkowski hyperboloidal
reference, physical-P lapse stabilization, private spatial-norm gauge,
fourth-order derivatives, ng3 symmetric quadratic ray ghosts, KO.1 and native
final-only algebraic projection. No C1 production option is adopted.

The preceding tensor identity gate derives mechanical C_Z4c=1 additions plus
the separate connection repair needed for physical spatial-Z covector
transport. This stage examines the actual finite-Omega stiffness and native
reference/short pulse. It supplies no nonlinear scri closure or black-hole
transition acceptance.

## Local matrices and transient amplification

Exact actual-kernel dual derivatives produce 1920 continuum Fourier matrices
and 384 small-Omega matrices, with reference and unrestricted finite-Theta/Z
jets. These are local primitive generators, distinct from coefficient-aware
constraint propagation and the global native stencil/ghost operator.

The Lambda-Theta matrix entry has a genuine double pole. At S1,a.5,
Omega²*d(Lambda_r,t)/dTheta tends -8. Nevertheless the sampled spectral radius
scales as 1/Omega. The analytical similarity

```
T = diag(1 on alpha/chi/P/Theta/beta/gtilde, Omega on Atilde/Lambda),
B = Omega T L T^-1
```

stays bounded in the sampled zero-frequency reference/off-constraint sequence.
It is not a uniformly equivalent unweighted norm at scri and is not imposed on
runtime fields. All 5040 sampled scalar RK3 tests for nonpositive-real roots
pass at the actual N24/36/48 pole.03 timesteps. Small positive frozen roots
remain; the largest repaired/norm Fourier real part is 2.8093. They are not
classified as physical subsidiary eigenmodes.

Independent exact propagators show substantial raw transient amplification:

| Grid | C0 raw RK step norm | Repaired C1 raw RK norm | Repaired C1 exact norm |
| --- | ---: | ---: | ---: |
| N24 | 1.52697 | 12.95109 | 12.99668 |
| N36 | 1.53239 | 48.30424 | 48.48761 |
| N48 | 1.53270 | 57.01233 | 57.22983 |

At Omega1e-5, the raw C1 RK norm is18583.23 while the weighted norm is1.52545.
The corresponding exact semigroup also has inverse-Omega amplification. This
is not an eigenvalue Omega-squared timestep restriction or a transient that
vanishes by reducing the timestep over a fixed physical interval. Weighted
RK-versus-exact one-step differences are about1.52%, reaching2.08% in diagnostic
k256 samples; C0 has similar fast-mode temporal errors.

The gate therefore supports bounded strict-interior exploratory tests at fixed
mesh/step, not uniform raw-norm stability. Kappa5 additionally has a positive
leading pole and is not used for the native C1 experiment.

The first immutable report copied a374-input count from the earlier identity
gate. Its own unchanged numerical receipt has370 inputs. A preserved v2
snapshot corrects that prose and explicitly reports the kappa5 leading pole;
mathematics, executable, matrices and numerical receipts are unchanged. The
native build records its actual v1 pin, and the launches additionally pin v2.
No as-built receipt or first frozen snapshot is rewritten.

## Actual native preflights

Six native objects are privately recompiled, with182 original objects and four
Kokkos libraries verified unchanged. The C1 delta is added after existing C0
Minkowski reference roundoff subtraction. No C1 reference RHS is subtracted.
All369 baseline sources, four norm-overlay files, three private headers,269
compiled repository dependencies and all link replacements are independently
verified. Compiled production identity is27c19d20; launch HEAD isaef47b0a.

The t=.05 reference completes in18.007 seconds. All three binary64 RST
snapshots are finite/positive/SPD and have maximum active drift1.49342e-14.
Final H/M/Z is8.425e-14/2.873e-14/1.347e-15. Initial fields, coordinates and
masks are bitwise identical to the baseline.

The t=.02 angular pulse completes in7.001 seconds:

| H/M/Z | C1 value | Ratio to unchanged same-grid C0 |
| --- | ---: | ---: |
| H | .00318248706 | 1.02309991 |
| M | .00668864913 | 1.36669030 |
| Z | .00168289263 | 1.40474897 |

Theta increases by28.9%. The three private snapshots versus two baseline
snapshots pass all active binary64 state checks; every BIN field is the exact
float32 cast. Initial fields/coordinates/masks and geometric initial data are
bitwise identical. Only output cadence differs, and the endpoint is exactly
t=.02. Historical eigenvalue receipt keys denote the Penrose spatial metric;
physical metric positivity follows by its positive Omega^-2 scaling.

Passing integrity/reference checks does not accept the pulse. The early
constraints are worse, so this experiment does not justify a long native run
without separate evidence.

The inherited shell pole columns still describe C0 terms; they omit the
candidate's added simple/double poles. They are not used to assess C1
compatibility. The H/M/Z constraints and binary64 state checks above apply to
the actual evolved C1 fields.

## Global screen

A separate actual full22 Cartesian consistency gate retains the same N16,
span2.2 stencil/ghost/projection operator as the preceding audit. Its native
raw RHS/Jv and final-only RK3 derivative sweeps pass. Projected continuous20
short propagation at t=.025/.05 agrees with independent canonical Taylor
action within2.78e-14. This is distinct from finite-RK native evolution.

The longer t=2 Arnoldi results are explicitly exploratory: local coarse/fine
truncation checks pass, but an independent long canonical comparison is not
run. They provide a negative candidate screen, not a rigorous long forward-
error bound or exact native t=2 result:

| Gauge | C1 H/M/Z | Ratios to verified continuous C0 H/M/Z |
| --- | --- | --- |
| Production physical-P | 1.64896 / 1.40380 / .349237 | .9584 / 1.1528 / 1.2813 |
| Spatial norm | 2.33509 / 1.87304 / .591101 | 1.0650 / 1.5348 / 1.5988 |

The chosen configuration-H1/momentum-L2 component norm is slightly smaller
for production and essentially unchanged for the norm gauge. It is not a
constraint energy. Outer momentum/Z localization strengthens. The native
short-pulse worsening and these exploratory negatives do not justify a full
native long-pulse or expensive canonical long run of this candidate.

A copied build command initially targeted an old scratch oracle path; the
frozen original was immediately restored and both original hashes reverified
before candidate use. The correction and original command are retained. No
production executable/source or frozen archive changed. Candidate output
paths are separate.

## Constraint weights are not arbitrary Theta falloffs

An independent undamped Minkowski covector-wave check gives an exact example
outside R=0: Z=(F(T-R)/R)dT in inertial Cartesian coordinates satisfies Box Z=0.
With R=r/Omega, A=sqrt(Omega²+b²)=alpha_ref (the conformal Minkowski lapse),
h_R=b/A and L=Omega-rOmega', its hyperboloidal fields are below. The physical
reference lapse is A/Omega.

```
Theta_phys = -A F/r,
Z_r = h_R L F/(r Omega).
```

Thus physical Theta has a finite nonzero scri limit while spatial Z can grow as
1/Omega. Symbolic identities and12 coordinate/normal contractions through
Omega1e-40 pass at100digits. This is an undamped example; the actual positive
kappa hyperboloidal-normal damping can change its asymptotic behavior.

F(T-R)du/R alone is generally not an exact covector wave: its spatial inertial
components have an l=1 angular residual. The exact gradient solution
d(F(u)/R) includes a radial1/R² tail and instead generically gives
Theta_phys=O(Omega), Z_r=O(1). Neither example supplies a universal stronger
Theta falloff for the actual damped tensor system. A compatible weighted
constraint evolution/boundary closure must be derived, rather than choosing a
convenient falloff to suppress the connection double pole.

## Reproducible evidence

The immutable [experiment archive](validation/hyperboloidal-covariant-constraint-experiments-20261009/README.md)
contains 174 cataloged files (4,352,807 bytes), including exact private source,
build and launch recipes, inputs, logs and independent receipts. Executables,
objects and large state arrays are represented by metadata and hashes.
The catalog SHA256 is
`f191ba8998aa725443846ed71d4c8f79139d9bea1006f9ef15976135f1f75af7`.
The private native executable SHA256 is
`84bb958271395e019c1f26d56f50253f64b7e87ade1bfd40d928d443d03918b6`.

Verify the archive using its `verify_archive.py`; do not rerun frozen collectors
or captured scripts in their archived paths. Production remains unchanged.
Stable finite gauge-pulse evolution and a subsequent wormhole-to-trumpet
transition with the Minkowski reference remain outstanding.

The [inner constraint-blend audit](hyperboloidal-inner-constraint-blend-audit.md)
records the subsequent compact-support C1 candidate. Its mixed native/global
results do not establish stabilization; production remains unchanged.
