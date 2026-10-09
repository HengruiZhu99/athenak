# Inner tensor blend and damping-profile experiments

These experiments retain production implementation27c19d20, the Minkowski
hyperboloidal reference and stabilized physical-P lapse. The current task is a
stable finite Minkowski gauge pulse, followed by a single-hole evolution through
its inner wormhole-to-trumpet transition with that same Minkowski reference.
Neither stage is accepted here.

## Inner C1 blend

After the full repaired C1 candidate worsened constraints, the next experiment
applies all mechanical C1 additions and the separate spatial-Z covector repair
with the same prescribed coefficient c=1-W_gauge. The coefficient is one inside
r=.45, smooth through .45--.85 and exactly zero for r>=.85. This is a prescribed
lower-order blend; it is not globally the covariant C1 system.

For S=1,a=.5, geometry .05--.95, Omega=1-W_geo*r². On the added terms' support,
Omega>=.2775, so the added double-pole expressions are bounded there. The outer
kernel is exactly C0 even off constraint. No Theta falloff or Omega floor is
imposed. The geometric P RHS receives its C1 trace addition; physical-P storage
and the gauge equations are retained.

The coefficient-aware physical subsidiary derivation includes

```
Delta M_i,t|gradc = gamma^ja (d_a c)(S1_ij-S0_ij)
                   -(d_i c) gamma^ja(S1_ja-S0_ja).
```

The actual full20 reference chain agrees with the independently derived eight
constraint equations on1000 samples, to7.454e-9 at the finest coefficient step.
Omitting grad-c produces a .0129084 relative discrepancy. All200 sampled local
constraint generators have negative real eigenvalues; this is not a global
stability result. The complete360-case principal basis is unchanged. In4004
nonlinear tensor/dual samples, Einstein additions and all outer additions are
zero; the cutoff and its first-three jets are exactly zero at/above .85.
Release and ASan/UBSan checks pass.

Six native objects are privately rebuilt. The native reference t=.05 has three
binary64 snapshots and maximum drift1.30702718e-14. The finite angular pulse
at t=.02 gives

| Constraint | Blend | Ratio to same-grid C0 |
| --- | ---: | ---: |
| H | .003109965971 | .9997859639 |
| M | .004894494948 | 1.0000911451 |
| Z | .001198001228 | .9999990254 |

All active binary64 states are finite, lapse/chi are positive and the metric is
SPD; every BIN field is its exact binary32 cast. Initial fields, coordinates and masks are bitwise equal
to the baseline. This agreement avoids full C1's early worsening but supplies
no useful improvement or stable-pulse acceptance.

The native shell pole diagnostic includes all added terms: its numerator is
C0_pole+c*C1_pole+c*C1_double/Omega. Regular additions are excluded by definition.
This extends the old C0-only diagnostic wherever the finite-width shell overlaps
the blend support. Independent snapshot reconstruction and a nontrivial
point control are recorded separately.


## Global blend screen

The N16 full22 native RHS and final-only RK3 derivative gates agree to
7.25e-10 and1.93e-10 respectively. Every matrix row on the672 active points
at r>=.85 is exactly equal to C0; configuration and gauge rows agree everywhere.
The initial gauge-pulse action is also exactly equal. These checks bind the
actual sampled native operator to the intended support of the lower-order terms.

Short projected-continuous propagation at .025/.05 agrees with independent
canonical Taylor action to2.86e-14. Longer t=2 Arnoldi histories remain an
exploratory screen with local truncation checks, without a long independent
canonical comparison. They are not finite-RK native evolutions.

| Gauge | Blend H/M/Z at t2 | Ratios to continuous C0 |
| --- | --- | --- |
| Production physical-P | 1.691591 / 1.218061 / .2647250 | .98322 / 1.00027 / .97123 |
| Spatial norm | 2.209822 / 1.268840 / .3520807 | 1.00783 / 1.03969 / .95232 |

The configuration-H1/momentum-L2 component amplification remains46.7621 and
35.9798, versus C0 values47.0983 and35.7290. This component norm is not an
invariant tensor energy or a proved symmetrizer. The shell seed has mixed changes, including up to17% worse shell constraints
in the spatial-norm gauge. These results provide no useful stabilization and do not justify
a long native pulse run or production adoption of the inner blend.

## Evidence and limitations

The immutable [experiment archive](validation/hyperboloidal-inner-constraint-blend-experiments-20261009/README.md)
contains174 cataloged files (4,251,706 bytes), with catalog SHA256
`4ff36f435f83ca345371cd917b26486a3fdddc5bb0bc1f2e8f7dd83e42a8defc`.
It includes exact sources, private build/launch commands, native inputs/logs,
independent audits, tensor/principal/subsidiary checks and global screens.
Executables, objects and large state/matrix arrays are recorded by hash and
metadata only. The native executable SHA256 is
`51768a5324dce84e91da885dd79976deb5a6a040c98e07d191e70ec48c9e11f1`.

Use its `verify_archive.py` for read-only verification. Captured files retain
original bytes; do not rerun frozen collectors or scripts in archived paths.
The pole snapshot utility is a Release check; ASan/UBSan coverage applies to
the separately recorded nonlinear tensor gate. Neither validates exact scri.
Production remains unchanged. The completed
[constraint-damping profile audit](hyperboloidal-damping-profile-audit.md)
records the next separate candidate and its marginal mixed outcome. The
[inner trumpet calibration](hyperboloidal-inner-trumpet-calibration.md)
prepares the later black-hole transition without changing the Minkowski reference.
