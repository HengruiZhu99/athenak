# Inner relative-lapse advection audit

This experiment retains the Minkowski hyperboloidal reference, C0 tensor
kernel, kappa_input=10, kappa2=0, spatial-norm shift and physical-P lapse/storage.
It changes one regular lapse source through the nonflat reference transition.
It does not provide a stable finite-pulse formulation or black-hole transition.

Let h=alpha_ref, W=W_gauge(.45,.85), c=1-W and G_i=partial_i h. Add

```
Delta alpha_dot = -c * [(alpha/h)*beta^i-beta_ref^i] * G_i.
```

The helper evaluates the equivalent difference form
`-c*[(beta-beta_ref)+beta*(alpha-h)/h] dot grad(h)`.
Together with the existing analytic-reference advection subtraction this gives

```
c*beta dot [grad(alpha)-(alpha/h)*grad(h)]
+W*[beta dot grad(alpha)-beta_ref dot grad(h)].
```

At c=1 this is alpha*beta dot grad(log(alpha/h)). The physical-P lapse pole,
regular logarithmic relaxation, shift, geometry, continuation, KO and final-only
RK3 algebraic projection remain unchanged. The exact reference addition is zero.
The source is exactly zero in the flat geometric core r<=.05 (G=0), and in the
outer collar r>=.85 (c=0). The gauge core r<=.45 includes nonflat reference
geometry and is distinct from the exact Cauchy core.

This is algebraically regular for finite positive lapse and positive h. It
introduces no live-lapse or Omega denominator and changes no principal spatial
coefficient. Its only value-Jacobian entries are

```
Delta J_alpha,alpha = -c*(beta dot G)/h,
Delta J_alpha,beta_i = -c*alpha*G_i/h.
```

The existing conformal-Q gauge branch already contains log-relative advection;
this experiment combines it with the stabilized physical-P pole. No exact prior
physical-P hybrid test was identified in the checked history.

## Local and native checks

The frozen mathematical gate has 10 passing commands and 376 unchanged inputs.
Release and ASan/UBSan checks cover 4004 nonlinear samples; all 360 principal
cases retain the complete basis. Among 1900 actual full20 matrices, the source
changes only the predicted regular lapse value row (normalized discrepancy
2.64e-15). All 380 independently retained baseline matrices are bitwise equal;
outer/core matrices and four leading scri pole matrices are unchanged. Both
native span2.1 and global span2.2 Nyquist frequencies are explicitly sampled.
All 132 sampled nonpositive-root scalar RK3 tests pass. Sampled positive
primitive roots slightly increase, from 3.145118116 to 3.147540032 at a=.5 on
the reference; this gate establishes implementation identity, not stabilization.
A failed Python report-rendering attempt is preserved separately from the final
passing numerical checks.

Six native objects compile in 9.6242 seconds. Removing the helper include and
one assembled `gauge_rhs.alpha` addition restores the entire production header
byte-for-byte. All 369 baseline source/input/helper files (365 src/CMake files plus four
auxiliary files), four norm overlays, 268 repository
dependencies, 182 original objects and four Kokkos libraries remain unchanged.
The private executable SHA256 is
`604846c7bc9f6f19042d35de9aa83fc0d95909a4d87f74d775ca8f737cf48203`;
its build receipt is
`d9cf0f305fb0474c4a24e961249ab0fca5ab4b53989241c77a9090e323b37e89`.
Build and native launches use HEAD2392ccd1 and implementation27c19d20.

The reference t=.05 completes in 17.0522 seconds with maximum binary64 drift
1.309329595e-14. The finite angular pulse t=.02 completes in 7.0640 seconds:

| Constraint | Candidate | Ratio to same-grid C0 |
| --- | ---: | ---: |
| H | .003101094546 | .996933995 |
| M | .004872043801 | .995503707 |
| Z | .001194585133 | .997147533 |

Each run’s three private binary64 snapshots pass finite fields, positive lapse/chi, SPD,
determinant/trace and exact BIN-cast checks. Initial active fields, coordinates,
masks and geometry match baseline bitwise. Only output cadence differs. Historical
physical_metric_eigen audit keys refer to Penrose gtilde/chi magnitudes; positivity
is equivalent to physical SPD for Omega>0.

The native recipe's initial Python parse and receipt-key guards, and the first
auditor's index-schema guard, fail before compile or snapshot checks respectively.
Their exact attempted scripts and failure observations are preserved. Corrected
scripts verify the actual gate schemas without modifying mathematical receipts,
source or evolution. The compact native freeze retains redundant staged-document
captures as hashes/metadata, with the original full-copy index preserved locally;
compiled source and run evidence are copied exactly.

## Global screen and outcome

The actual full22 native RHS and final-only RK3 tangent agree to 7.25e-10 and
1.91e-10. Exactly 3872 sparse entries change, all local alpha<-alpha/beta;
the analytic row prediction agrees to 7.06e-13. Every non-alpha row and every
outer row is exactly C0. The global N16 mesh has no exact-core cells, so core
identity is established by the separate local gate. The initial gauge action
changes only lapse; its instantaneous geometric constraint source is unchanged.

Short projected-continuous action at .025/.05 agrees with independent canonical
Taylor action to 1.612e-14. The longer t=2 Arnoldi screen is exploratory, with
local truncation checks and no long independent canonical comparison:

| Spatial-norm gauge | H | M | Z |
| --- | ---: | ---: | ---: |
| Lapse candidate | 2.94467448 | 1.63065867 | .36574625 |
| Ratio to continuous C0 | 1.34297 | 1.33616 | .98928 |

The configuration-H1/momentum-L2 component amplification increases from 35.7290
to 46.0072. This component norm is not an invariant tensor energy. Shell H/M/Z
ratios are 1.33388/1.44384/1.37682. These substantial deteriorations reject the
candidate and do not justify a long native/canonical run or production adoption.

## Puncture and scri scope

The additive implementation can subtract O(1) regular advection terms when the
live lapse collapses in the nonflat transition. An independent binary64 example
at alpha=1e-300 leaves about 1e-15 instead of the approximately 1e-300 direct
blend value. A future black-hole implementation would need to evaluate the
combined direct advection expression above and pass fresh checks. Finite-positive
pulse gates do not admit puncture-scale cancellation or establish lapse positivity.

Because the exact Cauchy core source is zero, the
[inner trumpet driver obstruction](hyperboloidal-inner-trumpet-calibration.md)
is unchanged. Outer compatibility and the existing finite-Q counterexample are
also unchanged. There is no exact-scri closure or wormhole-to-trumpet formation
claim. The Minkowski hyperboloidal reference remains in use throughout.


## Evidence

The [immutable archive](validation/hyperboloidal-lapse-and-composed-diagnostic-experiments-20261009/README.md)
contains 228 cataloged files (5,358,327 bytes), with catalog SHA256
`c5c433973b6bd7295b47e5124e9a0b1df2324976f3b1531d73da259d0767243b`.
It includes this candidate's local gate, exact private build/launch/audits and
global screen. A separately scoped
[composed-diagnostic attribution](hyperboloidal-composed-diagnostic-audit.md)
reads earlier saved states and performs no evolution. Neither changes production.
