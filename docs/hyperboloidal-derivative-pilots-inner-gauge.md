# Derivative pilots and the coupled inner gauge

This checkpoint records two completed flat-space derivative pilots and private
principal and nonlinear arithmetic checks of a candidate inner gauge. It does
not repair the failed native wave-map runs in
[the final matrix report](hyperboloidal-reference-wave-map-final-matrix.md).
The later black-hole acceptance target is explicitly a resolved inner
wormhole-to-trumpet transition with the **Minkowski hyperboloidal reference
retained throughout**. A black-hole reference or RHS subtraction is not part
of that target.

## Completed derivative pilots

The fixed-Lorentz-frame Kirchhoff formulation represents the actual native
lapse/shift pulse as four scalar coordinate displacements on physical
Minkowski spacetime. The completed pilots test its differentiated ray
integrands, supplying four values, sixteen first derivatives and forty
symmetric second derivatives at each ray. They do not perform the angular
integration needed to obtain the displacement solution. The ray direction
remains fixed during differentiation. These checks do not independently
evolve the nonlinear Z4c equations.

The old pilot and compact-radius comparison used the same 480 rays in 24
groups, at 60 and 80 decimal digits, with unchanged pulse, quadrature and
derivative formulas. The sample includes an initial origin point, a short-time
transition point and an outer-collar point, with two fixed boosts.

| Completed check | Result | Child runtime |
| --- | --- | --- |
| Original ray solver | 4,744 checks passed | 193.8207 s |
| Compact-radius solver and saved-jet comparison | 38,344 checks passed | 18.9809 s |
| Independent saved-data readback | All 480 ray keys and 28,800 jet components matched within the declared bounds | No oracle rerun |

The independent readback used 120-digit `Decimal` arithmetic. The largest
scaled old/new jet difference was `3.8369807166e-50`; the largest scaled metric
difference was `1.065114715834e-50`. The largest original ray-equation residual
was `5.6496846024350477e-52`. The compact pilot used 160 exact initial roots,
160 safeguarded Newton roots and 160 analytic outer roots; no fallback was
needed on that sample. Original source and saved-output hashes were unchanged.

For the new root variable, let `q=r/Omega`,
`lambda=(T-H(r))/k0`, `y=X-lambda*k_spatial` and `F=q-|y|`. Then

```
F_r = (L/Omega^2) K/k0 > 0,
K   = k0 - (b/A) nu.k_spatial.
```

The solver retains endpoint signs, future-ray checks, the actual compact-radius
and transformed-lambda widths, and the original ray residual. It uses a fixed
16-probe safeguarded Newton stage followed, if needed, by the unchanged fixed
512-step bisection cap. Height quadrature and endpoint signs are numerical;
they are not interval enclosures. Zero width in an analytic branch records an
exact formula branch, not a validated interval certificate.

These finite pilots neither pass the much larger 493,568-root derivative gate
nor establish angular quadrature convergence, global absence of caustics, or
coverage at native coordinate time. That larger gate was still running at the
earlier pilot checkpoint. Its subsequent full run completed all roots and
2,360 checks but failed 156 CMC-control ray/source-jet, exact-derivative and
scalar-wave-trace checks. The original failure and independent diagnosis are
preserved in [the failure capsule](validation/hyperboloidal-original-pulse-derivative-failure-20261009/README.md). A later
four-dimensional inverse must solve
`X+u(X)=Y_reference(t_native,x_native)` and separately check the Jacobian,
future time orientation, coverage and injectivity. Reference time is not
automatically native target time.

The diagnosed CMC control defect has now been corrected in a fresh private
source: the height-gradient factor `b=Omega*q/a` had frozen radius as a scalar;
it now uses the radius jet `Q=sqrt(sum(Y_i*Y_i))`. A reverse source/AST proof
preserves every other derivative-core method, including `NativeGraph`, and
the original control grids, boosts, precisions and tolerances. Independent
source review preceded the new attempt.

The corrected control slice passed 880 checks over 100 rows and 189,440 roots
at 80/110 digits. Enclosing runtime was `1540.031219 s`. Maximum saved local
identity, exact-derivative and scalar-wave-trace errors were `1.95031e-79`,
`4.89248e-27` and `5.71173e-29`. A separate saved-only qualification retained
1,480 original noncontrol checks with zero failures and deferred all 880 old
controls. The two source identities and receipts are separate; the original
2,360-check attempt remains FAIL. Independent saved-result review confirmed
both registries and unchanged pins without recomputing numerical targets or
decoding the control derivative jets.

The [CMC correction capsule](validation/hyperboloidal-CMC-control-correction-20261009/README.md)
preserves the source change, independent reviews, exact commands and results.
It establishes neither a coordinate inverse nor global timelike admissibility,
native RHS agreement or stable evolution. Its unchanged physical-reference
RWM oracle is not a manufactured solution of the compound interior
Bona--Masso proposal.

The completed evidence is in
[the compact capsule](validation/hyperboloidal-derivative-pilots-inner-pencil-20261009/README.md).
Its catalog SHA256 is
`79b850b690bb21b17898506b932370e1f48fa630b0023e0760de64243c4a1b5f`.
It contains 287 files, 6,100,522 bytes and 152 finite JSON files. All seven
NPZ/NPY/JSONL or oversized payload omissions and 315 external dependencies
remain identified by exact hashes. Source and log whitespace is preserved.

## Coupled inner principal family

The earlier weighted core connection driver weakens as `alpha^2 chi` collapses.
Increasing that coefficient alone crosses defective scalar speed coincidences.
The new candidate changes its metric-gradient coupling jointly. In the actual
constrained 20-field symbol, use

```
epsilon_alpha = 1,
epsilon_chi   = 2 mu^2/(1+mu)^2,
q            = (4 mu-2 epsilon_chi)/3,
C            = 2(1+mu)^2/(4 mu^2+5 mu+3).
```

For finite `f>0, mu>0`, define `H=h+2cchi`, `V=Lambda+2cchi` and
`X=cchi-C V`. The scalar block becomes four independent wave pairs with
speed squares `f,q,1,1`. The finite identity
`C(q-1)=2(mu-1)/3` cancels the scalar light-speed coincidence without dividing
by `q-1`; `epsilon_alpha=1` removes the lapse forcing at `q=f`. The two vector
blocks have speed squares `1,mu`; the two tensor blocks have speed square `1`.
The explicit transform and inverse are preserved in the capsule's corrected
inner assessment. The scalar storage is `pi=P/Omega`, with
`P=K_phys-2Theta_phys`, not `K_phys/Omega`.

The original displayed `A_t` row accidentally contained `2 beta/3`; its
correct term is `2 Lambda/3`. The erroneous note, one-line erratum, corrected
note and independent review are all retained. Only the corrected note is
authoritative.

Starting from the complete physical-reference wave-map gauge, let
`cW=1-W`, `A0=alpha^2 chi`, `B=cW G0+W A0` and `kappa=B/(A0+B)`. The proposed
reference-deviation additions are

```
Delta alpha_t = -2 cW alpha (P-P_hat)/Omega,
Delta beta_t^i = (B-A0)(Lambda^i-Lambda_hat^i)
  + alpha^2 (2 kappa^2-1/2) gtildeInv^{ij}
      (chi_j-chi chi_hat_j/chi_hat)
  - cW eta_I (beta^i-beta_hat^i).
```

The resulting normalized coefficients are `f=1+2cW/alpha` and `mu=B/A0`.
An implementation can evaluate the bounded `kappa` without forming `mu`.
At `W=1`, every addition vanishes; the branch must retain the same real outer
wave-map equations. The subsequent arithmetic tests below distinguish their
legacy and regrouped floating evaluations. At the Minkowski reference every deviation vanishes,
including the nonzero reference connection in the geometric transition.
That algebraic statement still requires nonlinear implementation and arithmetic
tests on the nonflat reference.

The subsequent private compiled principal gate passed 118 actual-kernel cases
in Release and Address/UndefinedBehavior-sanitized Debug, plus 18 exact rational
scalar cases. Printed matrices were byte-identical between builds. The largest
matrix error was `5.346834086594754e-13`, the largest explicit basis-inverse
error `4.440892098500626e-16`, and the largest sampled infinity-norm basis
condition `113.32241771251452`. The sample includes oblique propagation,
positive-definite nontrivial metrics and `mu=1`, `q=1`, `f=1`, `q=f`
coincidences. An independent saved-matrix readback reproduced the literal
20-field coefficients and direct scalar wave relations without rerunning the
kernel or original analyzer. Its largest direct wave-identity residual was
`6.659118e-13`.

The exact compiled source, commands, empty error logs, summaries, saved matrices,
independent readback and count/optimization guard failure lineage are in
[a separate compiled-gate capsule](validation/hyperboloidal-inner-joint-principal-compiled-20261009/README.md).
Its catalog SHA256 is
`b481b0813cfb4850358ed510bdc78344f1af93e504b613517f3a00997ac8d54a`.
It contains 254 files, 7,679,469 bytes and 64 finite JSON files. Three compiled
payloads are metadata only, as are 1,413 external compiler/header/runtime
dependencies. This capsule contains the completed finite principal gate;
the earlier derivative/pencil capsule remains unchanged.

These results establish neither a uniform diagonalizer at a puncture nor
nonlinear regularity or evolution stability. The finite nonlinear arithmetic
checks below address selected nonflat/high-contrast states; variable-coefficient
stability and actual puncture treatment remain separate gates. In a putative radial trumpet,
the constant connection response additionally needs `Lambda=O(r)` or derived
cancellation of stronger connection residues. The unchanged outer wave-map
condition also retains its conditional stationary mass-log obstruction.

## Nonlinear arithmetic failures and the new outer identity

Three private implementations retain their actual failed results. Source001
failed compilation because one `auto` declaration mixed different array types.
Source002 compiled and completed the fixed 15,740-record registry but failed
3,236 independent source comparisons. Source003 changed only three conditioned
field differences and their branch counters. It removed 1,364 prior failures;
its remaining 1,872 failures all match old failed comparisons, with no new
failure in that registry. Every remaining failure is at stored `W==1`, and
all eight returned split parts are bitwise the legacy outer wave-map helper.
The fixed test has no failed `W<1` comparison. Source003 overall remains FAIL.

For positive live/reference lapse `a,h` and chi `x,y`, source003 uses the
deviation expression for `a^2 x-h^2 y` only when both field ratios lie in the
closed interval `[1/2,2]`; otherwise it forms the complete scaled products
before subtracting. Each log-gradient difference uses its own field's near
predicate. The predicate uses binary exponents and mantissas without forming
a tiny ratio. Scaled products retain every registered first field-dual term,
including a zero primal factor with a nonzero tangent. This does not certify
arbitrary cancelling sums or arbitrary metric contrast.

The remaining failed outer rows cannot be repaired while preserving their old
floating results. A separately named outer001 implementation therefore keeps
the same real physical-P wave-map equations and explicitly changes the far
arithmetic contract. The legacy header differs only by its function name;
reference connection, scaled source and single pole assembly are unchanged.
Joint-near lapse/chi states call that legacy body directly. Far states use
complete products, with `G` and `Gh` the live/reference inverse conformal metric:

```
dV = a^2 x G-h^2 y Gh,
Lhat = h^2 y Gh-betaHat betaHat,
dL = dV-(beta-betaHat) beta-betaHat (beta-betaHat).

Ralpha = beta.grad(a)-(a/h) betaHat.grad(h),
Salpha = -a^2 P+a h Phat-a (beta-betaHat).grad(Omega)-a dL:C0,
Rbeta_i = a^2 x Lambda_i-h^2 y LambdaHat_i
  + beta_j (beta_ji-betaHat_ji)+(beta-betaHat)_j betaHat_ji
  + .5 a^2 Gij x_j-.5 h^2 Ghij y_j
  - a x Gij a_j+h y Ghij h_j,
Sbeta_i = 2 dVij Omega_j-dLjk C^(i+1)_jk
  - dLjk beta_i C0jk-Lhat_jk (beta-betaHat)_i C0jk.
```

Repeated spatial indices are summed, `C=Omega Gamma_reference` includes the
full nonflat physical reference connection, and each row is assembled as
`R+S/Omega` once. Stored `P=K_phys-2Theta_phys` remains independent of Theta.
No black-hole RHS subtraction, live-lapse division, floor or clipping is added.
The inner coefficient/Gauge suffix remains source003 byte-for-byte. This is
a new arithmetic implementation, not a retroactive source003 PASS.

Outer001 passed the unchanged 15,740-record gate in Release and sanitized
Debug. The independent saved-output audit retained each build's source,
commands, executable identity, oracle and failed-history pins.

| Check | Saved result in each build |
| --- | --- |
| Maximum split-part / assembled-row scaled error | `2.0213034860e-14` / `4.4021148870e-14`, gate `2e-10` |
| Actual22 gauge-dual maximum | `8.8482325815e-15`, gate `2e-10` |
| 2,520 three-level finite-difference sequences | Final maximum `1.4822275264e-8`, gate `5e-7` plus convergence/floor check |
| Exact reference split parts | All 336 rows zero |
| `W==1` joint-near rows | All 5,400 retain every legacy value/dual bit |
| `W==1` far rows | 576 rows pass the unchanged MP targets; 574 intentionally differ from old primal bits |
| Inner coefficient bypass at `W==1` | All 5,976 rows make zero coefficient calls |
| Actual22 geometric rows | All 90,720 saved bit comparisons unchanged |

Root elapsed times were 87.4783 s for Release and 97.4468 s for Debug with
address and undefined-behavior sanitizers. The compiler builds differ in
72 principal and 60 source split-part rows, within the same gates; full
cross-build bit identity is not claimed. All far rows in this main registry
have zero dual seeds. The separately fixed complete far-dual supplement below
failed, so the ordinary dual result does not validate that branch.

A separate source003 field-difference supplement passed 129 records and 684
component checks in both Release and sanitized Debug (3.3535 s / 3.1254 s).
It tests three exact-power normal witnesses, six relative field seeds including
zero-primal/nonzero-gradient tangents, and 108 cases at and around both near
thresholds over three reference scales. Three preserved old-near expressions
produce their expected lost-normal zero controls. This accepts those named
arithmetic units only; it leaves source003 FAIL and the original ineligible
supplement unaltered. Floating branch continuity and universal accuracy are
not established.

Sources, failures, independent reviews, completed gates and saved readbacks are
in [the arithmetic capsule](validation/hyperboloidal-inner-outer-arithmetic-20261009/README.md).
Scientific JSONL/array payloads, executables and files larger than 1 MiB are
represented by metadata and streamed hashes. The first saved-output reader's
signed-zero parsing failure is retained alongside its corrected fresh reader.

Production `src/` and root `CMakeLists.txt` remain byte-identical to
`27c19d20696ea6dd4704032c51dfd026218f64f2`; no candidate is adopted by this
checkpoint. Existing CPU Serial/double, vacuum, uniform single-MeshBlock
restrictions remain in force.

## Complete far-field derivative gate: actual Release failure

The direct outer-helper supplement completed its fixed 2,373 records and
4,900 helper calls after independent source review. It retains the same 112
high-contrast state bases, 17 complete field seeds, 448 additional rows with
zero primal lapse/chi gradients, 18 exact closed controls and three reused
legacy negative controls. Sixteen relative-alpha representatives include
five-level finite differences. Reference fields and coefficients have zero
tangent. Physical P and Theta are independently stored and seeded; the direct
gauge consumes P and has no separate Theta dependence. This tests the direct
outer helper, rather than the compound inner/full22 equations.

Compilation succeeded. The independent literal physical-P dual oracle uses
480/560 decimal digits, a cofactor inverse, the complete nonflat reference
connection, and unchanged `2e-10` output gates. Its precision check passed
`1e-220`; the exact closed Fraction controls have zero error. Primal split-part
and assembled-row errors are at most `4.9853135706e-15` and
`1.7917888212e-14`. The fixed finite-difference checks passed.

Release nevertheless failed 63 tangent comparisons: 47 split parts and
16 assembled rows. All failures use the metric-STF seed, across 26 input rows;
55 belong to the chi-gradient-contrast family and eight to the
large-alpha/small-chi family. Eight saved targets are exactly zero. Failures
occur in the Cauchy core, transition and outer collar, including `W==1`.
The first is an axial core regular-shift tangent returned as zero with saved
target `1.5510168630129906e29`. These results preserve the entire gate as FAIL.

Root elapsed was `24.50961625 s`; source and dependency pins stayed unchanged.
Conditional sanitized Debug and native adoption were not started. Source001's
earlier metadata self-capture failure is retained beside the fresh source002
scientific attempt, whose probe/oracle/runner bytes are unchanged.

The saved failure association retains all 26 complete consumed contexts.
Separate rounding of cancelling gradient terms and errors in the ordinary
inverse-metric directional derivative are two mechanisms to investigate.
Existing exports omit returned inverse entries and individual contraction
summands, so they do not determine either mechanism's contribution. A new
diagnostic must separate them before a correction is accepted. No seed change,
tolerance relaxation or successful flux repair is implied.

The [far-dual failure capsule](validation/hyperboloidal-far-dual-failure-20261009/README.md)
preserves exact source, commands, logs, oracle, independent reviews and saved
associations. Its scientific JSONL/query stdout and executable are metadata
only. This additional gate leaves production unchanged and establishes no
native evolution or black hole stability. The eventual black hole test must
survive the inner wormhole-to-trumpet transition while keeping the Minkowski
hyperboloidal reference throughout.
