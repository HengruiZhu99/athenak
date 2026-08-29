# Fully subtracted Ref-GH gauge repair

Date: 2026-08-29 (America/New_York)

Branch: `codex/ref-gh-fully-subtracted-gauge-repair-20260829`

Frozen parent: `223947486ac4498bab2e197feca56462c77e6d76`

## Claim boundary

This investigation begins from the completed
`GAUGE-DRIVER COUPLING DEFECT ISOLATED` discriminator.  It does not reinterpret
that result as a proof that cancellation is the sole exponential mechanism.
At this checkpoint the residual-variable algebra has been derived and
symbolically verified, the residual target/driver/source oracles pass at their
unchanged conditioned tolerances, the compact analytic residual Einstein
source has passed its current oracle gate, and the stationary trumpet
coefficient asymptotics have been derived and independently measured.  The
initial red perturbed-source result is retained below because it exposed an
ill-conditioned full-driver input rather than being hidden.  No production
equation has changed, no repaired fixed-point ladder has run, and no repaired
evolution, robustness, or performance claim is made.

The frozen A--E histories, stationary residual ladder, principal-symbol audit,
and compact Aurora evidence remain unchanged under
`artifacts/ref_gh_ordering_gauge_discriminator_20260829/`.  Both committed
`verified_compact_sha256.txt` manifests passed before this branch was created.

## Phase 0: frozen baseline

Local HEAD, upstream, and the remote discriminator branch were all exactly
`223947486ac4498bab2e197feca56462c77e6d76` before creating this branch in a
new worktree.  The pre-existing checkout at
`/home/hzhu/Desktop/research/gr/athenak` was dirty and was not modified.

The production gauge files at branch creation have the following SHA-256
identities:

| File | SHA-256 |
|---|---|
| `src/ref_gh/ref_gh_calcrhs.cpp` | `a08cf37d2aa2f202c6dadce34c5c97a8f6fc4347ccf32383f75fef09289734b7` |
| `src/ref_gh/gauge_driver.hpp` | `11f4dc3d11a4c72b27be62aba0c8c7dbc99927746b35302c72b8dd85e6ddf1a0` |
| `src/ref_gh/physical_gauge_target.hpp` | `09102ab48832133747ba1f466f557b64bb34da55344f3eee5f4fe4743813d532` |
| `src/ref_gh/standard_gh_source.hpp` | `e5798d984b2771950b5bc6c5997f7a290c7d246cffb129340734a9179714963a` |
| `src/ref_gh/reference_gauge_baseline.hpp` | `c89425337d76efc4067805010e4843318f758eff48dfb44c66b488af6debb5df` |

A diff against the frozen parent over these files returned zero.  A fresh
Release/Serial build passed the complete source-unit executable without
changed tolerances.  That includes the 216-point coefficient oracle,
2160-point expanded radial oracle, 2376-point geometry oracle, 2160-point
moving gauge/`dtTheta` oracle, and all 4320 compatible/standard all-61 RHS
comparisons.  The independent Python standard-GH, binary64 stationary-source,
and reference-frame audits also passed when invoked directly.  Deterministic
SymPy regeneration is recorded in the phase-0 artifact directory.

Compact local evidence is in
`artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase0_local/`.

## Exact residual-variable derivation

All gauge quantities in this section are reference-frame covector components.
Write

\[
 H_A=\bar H_A+h_A,\qquad
 \theta_A=\bar\theta_A+\vartheta_A,\qquad
 F_A=\bar F_A+f_A,
\]

where \(h_A=\delta H_A\), \(\vartheta_A=\delta\theta_A\), and
\(f_A=\delta F_A\).  Also write

\[
 \beta^i=\bar\beta^i+b^i,
 \qquad
 K^{\rm ref}_{iA}=\partial_i\bar H_A-\Omega_{Ai}{}^B\bar H_B.
\]

The production full-variable equations are

\[
 \partial_t H_A=
 \beta^i\partial_iH_A-\mu(H_A-F_A)+\theta_A
 +(\Omega_{At}{}^B-\beta^i\Omega_{Ai}{}^B)H_B,
\]

\[
 \partial_t\theta_A=
 -\eta\theta_A-\eta\beta^i\partial_iH_A
 +\Omega_{At}{}^B\theta_B
 +\eta\beta^i\Omega_{Ai}{}^BH_B.
\]

Direct substitution, collection of regular fields, and subtraction of the
analytic reference time derivatives gives

\[
\begin{split}
 \partial_t h_A={}&
 \beta^i\partial_i h_A-\mu(h_A-f_A)+\vartheta_A
 +(\Omega_{At}{}^B-\beta^i\Omega_{Ai}{}^B)h_B\\
 &+b^iK^{\rm ref}_{iA}+S^H_A,
\end{split}
\]

with the complete time-dependent reference forcing

\[
 S^H_A=
 \bar\beta^iK^{\rm ref}_{iA}
 -\mu(\bar H_A-\bar F_A)+\bar\theta_A
 +\Omega_{At}{}^B\bar H_B-\partial_t\bar H_A,
\]

and

\[
\begin{split}
 \partial_t\vartheta_A={}&
 -\eta\vartheta_A-\eta\beta^i\partial_i h_A
 +\Omega_{At}{}^B\vartheta_B
 +\eta\beta^i\Omega_{Ai}{}^Bh_B\\
 &-\eta b^iK^{\rm ref}_{iA}+S^\theta_A,
\end{split}
\]

where

\[
 S^\theta_A=
 -\eta\bar\theta_A+\Omega_{At}{}^B\bar\theta_B
 -\eta\bar\beta^iK^{\rm ref}_{iA}
 -\partial_t\bar\theta_A.
\]

These two forcing terms are not optional.  They vanish if the chosen
reference gauge pair itself satisfies the full driver with
\(\bar\beta^i,\bar F_A\), but that is a condition to verify, not an assumption
licensed by a stationary test.  In particular, a moving reference may have
nonzero \(\partial_t\bar H_A\) and \(\partial_t\bar\theta_A\).

For the static matched \(q=1\) trumpet,

\[
 \partial_t\bar H_A=\partial_t\bar\theta_A=0,
 \quad \bar F_A=\bar H_A,\quad \Omega_{At}{}^B=0,
 \quad \bar\theta_A=-\bar\beta^iK^{\rm ref}_{iA}.
\]

Consequently \(S^H_A=S^\theta_A=0\), confirming both signs in the simplified
equations from the controlling specification.  Setting
\(h=\vartheta=f=b=0\) then makes the driver residual exactly zero in exact
arithmetic.

The script
`scripts/ref_gh/verify_fully_subtracted_gauge_algebra.py` expands the original
and residual equations component by component with symbolic frame-motion
matrices.  It proves the general time-dependent identities and the static
matched reduction by exact SymPy simplification.

## Direct residual Einstein gauge source

Let \(E^A{}_a\) denote the reference coframe used to convert the stored frame
covector into coordinate components.  Define

\[
 B_a(g;\bar g)=-g_{ab}g^{cd}\bar\Gamma^b{}_{cd},\qquad
 \Delta B_a=B_a(g;\bar g)-B_a(\bar g;\bar g).
\]

Because \(B_a(\bar g;\bar g)=\bar H_a=E^A{}_a\bar H_A\), the gauge increment
covector is identically

\[
 J_a=\widehat H_a-B_a(g;\bar g)
    =E^A{}_a h_A-\Delta B_a.
\]

This is the production identity required to avoid constructing any of the
three singular full quantities separately.  Its coordinate derivative is

\[
 \partial_\mu J_a=
 (\partial_\mu E^A{}_a)h_A+E^A{}_a\partial_\mu h_A
 -\partial_\mu\Delta B_a.
\]

For \(\mu=t\), \(\partial_t h_A\) is the same-stage residual driver RHS above.
No reconstruction or cancellation with \(\partial_t\bar H_A\) appears.
Substitution into

\[
 -\nabla_aJ_b-\nabla_bJ_a
 +\gamma_0\left(2\delta^c{}_{(a}n_{b)}-g_{ab}n^c\right)J_c
\]

is algebraically identical to the present
`AddOrdinaryGaugePartialWaveSource` expression.  The symbolic oracle verifies
the product rule, \(J_a\), \(\partial_\mu J_a\), and every component of this
symmetric increment.

## Time-dependent reference policy

The production repair must evaluate \(S^H_A\) and \(S^\theta_A\) analytically
for a moving reference.  It must not infer their vanishing from the static
case.  Conversely, if a future reference baseline is deliberately defined to
obey the driver, the implementation may use that proved identity to remove the
forcing, provided the moving mixed-jet/`dtTheta` oracle remains an independent
gate.

## Phase 2 checkpoint: residual physical target scaffolding

`src/ref_gh/reference_residual.hpp` now carries a reference value, the
independently reconstructed physical value, and an algebraically evaluated
regular difference.  Its product, quotient, square-root, and cube-root rules
implement exact finite identities rather than `physical-reference`.  The
coordinate metric residual is constructed directly from

\[
 \Psi_{AB}-\eta_{AB},\qquad \Pi_{AB},\qquad \Phi_{IAB},
\]

and the reference frame.  In particular,

\[
 \delta g_{ab}=E^A{}_aE^B{}_b(\Psi_{AB}-\eta_{AB})
\]

and the product-rule expression for \(\delta(\partial_\mu g_{ab})\) contain no
subtraction of coordinate trumpet fields.  Inverse differences use

\[
 \delta g^{-1}=-g^{-1}\delta g\,\bar g^{-1}.
\]

`ComputePhysicalGaugeTargetResidual` evaluates the unchanged advective
1+log/conformal-Gamma target in this residual algebra.  It does not yet
participate in `CalcRHS`.

The source-unit gate covers both generic and compact analytic references for
the full 2160-point \((q,\dot q,\ddot q,r,\Omega)\) matrix, or 4320 backend
samples.  The matched reference returns bitwise-zero target, conformal-Gamma,
and shift residuals at every sample.  The established full physical target is
still used for non-residual outputs and remains unchanged.  Direct comparison
against `F-Fref` is gated only at \(r\ge0.8M\), where that subtraction remains
conditioned, and passes at `3.82012e-14` under the unchanged
`1024 epsilon_binary64` threshold.

Across all radii, the raw independently evaluated subtraction differs from the
residual path by as much as `1.3918e-06`.  This value is retained as a
diagnostic, not accepted by a weakened tolerance.  Near the puncture the old
subtraction is the quantity already demonstrated to be ill-conditioned, so it
cannot serve as a binary64 truth oracle for the repaired residual.  A
high-precision or independently generated residual oracle is still required
before production dispatch.  Consequently Phase 2 is a tested scaffolding
checkpoint, not yet complete qualification.

## Phase 3 checkpoint: residual driver and Einstein-source scaffolding

`ComputeGaugeDriverResidualRhs` implements the complete general residual
driver derived above, including frame motion, reference forcing, and
`delta(beta) Kref`. A source-unit identity test with independent synthetic
full/reference/residual data agrees with the full driver at `1.38778e-16`.
For the proved static matched `q=1` identities, the exact residual driver
returns bitwise zero.

`AddOrdinaryGaugeResidualPartialWaveSource` evaluates `J`, `dJ`, and the
ordinary-GH Einstein gauge increment with `ReferenceResidualValue` arithmetic.
It is explicitly oracle-only for `AnalyticRadialQPoint`: it is not dispatched
from `CalcRHS` and must not become the production implementation because its
recursive reference-Christoffel access would recreate the generic tensor cost.
The exact matched `q=1` source is bitwise zero across the expanded radial
sample set.

The new perturbed-state comparison is red under the unchanged
`1024 epsilon_binary64` tolerance. The maximum conditioned error is
`0.103832` at encoded location `1322174`, which decodes to analytic backend,
sample 1322 (`q=1`, `q_dot=q_ddot=0`, `r=0.8M`, first angular direction), and
an off-diagonal Einstein gauge-source category. The complete compact log is
under `artifacts/ref_gh_fully_subtracted_gauge_repair_20260829/phase3_local/`.

This mismatch is not yet assigned to the residual implementation: the legacy
comparison reconstructs singular full driver/source quantities in binary64,
which is the cancellation mechanism this repair is intended to avoid. It is
therefore evidence of an unresolved oracle disagreement, not evidence that
either side is correct. The next hard gate is an independent high-precision
or generated perturbed residual driver/source oracle. Production dispatch
remains disabled until that gate passes.

### Phase 3b diagnosis of the red source gate

The red phase-3 comparison was not an isolated Einstein-source test. It gave
the residual source the direct residual driver time derivative, while the
legacy source received an independently reconstructed full driver derivative.
That raw full derivative contains the singular binary64 cancellation being
removed. The source mismatch therefore inherited a driver-input mismatch.

When both source paths receive the same-stage regular derivative

\[
 \partial_t H_A=\partial_t\bar H_A+\partial_t\delta H_A,
\]

the complete source-unit gate passes without changing the
`1024 epsilon_binary64` tolerance. The all-radius raw source discrepancy is
`6.96333e-09`; the combined conditioned gate at radii at least `0.8M` is
`3.82012e-14`. The independently reconstructed full driver remains a separate
diagnostic and differs by as much as `1.05256`; it is not used as truth for the
subtracted trumpet driver.

The arbitrary-precision implicit-trumpet oracle now evaluates the exact
advective 1+log/conformal-Gamma target independently of the generated
binary64 table. At 80 decimal digits over

\[
 r/M=0.03,0.05,0.08,0.125,0.2,0.4,0.8,1.5,3,5,
\]

it finds

\[
 \max |F_A^{\rm ref}-H_A^{\rm ref}|=7.60\times10^{-75},\qquad
 \max |\widetilde\Gamma^i_{\rm ref}|=4.77\times10^{-80}.
\]

This independently proves the static identities used to remove the pure
reference forcing in the exact matched `q=1` branch. It does not yet qualify
the all-radius perturbed driver, moving-reference forcing, or production
dispatch. Compact evidence is in `phase3b_local`.

### Phase 3c compact analytic residual source

The compact radial-q implementation now evaluates the Einstein gauge increment
without the recursive generic reference accessors. The generated upper-index
gauge contraction is linear in \(g^{ab}\) and \(\partial_\mu g^{ab}\). Feeding
it the exact residuals of those quantities gives \(\Delta B^a\) and
\(\partial_\mu\Delta B^a\) directly. Lowering uses

\[
 \Delta B_a=\delta g_{ab}B^b(g)+\bar g_{ab}\Delta B^b,
\]

and its four-term differentiated product identity. The resulting \(J_a\),
\(\partial_\mu J_a\), covariant derivative, damping projector, and frame
projection are the same expressions as in the generic residual oracle.

The matched `q=1` compact source is bitwise zero at all ten expanded radii.
Perturbed compact/generic residual sources pass at the unchanged conditioned
radii. Their all-radius maximum difference is `3.05105e-12` at `r=0.03M`;
this is retained as a coefficient-conditioning diagnostic for Phase 4.
The compact function remains outside `CalcRHS` at this checkpoint.

## Phase 4: genuine stationary-trumpet coefficient asymptotics

The arbitrary-precision audit is implemented in
`scripts/ref_gh/analyze_fully_subtracted_trumpet_asymptotics.py`.  It starts
from the implicit (n=2) trumpet rather than the generated binary64 radial
table.  It constructs the reference coordinate two-jet, differentiates the
target with respect to the stored frame variables, evaluates the direct
residual (\Delta B_a) identities, substitutes the same-stage residual driver
into (\partial_tJ_a), and measures the maximum coefficient over every input
and output component in each named family.

### Leading trumpet fields

Let (R_0) be the limiting areal radius and let
(R_\alpha=(dR/d\alpha)_{\alpha=0}).  From

\[
 \frac{d\log r}{d\alpha}=\frac{R_\alpha}{\alpha R}
\]

one obtains

\[
 \alpha=a_0r^p[1+O(r^p)],\qquad
 p=\frac{R_0}{R_\alpha}=1.09129710479541717714\ldots .
\]

The remaining stationary fields obey

\[
 R=R_0+O(r^p),\qquad L=\frac{R}{r}=O(r^{-1}),
 \qquad B=B_0+O(r^p),\qquad \beta^i=Bx^i=O(r).
\]

Consequently the largest coframe entries are (O(r^{-1})), while
(e_0{}^t=O(r^{-p})), (e_0{}^i=O(r^{1-p})), and
(e_I{}^i=O(r)).  Radial differentiation lowers a power by one.  Applying
these rules to the exact frame formulas, rather than to independently
subtracted full fields, gives the following worst-component envelopes.

| Coefficient family | Derived power | Fitted power |
|---|---:|---:|
| (\bar H_A) | \(-2p\) | -2.182658 |
| (\mathcal K^{\rm ref}_{iA}) | \(-(3p+1)\) | -4.273998 |
| (\Omega_{Ai}{}^B) | \(-(p+1)\) | -2.091304 |
| (\partial_j\Omega_{Ai}{}^B) | \(-(p+2)\) | -3.091301 |
| (\bar\beta^i\Omega_{Ai}{}^B) | \(-p\) | -1.091373 |
| (\partial\delta\beta/\partial\Psi) | (p+1) | 2.091271 |
| (\delta\beta^i\mathcal K^{\rm ref}_{iA}/\delta\Psi) | \(-2p\) | -2.182727 |

The singular (\mathcal K^{\rm ref}) is therefore not by itself the
coefficient seen by a stored metric residual: the lapse/frame factors in
(\delta\beta) cancel (2p+2) powers.  The resulting (r^{-2p}) map is
still genuinely divergent.  It is not a pure-reference cancellation and is
not removed by this repair.

### Target and Einstein-source coefficient maps

The high-precision script forms directional Jacobians with respect to every
symmetric stored (\Psi_{AB}), (\Pi_{AB}), (\Phi_{IAB}), all four gauge
residuals, and all three (\Upsilon^i).  The direct (\Delta B_a) source and
the complete same-stage source have the following maximum powers:

| Map | Derived power | Fitted power |
|---|---:|---:|
| (\delta F/\delta\Psi) | \(-2p\) | -2.182607 |
| (\delta F/\delta\Pi) | \(-p\) | -1.091292 |
| (\delta F/\delta\Phi) | \(-(2p+2)\) | -4.182543 |
| (\delta F/\delta\Upsilon) | \(-(2p+1)\) | -3.182556 |
| direct (\Delta B:\Psi\mapsto S^{\rm gauge}) | \(-3p\) | -3.273938 |
| direct (\Delta B:\Pi\mapsto S^{\rm gauge}) | \(-2p\) | -2.182658 |
| direct (\Delta B:\Phi\mapsto S^{\rm gauge}) | \(-3p\) | -3.273962 |
| complete (\Psi\mapsto S^{\rm gauge}) | \(-3p\) | -3.273923 |
| complete (\Pi\mapsto S^{\rm gauge}) | \(-2p\) | -2.182577 |
| complete (\Phi\mapsto S^{\rm gauge}) | \(-(3p+2)\) | -5.273827 |
| complete (\Upsilon\mapsto S^{\rm gauge}) | \(-(3p+1)\) | -4.273841 |
| direct (h\mapsto S^{\rm gauge}), before (\partial_t h) substitution | (0) | -0.000017 |
| (\partial_t h\mapsto S^{\rm gauge}) | \(-p\) | -1.091285 |
| (\partial_i h\mapsto S^{\rm gauge}) | (1-p) | -0.091318 |
| complete same-stage (h\mapsto S^{\rm gauge}) | \(-2p\) | -2.182658 |

The apparently benign direct (h\) map is an important cancellation check:
its individual coordinate pieces contain inverse powers, but its projected
covariant combination is bounded.  Substituting the actual same-stage driver
reintroduces a genuine (r^{-2p}) lower-order map through
(\bar\beta^i\Omega_{Ai}{}^B).  The more singular complete (\Phi) and
(\Upsilon) maps arise through the unchanged physical gauge target followed
by the (\partial_t h\) contribution to (\partial_tJ_a); they are not
reference-minus-reference subtraction artifacts.

At 90 decimal digits, all 26 fitted powers agree with the analytic predictions
to at most (1.33\times10^{-4}), against a (5\times10^{-3}) gate.  Independent
identities for (F^{\rm ref}-H^{\rm ref}), reference conformal Gamma,
(\mathcal K^{\rm ref}), and the recovered shift pass a (10^{-45}) threshold;
the largest error is (2.38\times10^{-65}).  Repeating every target Jacobian
with centered-difference steps (10^{-30}) and (10^{-24}) produces identical
40-digit tables and powers.  Compact evidence is in `phase4_local`.

### Energy-estimate consequence

The standard principal symbol and its local symmetrizer are unchanged for
every (r>0).  This establishes local strong/symmetric hyperbolicity on any
punctured domain (r\ge r_{\min}>0), but it does not provide a resolution-
uniform puncture estimate.  Bounding the lower-order operator in the ordinary
stored-variable norm gives a Gronwall constant whose worst measured envelope
is at least (O(r_{\min}^{-(3p+2)})).  It therefore diverges as the first
included point moves inward.

A uniformly equivalent bounded change of norm has not been constructed here.
Simple frame weights can regularize individual triangular couplings—for
example an (\alpha) weight on the most singular time-frame gauge component—
but the coupled (\Psi,\Pi,\Phi,\Upsilon,h) maps require mutually different
powers, while the standard GH principal symmetrizer couples (\Pi) and
\(\Phi).  Any claimed weighted estimate must demonstrate positivity,
uniform equivalence (or explicitly accept a degenerate puncture norm), and
control of the weight-derivative terms.  The present evidence therefore rules
out only the naive uniform unweighted estimate; it neither proves nor assumes
that no mathematically natural degenerate weighted estimate exists.

This nonuniformity is a scientific limitation, not authorization to delete a
continuum term or tune the gauge.  Exact matched-state initialization and the
fixed-point ladder remain the next valid discriminators because every
divergent map multiplies an exact residual that should initially vanish.

## Phase 5: exact matched-state initialization

`IsExactMatchedQ1StaticReference` is a shared strict predicate.  It returns
true only when the reference is q-controlled, both q-control modes are off,
the reference is not time dependent, and the binary64 state is exactly

\[
 q=1,\qquad \dot q=0,\qquad \ddot q=0.
\]

There is deliberately no tolerance and no radial condition.  The source-unit
test verifies that changing any Boolean condition, changing (q) by one ULP,
or supplying the smallest nonzero binary64 value for either time derivative
disables the predicate.

For this exact reference, the stationary physical metric is the reference
metric itself.  Initial and physical-boundary data are therefore filled as

\[
 \Psi_{AB}=\eta_{AB},\qquad \Pi_{AB}=0,\qquad \Phi_{IAB}=0
\]

without a physical/reference projection.  When gauge-reference subtraction
is enabled, the independently proved identities from Phase 3b similarly give

\[
 \delta\widehat H_A=0,\qquad \delta\theta_A=0,\qquad \Upsilon^i=0
\]

as exact binary64 fills.  Prescribed q, feedback q, a moving reference, a
nonunit q, or unsubtracted full-gauge storage continues through the existing
general projection path.

A focused (16^3) initialized-mesh check on the analytic backend reports
`field Linf=0`, `stored_Hhat_A Linf=0`, and `stored_theta_A Linf=0`.  Physical
metric/lapse/shift reconstruction errors are at most (3.34\times10^{-16}).
The source-unit suite remains green.  The initial production RHS is still
(5.80\times10^{-14}): this checkpoint has not yet dispatched the residual
driver or Einstein gauge source, so that value is retained as the red/legacy
production baseline for Phase 6 rather than attributed to initialization.
Compact evidence is in `phase5_local`.

## Principal part

The rewrite is an algebraic change of dependent variables plus lower-order
source collection.  It does not change the derivative coefficients of the
STANDARD first-order GH plus driver system.  Therefore the already-audited
standard principal symbol and characteristic fields are preserved.  The
separate compatible-ordering loss of strong hyperbolicity is also unchanged;
compatible ordering remains an oracle/research option and is not the
production candidate.  Phase 4 shows that the unchanged local principal
symbol does not by itself provide a resolution-uniform bound on the singular
lower-order coefficient maps in the ordinary stored-variable norm.

## Phase 6 local production residual dispatch

Commit `ab30fa963f5d1d7ce54748ffb287c91c87705153` dispatches the
cancellation-free equations from production `CalcRHS` only under the strict
static, uncontrolled, unprescribed `q=1` predicate.  The driver consumes the
stored residuals and spatial derivatives directly.  Its genuine reference
coefficient is generated as

\[
 {\cal K}^{\rm ref}_{iA}
 =e_A{}^a\partial_iH^{\rm ref}_a,
\]

rather than reconstructed as the cancellation-prone
\(\partial_iH_A^{\rm ref}-\Omega_{Ai}{}^BH_B^{\rm ref}\).  The same-stage
residual Hhat RHS supplies \(\partial_t\delta H_A\) to the direct residual
Einstein gauge increment.  Generic, moving, prescribed, and feedback
production modes retain the legacy dispatch until separately qualified.

The deterministic generator produced byte-identical headers in two fresh
passes.  The direct Kref oracle passes at `1.15471e-15` under the unchanged
`256*epsilon` conditioned gate.  The all-61 oracle now compares the legacy
generic full-reference evaluator to the fully subtracted compact evaluator for
all 4320 q/rate/acceleration/radius/Phi-ordering samples.  It includes the
moving-reference `dtTheta` forcing, gauge target, gauge driver, ordinary-GH
Einstein increment, gamma0, gamma2, and both compatible and STANDARD Phi
ordering, and passes at `4.13003e-14` under the unchanged `256*epsilon` gate.

Removing a tautological overwrite in the residual-target oracle exposes the
all-radius physical-target association discrepancy `6.02413e-10`.  The raw
target-delta and full-driver diagnostics remain `1.3918e-06` and `1.05256`.
These are near-puncture diagnostics because neither binary64 full construction
is an independent truth there.  Exact matched target, driver, and Einstein
gauge-source residuals remain bitwise zero at every sampled radius; perturbed
comparisons retain the unchanged tolerance in the established conditioned
region `r>=0.8M`.

A focused 16^3 STANDARD production initialization has exact-zero Hhat, theta,
Upsilon, ordinary-gauge Pi increment, and KO gauge sectors.  The remaining
total Pi RHS is `5.681872526233013e-14`, entirely in the existing covariant
vacuum source at its maximum.  This is a local Kokkos Serial algebra/fixed-point
checkpoint.  It does not qualify an Aurora device path, the 64/96/128 residual
ladder, or any evolution.

### First Aurora attempt and staged all-61 portability correction

Aurora debug job `8791211` configured the exact source checkpoint
`ef130480bfdbecc5b7d8f21169085be5da6c8cc4` with IntelLLVM 2025.3.2,
Kokkos 4.7.2 `SERIAL;SYCL`, and `Kokkos_ARCH_INTEL_PVC=ON`.  Compilation
failed in the monolithic `source_unit.cpp` device image before rank mapping or
kernel execution.  IGC reported an internal segmentation violation and `icpx`
exit 245 while lowering the all-61 oracle.  PBS recorded exit 2 after
00:10:41.  This is a preserved compiler failure, not a numerical result.

The equation-preserving portability correction stages the legacy-generic and
fully-subtracted compact RHS evaluations in two independent device kernels,
stores their 61-component outputs, and applies the original conditioned
comparison in a third reduction.  The parameter matrix, 4320 samples, both
Phi orderings, generic oracle, scale definition, and `256*epsilon` tolerance
are unchanged.  A fresh local Kokkos Serial build and source-unit run retain
the exact `4.13003e-14` all-61 result.  Aurora device qualification remains
open until the focused rerun compiles and executes these staged kernels.

## Remaining gates

The following are not yet complete: independent all-radius high-precision
qualification of cancellation-free \(\delta F_A\), independent qualification
of the direct \(\Delta B_a\) and derivative evaluator below the conditioned
binary64 region, Aurora/PVC device equivalence, the repaired 64/96/128
fixed-point ladder, the 3M/5M discriminator, the resolution ladder, 20M, and
conditional 100M qualification.  Exact matched-state fill and strict static
q=1 residual production dispatch are locally tested.  General moving-reference
production dispatch remains legacy pending a separate qualification.
Performance optimization remains out of scope for this campaign.
