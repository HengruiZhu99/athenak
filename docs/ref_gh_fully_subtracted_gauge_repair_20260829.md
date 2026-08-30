# Fully subtracted Ref-GH gauge repair

Date: 2026-08-29 (America/New_York)

Branch: `codex/ref-gh-fully-subtracted-gauge-repair-20260829`

Frozen parent: `223947486ac4498bab2e197feca56462c77e6d76`

## Phase 8 launch checkpoint

The Phase-7 ladder passed, so the next authorized scientific step is the
repaired positive-time discriminator.  The first job is deliberately bounded
to the frozen Case-D `96^3` STANDARD, `gamma0=gamma2=1`, gauge-enabled setup
through `t=3M`.  It retains q=1 with both q-control modes disabled, FD4, RK4,
CFL 0.05, KO 0.02, the existing gauge gains, the `[-2M,2M]^3` outflow box,
and the complete puncture-stencil diagnostic exclusion.  Field outputs are
disabled; restart checkpoints remain at 1M intervals on Aurora.

`scripts/ref_gh/aurora_fully_subtracted_phase8_t3.pbs` reuses the already
qualified Phase-7 production executable only after checking its SHA-256 and
proving that `src/` is identical between the build commit and launch commit.
It performs no rebuild and no additional generic GPU test.  The job stops at
3M even if successful; a 5M restart continuation is permitted only after the
3M histories are reviewed.

`scripts/ref_gh/analyze_fully_subtracted_phase8.py` records all finite-state,
timestep, GH/reduction/curl, metric-error, near-puncture, RHS-max-location, and
gauge-off-control comparisons without hiding them behind a tunable scientific
threshold.  Its explicit old-mode discriminator uses the frozen Case-D GH
growth rate `26.654903904216415/M` and flags a recurrence only when the new fit
is within 25 percent of that rate with `R^2 >= 0.95`.  A local replay exactly
recovers the frozen Case-D e-folding time `0.037516548684379786M` and rejects
that run, while accepting the completed Case-A gauge-off 5M control.  These are
analyzer-validation results, not repaired evolution evidence.

The first launch, Aurora debug job `8791456`, did not produce evolved
scientific evidence.  The source-identical Phase-7 executable initialized the
96^3 state on 12 distinct PVC tiles and reproduced the exact-zero residual
gauge sectors at cycle zero.  Immediately after cycle-zero output, all ranks
reported Level Zero `NotPresent` GPU write page faults and the Intel runtime
aborted with PBS exit 134.  No positive time was reached.  This is currently
classified as a first-stage PVC task/view portability failure, not recurrence
of the old `0.0375M` mode and not a formulation verdict.  Compact logs are in
`phase8_aurora_8791456_cycle0_gpu_fault`; the 1.46-GB initial restart is
recorded by remote path but not committed.  A single fence-instrumented cycle
is the next bounded discriminator before any further 3M attempt.

The focused localization launcher enables only the existing
`debug_task_fences` option, caps the run at one RK cycle, and disables restart
and field output.  It retains every scientific parameter and the same 12-tile
96^3 workload.  Its sole purpose is to identify the last completed task fence;
it cannot support an evolution or stability claim.

Aurora job `8791465` executed that localization on node `x4117c6s7b0n0`.
Across 12 ranks it recorded 60 completed instances of each CalcRHS subkernel
and each projected metric/gauge boundary kernel, 48 RK updates, and 12 final
timestep reductions before the same Level Zero write fault appeared.  The
interleaved stdout cannot attribute the original corruption to the last line
printed, and the absence of an error at a Kokkos fence does not exclude an
earlier in-bounds corruption of an adjacent allocation or a GPU-aware MPI/view
lifetime defect.  PBS exit 143 reflects peer termination after device abort;
no positive-time history was produced.  This narrows but does not resolve the
portability blocker.  Bounds/ASan instrumentation is required before any
production correction or renewed 3M attempt.

The first local Debug/Serial ASan+UBSan/Kokkos-bounds discriminator found a
separate host-side undefined behavior before evolution: the temporary
`OutputParameters` object was default-initialized, so its implicit copy loaded
format-specific `bool` members that had never been assigned.  UBSan observed
the invalid value 11.  Commit `c19c6058` value-initializes the temporary object;
this is an output-lifecycle correction and changes no Ref-GH mathematics or
numerics.  The corrected build completed all four RK stages, final timestep,
diagnostics, and final history output for a bounded `16^3`, one-cycle exact
matched run at `t=0.01`, with no sanitizer or Kokkos-bounds report.  Compact
evidence is preserved under `phase8_local_sanitizer_20260829`.  This local pass
does not exercise GPU-aware MPI and therefore does not clear the PVC blocker.
A local MPI sanitizer attempt was unusable because the OpenMPI launcher itself
entered the Intel DRM flush wait before spawning ranks; it was terminated and
is not counted as numerical evidence.

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
ill-conditioned full-driver input rather than being hidden.  No intended
continuum equation or frozen numerical parameter has changed.  The repaired
64/96/128 cycle-zero ladder has passed, but no repaired positive-time
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
or supplying the smallest positive normal binary64 value for either time
derivative disables the predicate.  A subnormal is deliberately not used in
this portability test because Aurora's active Intel floating-point mode may
flush it to zero before the exact production predicate sees it.

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
the exact `4.13003e-14` all-61 result.

Focused rerun `8791265` compiled the staged source at commit
`7838324471d0ecb2fe7592bd497230fc5a7d4c40`, but IGC again segfaulted while
lowering the complete `source_unit.cpp` device image.  The log establishes why
arithmetic staging alone was insufficient: Kokkos 4.7.2 explicitly supplies
`-fsycl-device-code-split=off`, so every independent kernel in this 6000-line
validation translation unit remains in one SPIR-V module.  Mature FO-GH unit
tests and Z4c production execution instead keep focused device kernels in
smaller compilation units.  The follow-up therefore overrides device splitting
to `per_kernel` only for `source_unit.cpp` when the validation oracle and SYCL
are both enabled.  Production executables omit this file and retain Kokkos'
default.  The equations and runtime task graph are unchanged.  Aurora device
qualification remained open pending the same focused gate.

Focused rerun `8791292` used commit
`28eee9d8efaab0d567fcbeda5ae549210c87d9f8`.  The complete executable compiled
successfully with the source-unit-only `per_kernel` option, and all 12 ranks
reported distinct Level Zero PVC tile mappings from `0.0` through `5.1` on
Aurora node `x4516c7s4b0n0`.  This resolves the prior IGC image-size blocker.
The one-rank source-unit process then stopped in its first host-side predicate
check before any oracle kernel ran.  Its false-case test supplied
`denorm_min()` as a nonzero rate/acceleration, but the active Intel mode flushed
that subnormal to zero and thereby presented the exact accepted state.  The
follow-up changes only those two test probes to `min()`, the smallest positive
normal value.  The exact production predicate, equations, tolerances, and task
graph are unchanged, and a fresh local Kokkos Serial source-unit run passes
with the identical all-61 maximum `4.13003e-14`.

Job `8791292` therefore establishes successful PVC compilation and distinct
12-tile mapping, but not all-61 device equivalence, production fixed-point
execution, or evolution.  Its full build log, compile-option proof, mapping,
provenance, PBS record, and checksums are preserved in
`phase6_aurora_8791292_predicate_failed`.

Focused job `8791352` used the test-portable predicate probes at commit
`4ec57dd25358ac0d774f4cb5f1c7a89b041dac67`.  The one-tile PVC source-unit
suite passed completely.  In particular, the moving-reference mixed-jet path
passed and all 4320 compatible/STANDARD all-61 comparisons passed with
conditioned maximum `5.46091e-14` under the unchanged `256*epsilon` gate.  The
12-tile 96^3 STANDARD exact matched fixed point then completed with exact-zero
actual Hhat/theta/Upsilon, ordinary-gauge Pi increment, driver
Hhat/theta/Upsilon, and KO Hhat/theta/Upsilon.  Its reproduction and production
rerun conditioned errors were both exactly zero against tolerance `5e-13`.
All 12 ranks reported distinct Level Zero PVC mappings from `0.0` through
`5.1`.

PBS nevertheless returned exit 1 after the numerical work because Aurora's
default Python 3.6.15 rejected the diagnostic-only f-string debug syntax
`{reproduction=}` in the postprocessor.  The saved TSV passes the unchanged
parser under Python 3.12.3, and the launcher now spells only that unused error
message in Python-3.6-compatible form.  No numerical rerun is required to
reinterpret already-written values.  This is therefore a numerical Phase-6
device pass with a transparently preserved wrapper failure, not a green PBS
job.  It qualifies only the named device oracles and one 96^3 cycle-zero fixed
point; it is not positive-time evolution evidence.

## Phase 7 diagnostic preparation

The cycle-zero stationary diagnostic now evaluates the existing
cancellation-free physical target on the actual stored state and reports
`deltaF_A`, delta conformal Gamma, and delta shift separately.  It retains the
legacy independently reconstructed `F-H_constraint` value as a secondary
conditioning diagnostic.  This changes no evolved field, RHS, task ordering,
tolerance, stencil, or mask.

A local 16^3 STANDARD matched-state check reports exact zero for all three new
residual quantities.  Stored Hhat/theta, actual and driver gauge RHS sectors,
the ordinary-gauge Pi increment, and all gauge KO sectors are also exact zero.
The remaining total Pi RHS is `5.681872526233013e-14`, entirely in the existing
covariant-vacuum source.  The complete source-unit suite remains unchanged at
`4.13003e-14` for the 4320 all-61 comparisons.  The dedicated Phase-7 launcher
and Python-3.6-compatible analyzer will next apply these exact-zero gates at
64^3, 96^3, and 128^3 with the full FD4+KO puncture stencil excluded.  This
local diagnostic checkpoint is not the three-resolution ladder result.

## Phase 7 Aurora stationary residual ladder

Aurora job `8791429` passed with PBS exit 0 at commit
`3c9a34c8c3123c2570eb33e8ec77368feb1f1c61`.  A production source-unit-off
SYCL image ran across 12 distinct PVC tiles.  The fixed physical box and 16^3
MeshBlocks produced 64, 216, and 512 MeshBlocks at 64^3, 96^3, and 128^3.
The FD4+KO mask excludes the full three-cell stencil, so the first included
radius moves as

\[
 r_{\min}/M=0.2231696384,\ 0.1487797589,\ 0.1115848192,
 \qquad r_{\min}/h=\sqrt{12.75}.
\]

At all three resolutions the stored Hhat/theta residuals, production
`deltaF_A`, delta conformal Gamma, delta shift, actual and driver
Hhat/theta/Upsilon RHS, ordinary-gauge Pi increment, and all gauge KO sectors
are bitwise zero.  Reproduction and production-rerun decomposition errors are
also exactly zero.  Hence the frozen discriminator's moving-`r/h`
`r^-5.2 -> r^-7.27` gauge envelope is absent after analytic subtraction.

The old full-target cancellation diagnostic is intentionally retained and
still grows from `3.03757e-13` to `1.49782e-11`.  It no longer enters the exact
matched production path.  The complete nonzero RHS is confined to the existing
covariant-vacuum Pi source and changes from `2.07099e-13` through
`3.10020e-13` to `3.80172e-13`.  The frame-reference Ricci diagnostic remains
at `1.39e-16` to `3.33e-16`, while the secondary coordinate-reference Ricci
reconstruction grows from `4.14e-10` to `4.69e-8`.

For the strict static, uncontrolled, unprescribed q=1 STANDARD dispatch, the
appropriate checkpoint classification is:

`FULLY SUBTRACTED GAUGE FORMULATION ALGEBRAICALLY QUALIFIED`

This claim does not extend to positive-time behavior, moving references,
feedback/prescribed q, or production readiness.

## Remaining gates

The following are not yet complete: independent all-radius high-precision
qualification of cancellation-free \(\delta F_A\), independent qualification
of the direct \(\Delta B_a\) and derivative evaluator below the conditioned
binary64 region, the 3M/5M discriminator, the evolved resolution ladder, 20M,
and conditional 100M qualification.
The Phase-6 PVC all-61 and 96^3 exact fixed-point workloads pass.  Exact
matched-state fill and strict static q=1 residual production dispatch are
locally and device tested.  General moving-reference production dispatch
remains legacy pending a separate qualification.  Performance optimization
remains out of scope for this campaign.

## Phase 8 PVC bounds/lifecycle gate

Aurora debug job `8791507` tested commit
`c3ead1836e5c69aceab277e092a8dd2b81149c13` on twelve distinct Level Zero PVC
tiles with Kokkos bounds checking enabled. The exact 96^3 STANDARD,
gamma0=gamma2=1, gauge-enabled q=1 state completed initialization, cycle-zero
diagnostics, physical boundaries, communication cleanup, and a finite t=0
history row. The first evolved stage then reproduced the Level Zero level-2
`NotPresent` write fault before any RK update or positive-time history. PBS
recorded exit 143. No Kokkos bounds violation was reported.

The rank-tagged trace narrows the synchronous boundary: ranks 1--11 completed
the post-RHS-zero fence, while rank 0 did not print it. Interleaved rank output
does not establish that zeroing is the corrupting operation, because completed
ranks can already have entered the following Psi kernel. The output-parameter
value-initialization correction therefore fixes genuine local UB but is not the
sole PVC cause.

Compiler diagnostics report approximately 1289--1296 spilled Reals for the
analytic active-cell RHS kernel. The fully subtracted analytic dispatch
currently embeds its gauge-driver calculation in that already large source/Pi
kernel, unlike the separate-purpose Z4c RHS pattern and unlike the existing
generic Ref-GH gauge kernel. This private-memory evidence motivates one
equation-preserving portability correction: dispatch the analytic gauge driver
through the existing separate kernel and remove only its duplicate block from
the main source/Pi kernel. The diagnosis remains a hypothesis until that exact
source passes a focused PVC evolved-cycle gate.
