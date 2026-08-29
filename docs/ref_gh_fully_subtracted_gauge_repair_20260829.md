# Fully subtracted Ref-GH gauge repair

Date: 2026-08-29 (America/New_York)

Branch: `codex/ref-gh-fully-subtracted-gauge-repair-20260829`

Frozen parent: `223947486ac4498bab2e197feca56462c77e6d76`

## Claim boundary

This investigation begins from the completed
`GAUGE-DRIVER COUPLING DEFECT ISOLATED` discriminator.  It does not reinterpret
that result as a proof that cancellation is the sole exponential mechanism.
At this checkpoint only the residual-variable algebra has been derived and
symbolically verified.  No production equation has changed, no repaired
fixed-point ladder has run, and no repaired evolution, robustness, or
performance claim is made.

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

## Principal part

The rewrite is an algebraic change of dependent variables plus lower-order
source collection.  It does not change the derivative coefficients of the
STANDARD first-order GH plus driver system.  Therefore the already-audited
standard principal symbol and characteristic fields are preserved.  The
separate compatible-ordering loss of strong hyperbolicity is also unchanged;
compatible ordering remains an oracle/research option and is not the
production candidate.  A full claim about uniform lower-order behavior awaits
the trumpet coefficient asymptotics required by Phase 4.

## Remaining gates

The following are not yet complete: independent all-radius high-precision
qualification of cancellation-free \(\delta F_A\), direct
\(\Delta B_a\) and derivative evaluators, production residual dispatch,
exact matched-state fill, host/device and all-61 equivalence, the repaired
64/96/128 fixed-point ladder, the 3M/5M discriminator, the resolution ladder,
20M, and conditional 100M qualification.  Performance optimization remains
out of scope for this campaign.
