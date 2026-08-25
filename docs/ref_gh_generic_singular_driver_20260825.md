# Generic singular-reference Ref-GH driver: implementation and qualification log

Status: work in progress. The stationary hyperbolic-gauge gate has advanced,
but the literal same-`r/h` puncture-exponent comparison remains a hard stop for
closed-loop control. No controller or wormhole-evolution claim is made.

## Frozen base and branch

- Repository: `HengruiZhu99/athenak`
- Required parent branch: `codex/ref-gh-gaussian-reference-gamma2-20260824`
- Local, upstream-tracking, and remote parent SHA before editing:
  `0e248310a562c8a84327421eecf70f2f5d1da4a3`
- Work branch: `codex/ref-gh-generic-singular-driver-20260825`
- Kokkos submodule: `6739bc623081648af9e752b616d9671527922cbf`

The pre-existing dirty worktree at `/home/hzhu/Desktop/research/gr/athenak`
was not modified.  All work described here is in the clean mandated-base
worktree `/home/hzhu/Desktop/research/gr/athenak-ref-gh-feedback-continuation-20260823`.

## Implemented in this checkpoint

1. A device-callable, reference-independent local estimator

   \[
   q_{\rm loc}=-\frac16 X^k\gamma^{ij}\partial_k\gamma_{ij}
   \]

   from the physical coordinate metric and derivatives reconstructed from the
   first-order state.  The diagnostic lapse exponent

   \[
   p_{\rm loc}=X^k\partial_k\ln\alpha
   \]

   is evaluated from \(\alpha=(-g^{00})^{-1/2}\).  Invalid spatial or spacetime
   metrics fail closed; no values are clipped.

2. The specified actual-cell shell and weights:

   \[
   2h\le r<8h,\qquad 8h<R_G/2,\qquad w=(2h/r)^3.
   \]

3. An independent fourth-order diagnostic that reconstructs the physical
   \(\gamma_{ij}\) at neighboring Cartesian cell centers and applies AthenaK's
   centered spatial stencil directly to those components.  It does not use
   `Phi`, radial rays, interpolation, binning, or extrapolation to the puncture.

4. A Gaussian-localized analytic reference provider

   \[
   \sigma_q=\exp[-q(t)W(r)\ln(r/M)],\qquad
   W=\exp[-(r/R_G)^2],
   \]

   used as the isotropic positive spatial Cholesky factor.  The lapse is one
   and the shift is zero in this first reference-jet gate.  The prescribed
   trajectory uses a quintic endpoint-flat transition and analytic two-jets;
   no logarithm is hidden or regularized away.

5. Staged-cache integration for the new `generic_singular` reference, including
   provider metadata and the independent full-geometry cache oracle.

6. A device scan over all requested prescribed trajectories
   \(\tau/M=4,8,16\), widths \(R_G/M=2,3,4\), and
   \(h/M=1/16,1/24,1/32,1/48,1/64,1/128\).  A static \(q=1.5\)
   control is recorded beside every dynamic midpoint case.

## Initial-geometry estimator evidence

The first-order-state estimator passes the pointwise checks:

- Minkowski: exactly zero.
- Wormhole: maximum error against
  \(q_{\rm loc}(r)=M/(r+M/2)\) is at most \(1.6\times10^{-15}\).
- Stationary trumpet: maximum error against the independent profile-table
  derivative is at most \(7.1\times10^{-13}\).
- The weighted estimates approach the expected asymptotic values without being
  forced to equal them at finite radius.

The production estimate uses the complete shell. Following the subsequent
puncture-stencil instruction, the direct-FD comparison conservatively discards
a point whenever the puncture lies in the complete axis-aligned support box of
any contributing stencil. For the fourth-order direct derivative this is the
box \(|X^i|\le2h\) in all three directions. The `safe state` and direct-FD
columns use the same retained points.

| \(h/M\) | wormhole production | wormhole safe state | wormhole FD | trumpet production | trumpet safe state | trumpet FD |
|---:|---:|---:|---:|---:|---:|---:|
| 1/16 | 1.32184 | 1.11591 | 1.11851 | 0.856373 | 0.791536 | 0.792402 |
| 1/24 | 1.48531 | 1.30752 | 1.31097 | 0.902691 | 0.856383 | 0.857347 |
| 1/32 | 1.58484 | 1.43073 | 1.43477 | 0.926983 | 0.891313 | 0.892331 |
| 1/48 | 1.70032 | 1.58002 | 1.58484 | 0.951826 | 0.927678 | 0.928754 |
| 1/64 | 1.76543 | 1.66723 | 1.67252 | 0.964342 | 0.946240 | 0.947345 |

The selected production shell contains 2,144 cell centers with effective
sample size \(N_{eff}=595.547\). The puncture-clear direct-FD subset contains
480 centers with \(N_{eff}=370.885\), at every resolution.

## Hard-gate failure: direct FD cannot converge on a fixed-r/h shell

The strict test exits nonzero because the direct-FD estimator does not converge
toward the first-order-state estimator. The puncture-overlap exclusion reduces
the discrepancy substantially, but it still grows from 0.00260102 to
0.00528934 for the wormhole and from 0.000865660 to 0.00110559 for the trumpet
between \(h=M/16\) and \(M/64\).

This is not a threshold choice.  For a pure power-law spatial metric
\(\gamma_{ij}=r^{-2q}\delta_{ij}\), write a selected cell as
\(x^i=h\xi^i\).  Any fixed centered stencil has

\[
D_h\gamma(h\xi)=h^{-2q-1}D_1\gamma(\xi),
\]

while \(X^k=h\xi^k\) and \(\gamma^{ij}=h^{2q}\delta^{ij}\).  Therefore
\(X^k\gamma^{ij}D_k\gamma_{ij}\) is independent of \(h\) at fixed
\(r/h\).  Unless the stencil differentiates that singular power exactly, its
bias cannot vanish under refinement.  For the fourth-order stencil, the
weighted pure-power limits on this shell are approximately 0.989337 for
\(q=1\) and 1.575776 for \(q=2\).

Consequently the combined success claim `LOCAL PUNCTURE-EXPONENT ESTIMATOR
ESTABLISHED` is not supported under the controlling rules, even though the
production first-order-state estimator itself passes its analytic checks.

## Prescribed Gaussian reference-jet evidence

At the midpoint of the \(q:2\to1\) trajectory, the staged production cache and
the independent full reference geometry agree to a conditioned scaled
\(L_\infty\) error of \(1.11\times10^{-15}\).

For the representative \(\tau=8M,R_G=3M\) scan:

| measure | \(h=M/16\) | \(h=M/128\) | selected model |
|---|---:|---:|---|
| \(|\dot q W\ln\rho|\) | 0.683316 | 1.170901 | log |
| \(\dot q^2W^2\ln^2\rho\) | 0.466920 | 1.371010 | log-squared |
| reference Ricci maximum | 1.42809 | 4.11641 | log-squared |
| total source maximum | 2.85617 | 8.23283 | log-squared |

The compact analyzer reports `prescribed_q_gate=PASS` and zero dynamic groups
with classified positive algebraic growth.  This supports only the prescribed
reference-jet gate; it is not evidence for a controller or a physical solution.

## Standard gamma0/gamma2 checkpoint

Runtime `<ref_gh>/gamma2` now defaults to zero. For fixed `gamma1=-1`, the
implemented additions are

\[
  \delta\Pi_{AB,t}=-\gamma_2\beta^I C_{IAB},\qquad
  \delta\Phi_{IAB,t}=\alpha\gamma_2 C_{IAB}.
\]

The device algebra checks the forward/inverse characteristic map
\(u^{1\pm}=\Pi\pm s^I\Phi_I-\gamma_2\Psi\), the standard symmetrizer,
and the frozen reduction and curl subsidiary damping to
\(3.33\times10^{-16}\).

A six-case periodic runtime matrix at \(t=0.2\) now covers
`gamma2=0,0.5,1.0` with KO off and on. Every reduction and curl history agrees
with

\[
 \|C(t)\|_2=\|C_{\gamma_2=0}(t)\|_2e^{-\gamma_2t}
\]

to maximum absolute growth-factor errors \(7.07\times10^{-7}\) and
\(2.87\times10^{-6}\), respectively. The `gamma2=0` histories independently
measure the KO contribution instead of folding it into gamma2.

For gamma0, a separate reduction-clean transverse GH constraint eigenmode is
constructed from the linearized subsidiary system in Lindblom et al. (2006),
Eq. (21). For `gamma0=0.25,0.5,1.0`, with KO off and on, all six histories obey
\(\|C_y(t)\|_2=\|C_y(0)\|_2e^{-(\gamma_0/2+\lambda_{KO})t}\) with maximum
error \(5.35\times10^{-7}\). The largest accompanying reduction norm is
\(7.19\times10^{-18}\), and curl is zero. These tests establish the stated
local linearized/robust subsidiary behavior, not nonlinear black-hole,
long-time, SMR, or GPU constraint control.

## Subsequent hyperbolic-gauge checkpoint

The code now evolves 61 fields: the original 50 Einstein fields plus
\(\widehat H_A\), \(\theta_A\), and \(\Upsilon^i\). The sign relation was
implemented after equating

\[
C^a=\widehat H^a+g^{bc}\Gamma^a{}_{bc}
   =H_{bg}^a+g^{bc}(\Gamma^a{}_{bc}-\bar\Gamma^a{}_{bc}),
\]

which gives

\[
H_{bg}^a=\widehat H^a+g^{bc}\bar\Gamma^a{}_{bc}.
\]

The improved Lindblom--Szilagyi driver, exact frame-motion terms, advective
1+log target, conformal Gamma-driver target, nonzero-\(\widehat H\) source,
combined characteristics, boundary communication, output, and restart paths
are implemented and locally tested. Raw gauge fields were not puncture regular;
the evolved arrays therefore store equation-preserving differences from an
analytic static-reference gauge baseline. The exact trumpet is a roundoff
fixed point through \(t=1\) at three resolutions.

With KO enabled, the full evolved stencil reaches three cells rather than the
two-cell centered-derivative radius. Corrected source/history and interpolation-
support masks use this maximum footprint. A regular perturbed
\(24^3/32^3/48^3\) ladder is finite through \(t=1\), with field and native-
constraint L2 self-orders 4.700 and 3.948 at the final time. Exact checkpoint
orders are approximately, but not uniformly, fourth order.

Time-dependent-reference gauge subtraction now uses a minimal mixed jet rather
than a full third-order tensor. Each scalar reference jet carries the twelve
components \(\partial_t\partial_i\partial_q\); staged kernels form
\(\partial_t\partial_i H_A^{ref}\) and then differentiate

\[
 \theta_A^{ref}=-\beta^i\partial_iH_A^{ref}
 -(\Omega_{At}{}^B-\beta^i\Omega_{Ai}{}^B)H_B^{ref}
\]

analytically. The delta-theta RHS subtracts this result. Provider storage grows
from 64 to 100 Reals, symmetric metric storage reduces the staging workspace
from 416 to 410 Reals, and the hot 313-Real evolution cache is unchanged.

Closed-form jet tests pass to \(2.22\times10^{-16}\). Independent validation
against a fourth-order time difference passes for the smooth lapse and moving-
spatial-frame references at every tested RK stage (maximum scaled errors
\(4.94\times10^{-14}\) and \(2.89\times10^{-15}\)), and for the generic
reference at the interior time \(t=4M\) (\(6.71\times10^{-13}\)). Serial and
Kokkos OpenMP source/cache oracles pass, and both smooth references complete a
one-cycle binary64 full-state smoke run.

One limitation is deliberately preserved: the generic transition's clamped
quintic smoothstep is only \(C^2\) at \(t=0\). A centered fourth-order
validation stencil that crosses that endpoint is therefore not a valid
fourth-order oracle and fails by \(2.65\times10^{-6}\). The threshold was not
weakened. The analytic generic path completes one bounded cycle with the
endpoint finite-difference oracle disabled, but that is neither stability nor
convergence evidence.

Primary sources:

- [Lindblom et al., A New Generalized Harmonic Evolution System](https://arxiv.org/abs/gr-qc/0512093)
- [Lindblom et al., Gauge Drivers for the Generalized Harmonic Einstein Equations](https://arxiv.org/abs/0711.2084)
- [Lindblom and Szilagyi, An Improved Gauge Driver for the Generalized Harmonic Einstein System](https://arxiv.org/abs/0904.4873)

## Qualification status and next decision

- First-order-state local estimator: analytic initial-data checks passed.
- Direct-FD same-shell comparison: failed for a scale-invariance reason.
- Prescribed Gaussian reference two-jets/cache: passed locally on Kokkos Serial.
- Closed-loop q control: not started; blocked by the hard estimator gate.
- Standard gamma0/gamma2 damping: local algebra plus the KO-off/on GH,
  reduction, and curl subsidiary matrices passed.
- Hyperbolic gauge driver and physical target: implemented; local oracles and
  static-reference stationary fixed-point gate passed.
- Smooth time-dependent gauge-reference subtraction: analytic mixed-jet and
  one-cycle local CPU gates passed; PVC portability is untested.
- Generic time-dependent subtraction: interior-time analytic gate passed and a
  one-cycle finite smoke completed, but the clamped-quintic `t=0` validation
  endpoint remains an explicit smoothness limitation.
- Regular perturbed stationary trumpet: finite and approximately fourth-order
  in masked L2 self-differences through `t=1` on local uniform grids.
- New 61-field restart path: focused direct/split gate passed.
- Generic puncture stability/convergence, closed-loop q recovery, wormhole
  evolution, SMR, CUDA, and Aurora PVC: not established.

No Aurora job was submitted in this checkpoint. The SSH ControlMaster socket is
currently absent, and noninteractive login cannot satisfy Aurora's keyboard-
interactive authentication. Local B580 Level Zero enumeration also hung, so no
SYCL build or device claim was made.

The controlling specification needs one clarification before closed-loop work:
either retain the first-order-state estimator as the controller quantity and
validate the direct-FD diagnostic on a fixed physical annulus, or replace the
direct-FD convergence demand with a precomputed fixed-\(r/h\) stencil-bias
comparison.  Both preserve actual Cartesian sampling and avoid interpolation,
but neither is the literal current same-shell convergence requirement.
