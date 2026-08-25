# Generic singular-reference Ref-GH driver: implementation and qualification log

Status: work in progress, stopped at the initial puncture-exponent hard gate on
2026-08-24.  No closed-loop controller or black-hole evolution claim is made.

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

| \(h/M\) | wormhole state \(q_{est}\) | wormhole direct FD | trumpet state \(q_{est}\) | trumpet direct FD |
|---:|---:|---:|---:|---:|
| 1/16 | 1.32184 | 1.15447 | 0.856373 | 0.847683 |
| 1/24 | 1.48531 | 1.26485 | 0.902691 | 0.893362 |
| 1/32 | 1.58484 | 1.32892 | 0.926983 | 0.917320 |
| 1/48 | 1.70032 | 1.40061 | 0.951826 | 0.941823 |
| 1/64 | 1.76543 | 1.43989 | 0.964342 | 0.954166 |

The selected shell contains 2,144 cell centers with effective sample size
\(N_{eff}=595.547\) at every resolution.

## Hard-gate failure: direct FD cannot converge on a fixed-r/h shell

The strict test exits nonzero because the direct-FD estimator does not converge
toward the first-order-state estimator.  The weighted discrepancy grows from
0.167366 to 0.325534 for the wormhole and from 0.00869049 to 0.0101752 for the
trumpet between \(h=M/16\) and \(M/64\).

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

## GH and gauge equations still to implement

The parent code evolves only 50 Einstein fields and uses the background
wave-map specialization equivalent to ordinary
\(\widehat H^a=-g^{bc}\bar\Gamma^a{}_{bc}\).  Runtime `gamma2` and evolved
\(\widehat H_A,\theta_A,\Upsilon^i\) do not yet exist.

The sign relation to implement follows by equating

\[
C^a=\widehat H^a+g^{bc}\Gamma^a{}_{bc}
   =H_{bg}^a+g^{bc}(\Gamma^a{}_{bc}-\bar\Gamma^a{}_{bc}),
\]

which gives

\[
H_{bg}^a=\widehat H^a+g^{bc}\bar\Gamma^a{}_{bc}.
\]

The source, frame-motion, full \(\gamma_0/\gamma_2\), characteristic, restart,
boundary, and stationary fixed-point work remains unimplemented in this branch.
The improved driver must be translated from Eqs. (9) and (11) and the combined
characteristics from Appendix B of Lindblom--Szilagyi (2009); the first-order GH
system and damping terms are Eqs. (35)--(37) of Lindblom et al. (2006); the
conformal Gamma-driver target with \(\Upsilon^i\) is Eqs. (60)--(62) of
Lindblom et al. (2008).

Primary sources:

- [Lindblom et al., A New Generalized Harmonic Evolution System](https://arxiv.org/abs/gr-qc/0512093)
- [Lindblom et al., Gauge Drivers for the Generalized Harmonic Einstein Equations](https://arxiv.org/abs/0711.2084)
- [Lindblom and Szilagyi, An Improved Gauge Driver for the Generalized Harmonic Einstein System](https://arxiv.org/abs/0904.4873)

## Qualification status and next decision

- First-order-state local estimator: analytic initial-data checks passed.
- Direct-FD same-shell comparison: failed for a scale-invariance reason.
- Prescribed Gaussian reference two-jets/cache: passed locally on Kokkos Serial.
- Closed-loop q control: not started; blocked by the hard estimator gate.
- Hyperbolic gauge driver, physical gauge target, runtime gamma2: not implemented.
- Stationary trumpet, generic-reference trumpet, wormhole evolution: not run.
- SMR, restart of new state, CUDA, Aurora PVC: not run.

No Aurora allocation was requested because the failure is a local mathematical
discriminator and the specification requires stationary/controller gates before
black-hole production work.

The controlling specification needs one clarification before closed-loop work:
either retain the first-order-state estimator as the controller quantity and
validate the direct-FD diagnostic on a fixed physical annulus, or replace the
direct-FD convergence demand with a precomputed fixed-\(r/h\) stencil-bias
comparison.  Both preserve actual Cartesian sampling and avoid interpolation,
but neither is the literal current same-shell convergence requirement.
