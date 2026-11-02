# N16 zero-restoration failure investigation

The evidence supports a coupled radial gauge/boundary instability at scri. The extra restoring term improves robustness of the present N16 discretization: original A survives10M, whereas NZ1 fails at2.3145M. This is not a proof that the term is essential to the continuum equations. A boundary compatibility defect, an insufficiently resolved boundary mode, and loss of useful damping remain distinguishable possibilities. No science changes or replay were made.

## Evidence sequence

The initial constraint histories agree with A. By0.45M, NZ1's cumulative scri pole maximum is.0569 versus A's.00669, and clock maximum.0782 versus.00211, while own-candidate H/M still agree to about0.02%. By.60M momentum is1.86times A and pole/clock are1.24/2.10, versus.0281/.0150. At2.25M H/M ratios are92.5/585.1. These maxima and diagnostic snapshots do not resolve the exact first unstable mode, but the boundary residuals separate before the large exterior constraint growth. Nullness remains near machine precision throughout; it is an algebraic identity enforced by projection, not a stability certificate.

The failure capture contains finite fields and all1536 surface metrics are indefinite. Their negative eigenvectors align with the radial direction to within about3e-12 in absolute cosine; the failure is nearly spherical across the atlas, not a localized seam or a single corrupted MPI rank. All volume metrics remain positive. The checkpoint is unadmitted, but its stored arrays are the rollback to the last accepted state before the rejected attempt (Step catch restores base and surface_base). Reading these arrays for postmortem analysis does not authorize or perform a restart.

At that last accepted state, angular-mean quantities are:

| Quantity | Last volume row r=.9921875 | Scri r=1 |
|---|---:|---:|
| radial conformal metric | .296581 | .0299047 |
| chi | .0731313 | .0505339 |
| radial beta+x | +.730439 | -.299937 |
| radial auxiliary D | +133.887 | -16930.7 |

At scri, chi*q-1=.68984: null projection remains valid with s=sqrt(chi*q)≈1.29994, but the stationary regularity target chi*q=1 is badly violated. Nullness and this regularity condition are different. The dependent-D closure develops a very large trace across the final half-cell. Original A at10M has radial beta+x≈5.91e-6 and D≈-.1608; this is a healthy reference at a different time, not a matched-time endpoint comparison.

## Immediate metric-collapse mechanism

A separate read-only local probe evaluates the unchanged NZ1 RHS at the rollback arrays. It never invokes Step, LoadCheckpoint, or checkpoint output. Reconstructed dependent D agrees with the stored trace within1.4e-10 absolute despite magnitude~17000. This checks that the probe samples the same closure rather than substituting a different trace.

For a nearly spherical metric, the radial contraction of the metric equation has the form

    dt g_rr = beta^r partial_r g_rr
              + (4/3) g_rr (partial_r beta^r - beta^r/r)
              - 2 alpha A_rr + Q_rr.

The probe computes the full Cartesian contractions, not a spherical replacement. At the final valid surface the angular-mean contributions are:

- metric advection: -41.2141;
- shift strain (including trace subtraction): -11.0798;
- curvature: +.310020;
- remaining contribution, principally angular dissipation: +4.28e-7 mean, at most.00252 in magnitude pointwise;
- total metric rate: -51.9839.

Thus the immediate collapse is driven by advection and shift gradients. The remaining positive radial metric.0299047 divided by the magnitude of its rate gives~.000575 coordinate time, comparable to dt=.0005. This is a late nonlinear fast timescale, not proof that the initial timestep caused the instability or that simply reducing dt fixes it. The isolated angular/penalty CFL values remain.38145/.384; no isolated bound was crossed. Those bounds do not establish coupled stability, and the report explicitly states that the bulk CFL guard is disabled. The last diagnostic bulk estimate is not an exact spectrum at the rejected stage.

There is severe radial stencil sensitivity by this point. The final seven angular-mean metric samples at radii [.9140625,.9296875,.9453125,.9609375,.9765625,.9921875,1] are [1.13897,.97657,2.89404,.82710,.28516,.29658,.02990]. The actual endpoint interpolation derivative is+31.7047, whereas the final-interval secant is-34.1346. With negative radial beta, the positive native derivative drives a negative metric rate. This establishes an oscillatory, underresolved radial boundary profile at failure; it does not establish that underresolution alone initiated it. No lower-order derivative was used to evolve or repair this state.

## Auxiliary-D compatibility warning

The algebraic closure ensures the shift rate matches the differentiated null gauge. It does not enforce the independent differential trace condition

    dt D - dt Lambda - beta.grad(D-Lambda) + D = 0.

The probe differentiates the discrete algebraic D trace in the direction of the full native volume and geometric surface RHS, using symmetric perturbations epsilon1e-8 and5e-9. No time step is taken. The resulting radial dtD is about-5.23e7, whereas the sampled continuum auxiliary equation predicts+1.11e7. The mean mismatch norm is6.34e7. Halving epsilon changes the derivative by only2.71e-9 relative. The two-step NZ1 reference has mismatch below5.43e-6, but this tiny early difference is near finite-difference cancellation and is not an accuracy measurement.

This is a major boundary/interior mismatch in the failing state. It is not by itself proof of a new continuum inconsistency: the trace is differentiated discretely, the comparison samples the continuum auxiliary equation, volume KO enters the directional rate, and there is no independent surface-D filter. The same analysis on healthy A at10M gives a nonzero mismatch norm~1.03 (dtD≈.0373 versus auxiliary prediction≈-.9910). Thus algebraic closure alone is not a verified discretely compatible auxiliary boundary scheme even for A. A refinement/characteristic audit of that differential compatibility is more informative than treating small null residuals as validation.

## Interpretation and next discriminating work

The most supported causal chain is: loss of the additional outer restoring feedback permits a boundary/interior shift mismatch; the dependent-D/geometric system develops strong radial gradients and oscillation; metric advection and strain collapse the radial surface metric. The first arrow is an inference from the controlled A/NZ1 comparison; the later metric-rate mechanism is measured directly. No puncture floor, lapse sign, NaN, storage, or scheduler event caused the numerical stop.

The removed source acts on v=beta+x with strength4WC/Omega, so it can strongly suppress near-scri gauge deviations while being troublesome to resolve at a narrow activation collar. Stability benefit and collar discretization error can coexist. Removing it is therefore not automatically an accuracy improvement. This experiment gives no evidence for an earlier convergent regime.

The authorized N24 run is still running at the last check. Its failure time/profile or survival will test resolution sensitivity, not fourth-order convergence. Further diagnosis should examine linearized boundary modes and the differential D-Lambda trace with matched interior characteristics, separately from changing timestep or restoring profile. No extra long runs or equation changes are initiated by this investigation.

## Reproduction and integrity

probe.cpp and probe-a.cpp compile against the current source matching the frozen NZ1 manifest, with the latter using the original-A flag and NI1 identity. Their command records and logs are saved. The A probe does not modify or reinterpret the A checkpoint. Raw CSVs and probe binaries are under output/zero-restoration/investigation; hashes and exact commands are in provenance.json. analyze.py reconstructs the postmortem radial curves and summary from original snapshots and probe CSVs. radial-postmortem.png was visually inspected. Input hashes were checked before and after all analysis. The failed checkpoint remains unadmitted; the failed capture is never reconstructed as a valid10M endpoint.
