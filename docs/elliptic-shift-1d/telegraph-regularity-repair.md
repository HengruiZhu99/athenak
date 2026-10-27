# Puncture regularity repair: candidates and controls

The user authorized repairing the late puncture failure and asked whether the telegrapher term already has conformal-factor attenuation. The previous `origin/project/telegrapher_lapse` implementation contracts the flux derivative with `oopsi4 * g_uu`, where `oopsi4 = chi_guarded**(-4/chi_psi_power)`. The current spherical lapse correction already carries `chi`; its AF-regular window is one at the puncture. The flux relaxation equation has no additional chi factor. Conformal compactification and the reference-reduced lapse mean these are related prescriptions, not identical coordinate equations.

The unfloored long trial failed at t43.836474609375 through log-lapse underflow. The default physical floor1e-10 trial failed later at t84.798291015625. At its terminal checkpoint, node4 has conformal lapse7.59e12, while the Courant-maximizing node3 remains floored. The elliptic shift is3.93e7 at node3. Thus lower-bounding the lapse does not prevent a positive logarithmic RK overshoot or the resulting shift amplification. Both failed trials and their terminal paper plots remain preserved.

## Explicit numerical controls

- `lapse_update=log` preserves the historical RK variable. `total` integrates the conformal lapse directly at ordinary positive-radius RK stages, floors the physical lapse, then converts to logarithmic storage. The physical compensation remainder includes storage/reconstruction roundoff. The AF coefficient endpoint and scri retain their original time variables. This changes the discrete method, not the continuum gauge equation.
- `puncture_stencil=coefficient` preserves the historical stencil. `positive` excludes the r=0 asymptotic-coefficient slot from bulk derivative and advection stencils. Every positive-radius node remains evolved. AF endpoint coefficients are still independently evolved; unused exponentials of those coefficients are avoided. This is a disclosed finite-grid inner closure and requires validation; it is not an exact asymptotic boundary theorem.
- `inner_total_derivatives_radius` defaults to0. Positive values select a separate tested-but-unsuccessful control: direct reconstructed-field derivatives inside half the specified radius, blended back to historical chart derivatives by that radius. Upwind transport uses the same variable choice. This option is **disabled in the current duration candidate**.

Nine targeted implementation test groups pass, including independent projected total-lapse RK, compensated split restart, CLI method-binding rejection, positive-node polynomial derivatives, finite large endpoint coefficients with no bulk feedback, and Python/native attenuated rate and characteristic comparisons. Passing these tests establishes implementation agreement, not nonlinear stability.

## Gauge attenuation candidate

For nonnegative input epsilon=`lapse_attenuation` (default0), define

    A = (1+r^2)/2
    s(alpha,r) = (1+epsilon)*alpha/(alpha+epsilon*A)
    delta D_t alpha = s*chi*[w*B_r + (w_r+2*w/r)*B]
    tau D_t B = kappa*w*(alpha-A)_r - B.

Zero epsilon recovers the original equations. For positive epsilon, s=1 on the reference lapse alpha=A. The logarithmic source is `(1+epsilon)*chi*div(wB)/(alpha+epsilon*A)`, so its coefficient stays bounded as alpha approaches zero. The factor is outside the existing divergence: no derivative of s is inserted. This is an explicitly modified gauge equation, not merely a new implementation of the historical equation.

The frozen coupled speeds become `-beta +/- sqrt(chi*(F+s*kappa*w*w/tau))`. Characteristic tracking and the timestep guard use this modified expression. The current trials use epsilon1e-4, total-lapse RK, positive-node inner closure, physical floor1e-10, and **no direct-total-derivative collar**. The Einstein/Z4 variables remain independently evolved. No constraint projection, shift cap, positive-radius excision, or lapse upper bound is used.

The predeclared duration criteria and full input are in `telegraph-attenuated-duration-protocol-v1.json` and `inputs/elliptic_shift_1d/telegraph-attenuated-duration-v1.json`. Primary acceptance uses the N1025 apparent-horizon-exterior constraints and apparent-horizon mass. N513 is an exploratory resolution control. A single-grid duration pass would still not be convergence evidence.

## Preserved unsuccessful controls

| Control | Resolution | Outcome |
|---|---:|---|
| Total-lapse RK from degraded t84.75 profile, zero compensation |1025|Fails at84.8822265625; explicitly not an exact restart|
| Fresh total-lapse RK, historical coefficient stencil |257|Speed guard at17.27578125|
| Same |513|Speed guard at42.60048828125|
| Total-lapse RK plus positive-node stencil |257|Speed guard at32.90703125|
| Above plus direct-total derivatives inside radius0.001 |257|Positivity guard at4.74140625|
| Same |513|Positivity guard at5.67431640625|
| Attenuation epsilon1e-4, total RK, positive-node stencil |257|Speed guard at57.1580078125; exterior accuracy already unacceptable|
| Same with epsilon1e-3 |257|Speed guard at73.5423828125; exterior accuracy already unacceptable|

The N257 attenuation pilots had approximately7% apparent-horizon mass error and exterior H RMS about0.75 by t31.25. Their longer survival does not constitute success. No failed trajectory was restarted or relabeled. Fresh N513 and N1025 epsilon1e-4 trials target500 under `output/elliptic-shift-1d/telegraph-attenuated-duration-v1`; their final assessment is pending.


## Duration assessment and interim evidence

Use `paper/collect_long_puncture.py` with the explicit candidate `--case`, current `--protocol`, and a new export directory. Running exports require `--through`. Then run `tst/elliptic_shift_1d/assess_telegraph_duration.py --export EXPORT --output NEW_ASSESSMENT.json`. The assessor applies only the declared N1025 budget, rejects missing report times and copied-manifest receipt mismatches, and never promotes a healthy partial trajectory to a duration pass. Recorded stage extrema may extend beyond an interim saved-profile cutoff; both times are labeled. Seven gate tests cover incomplete duration, terminal failure despite healthy exterior, missing reports/horizon, nonfinite constraints, mass and boundary violations, coarse-grid mislabeling, and an explicitly synthetic complete-coverage positive control.

The validated export `evidence/telegraph-attenuated-interim-t60-v2/n1025` contains all1201 saved profiles through t60 in its receipts. Its assessment is **pending**: the saved-state accuracy budget passes to this cutoff, but target500 is not reached. Peak relative AH error is2.64272e-6 (initial transient); late exterior RMS maxima are H1.79450e-7, M2.44246e-6, Theta4.99959e-8, Z5.21288e-7. Captured stage extrema through t68 include scri-pole2.83285e-6 and mappedCFL0.200000000008. These are duration-candidate diagnostics, not convergence evidence.

A live-export race was found while constructing the assessment: the collector could read a manifest, then hash a newer version before copying the earlier parsed content. The collector now reads each manifest once and hashes/copies those exact captured bytes; history receipts similarly hash the exact bytes parsed. The rejected first t60 export is retained under tmp as a receipt-race diagnostic, not scientific evidence. No numerical run was changed or restarted.


The interim figure `evidence/telegraph-attenuated-interim-t60-v2/repair-status.png` shows exterior constraints, relative AH mass error, physical inner lapse, and conformal lapse across the domain. It uses the validated export without smoothing the radial samples. The lapse at0<r<=1e-4 changes by at most3.04355e-4 fractionally between t25 and60 (maximum absolute change5.04193e-10). Five negative neighboring differences remain; the largest local fractional drop is0.250054, so the near-puncture ripple is not claimed eliminated. The minimum saved physical lapse in this region at60 is4.61814e-7, above the1e-10 floor. The maximum coupled speed in this region is positive6.52221e-5, so this region cannot be assumed pure gauge outflow. These diagnostics support finite-grid settling only; neither a continuum noncrossing claim nor an excision argument follows. Profile hashes and measurements are recorded in lapse-settling.json.
