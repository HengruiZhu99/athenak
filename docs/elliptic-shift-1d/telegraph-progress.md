# Telegrapher goal — active, not scientifically complete

The user authorized implementing the spherical conformal telegrapher lapse, running controlled tests locally, tracking characteristics during evolution, and iterating until the lapse front and Hamiltonian errors converge robustly. Historical suspended jobs remain suspended; new runs have separate source snapshots and directories.

## Implementation and checks

The new compiled library wraps the unchanged eight-field geometry kernel and evolves the auxiliary radial flux separately. Its conformal normalization, two angular divergence terms, reference subtraction, compact-support coupling, timestep guards, and stage characteristic system are described in `telegraph-formulation.md`. The protocol is `telegraph-protocol-v1.json`.

The 51-test suite passed, followed by both telegrapher tests including the new 52nd test. Evidence is in `evidence/telegraph-implementation-v1`. Independent Python/compiled nonzero-flux rates agree to 2.04e-12. Disabled flux is bitwise identical to the existing integrator. Nonzero-flux RK stages, stage-ray rates, ray activation, and all compensated restart state are verified. The production CLI also checks initial invariant mass and bitwise resumed geometry/B/rays.

## Preserved configuration failure

The first two pilot batches accidentally used the initial-data constructor's default mass 0.1 instead of 1/3. Their outputs are retained under `telegraph-pilot-v1` and `telegraph-resolved-pilot-v1`, each with a configuration-rejection receipt. They are excluded from comparison to Figure 6 and from parameter selection. The repaired driver passes mass explicitly, records actual initial-data metadata, and exports analytic initial fields/derivatives. Its N513 initial geometry matches the archived Figure 6 control bitwise. The subsequent zero-flux evolution matches the archived t=3 physical fields to 2.3e-12.

## Completed corrected pilots through t=3

All use M=1/3, nu=1, tau=.02, fixed coordinate support, CFL=.05, the full AF domain, and unprojected constraints. The N513 sweep finds that stronger flux can smooth the lapse while increasing errors around the outer taper. Therefore the visibly smoothest curve is not selected as an automatic success.

For kappa=.001, N513/1025/2049 give maximum accepted-state Hamiltonian RMS
1.870759867e-3 / 1.537463471e-4 / 2.705354120e-6.
Their maximum accepted Hamiltonian point errors on r<.8 are
1.539732276e-2 / 1.706909978e-3 / 9.382915309e-5.
At t=2.5, physical-lapse gradient maxima on .03<r<.3 are
8.18780 / 8.29144 / 8.32886, and minimum stage-integrated inward log(J) is
-1.95201 / -1.94880 / -1.94853. The front gradient now stabilizes instead of diverging under refinement.

This is not admission: full-trajectory field peak orders on these grids are only 2.893 bulk and 2.802 endpoint. The finest all-stage geometric scri numerator is 1.9922e-6, above the 1e-7 gate. Fine Hamiltonian errors alone cannot override these failures. No continuum characteristic crossing or full-system stability theorem is claimed.

Saved assessments: `evidence/telegraph-front-pilot-v1` and `evidence/telegraph-front-convergence-v1`. Original simulations and source/executable hashes are under the corresponding `output/elliptic-shift-1d` directories.

## Finer early-front result

The completed N1025/2049/4097 collar sequence through t=3 gives bulk/endpoint field peak orders 3.832/4.819. Finest accepted H RMS is 4.326997e-8 and H point maximum on r<.8 is 3.813122e-6. At t=2.5 the local front Hamiltonian maximum is 5.439811e-7, the sampled physical-lapse gradient is 8.343414, and inward log(J) minimum is -1.948491. These support convergence of the early smoothed front. They do not complete the goal: finest momentum RMS order is 2.389, connection-constraint RMS order is 3.415, and the scri pole maximum is 1.057454e-7, slightly above the unchanged 1e-7 threshold. Evidence: `evidence/telegraph-fine-front-v1`.

## Longer failure and second candidate

Exact-source continuations of the original collar candidate reached t=6 at N513/1025/2049. Maximum accepted Hamiltonian RMS is .0617382/.00711459/.000523335 and the corresponding point maxima on r<.8 are 43.754/10.9438/2.07120. Bulk field peak order drops to 1.300. This candidate fails the declared test.

The inward front traverses the inner coupling taper after t=3. The coupling then vanishes and the front steepens again. By t=6 the gradient maximum also reaches the first physical cell on the middle and fine grids. Thus taper loss and eventual coordinate underresolution both matter; removing the taper alone is not a demonstrated cure. Evidence with verified checkpoint ancestry is `evidence/telegraph-collar-long-v3`.

The `af-regular` candidate removes only the inner taper. The derivation in `telegraph-formulation.md` shows that the added stored lapse rate vanishes quadratically at the AF endpoint, preserving its leading coefficient equation. Independent compiled/Python tests, nonzero-flux RK stages, both-profile CLI restarts and an analytic-factor lapse-gradient check pass. The middle grid's t=6 inward ray-map minimum changes from log(J)=-10.1769 to -8.62071, but its Hamiltonian point maximum is still 11.9187. Stronger ray separation alone does not establish restored Einstein accuracy. See `evidence/telegraph-characteristics-long-v2`.

The characteristic plots distinguish measured ray compression from a continuum crossing claim. A local frozen damping rate cannot be used as a nonlinear acceptance threshold. The AF shift has leading behavior beta=b1*r: its inward advection can compress coordinates exponentially without a finite-time crossing. The saved front approaching the first physical cell requires spatial-resolution assessment before interpreting its endpoint behavior.

## Diagnostic correction

Direct finite differences of physical lapse gave a spurious nonzero endpoint derivative once the AF layer became underresolved. New trajectory plots differentiate the stored log lapse and the analytic AF factor, enforcing the exact finite-time alpha'(0)=0 without altering evolved data. An independent Jet derivative verifies the formula. Earlier plot versions remain preserved with supersession notes. Their field/constraint results are unaffected.

## Work remaining

The AF-regular N2049 run and a separately declared coarse parameter pilot are in progress. The pilot varies kappa at fixed support, and broadens the outer transition at fixed kappa; it cannot establish convergence. Its input and pre-run protocol are `telegraph-af-parameter-pilot-v1.json` and `telegraph-protocol-af-parameter-v1.json`.

Each dispatcher records PID, child PID, command and terminal status in its `launch.json`. Revalidate those handles before waiting or restarting. The t=10 extension, independent dt/shift/KO controls, and parameter-robustness checks remain required. The driver now exposes independent quadrature and dissipation controls; making them available is not evidence that those experiments passed. Historical suspended jobs remain suspended. The goal remains active and incomplete.


## Mapped-grid iteration

All four uniform-grid AF parameter pilots finished t=6 and failed. Raising kappa to .005/.02 worsened the coarse late Hamiltonian maximum to 62.472/141.830. Broadening the outer transition from .08 to .24 barely changed that late inner maximum. These are rejection/diagnosis pilots, not evidence against all relaxation parameters at adequate resolution.

The first numerical map r=sinh(6x)/sinh(6) reduced the N513 maximum H on r<.8 from 46.4475 (uniform, transition=.24) to .110355, but raised the scri pole residual to .0229351 because its outer grid is coarser. N1025 reached t=6 with H maximum .0721403 and scri pole .00158487. Neither passes.

A second C7 map clusters only the inner region and preserves r=x exactly for x>=.4. Its N513 scri pole .000159142 is much closer to the uniform-grid value, but its H maximum .281583 still fails. Both mapped studies are continuing at N2049; the inner-only N1025 also finished. These maps retain the continuum equations, actual scri, the AF endpoint, and unprojected constraints. The optional geometry-kernel changes apply physical-measure quadrature weights, mapped dissipative operators and local scri penalty spacing. Uniform nonzero-flux evolution agrees bitwise with the pre-map library, and all 52 tests passed; the later inner-map helper also passes its targeted tests.

Active output roots: `telegraph-mapped-v1` and `telegraph-inner-mapped-v1`. The next separately declared pilot is `telegraph-inner-mapped-parameters-v1`; it tests fixed kappa/tau on the better-resolved inner grid after a worker becomes free. It is prepared, not yet presumed launched. Read current `launch.json` and manifests for live status. Do not mark this goal complete.


## Latest assessment checkpoint

The global sinh map has now completed all three grids through t=6. The finest H RMS is 3.76936e-5, H point maximum .0465682, and scri pole maximum .000398203. H point convergence is only about .63 order; momentum RMS convergence is about 1.22 order. The global map is rejected under the unchanged criteria. Its plots and review are in `evidence/telegraph-mapped-v1`.

Stage tracking of the original zero-flux controls gives inward minimum J=6.52882e-4 / 1.26791e-5 at t=2.5 for N513/1025, and 1.26247e-5 / 1.21675e-8 at t=3. This is much stronger compression than inferred from the old coarse snapshot cadence. The telegrapher candidate has J about .142 at t=2.5, with resolved spatial agreement. This supports the early gauge-steepening diagnosis but does not constitute a continuum characteristic-crossing proof. See `evidence/telegraph-characteristics-early-v1` and the explicit scope discussion in `telegraph-formulation.md`.

All four inner-map parameter pilots have also completed. Kappa=.005,tau=.02 reduces the coarse H maximum to .0513663 and H RMS to 4.50167e-5, versus .281583 and the larger RMS for kappa=.001. The smaller tau=.005 gives a similar H point maximum but larger RMS at kappa=.005. The selected kappa=.005,tau=.02 candidate is now being refined at N1025/2049 in `telegraph-inner-k005-convergence-v1`, one worker, alongside the remaining kappa=.001 inner-map N2049 run. The already completed N513 parameter case supplies the third grid. None is presumed admitted. Full independent controls and the t=10 extension remain unrun.


## Completed spatial audits and next resolution controls

The original inner-only map at kappa=.001 completed all three grids through t=6 and fails. Finest H RMS is 4.22884e-5, H point maximum .0633067, bulk field order 1.468 and momentum RMS order 1.324. Its complete review is `evidence/telegraph-inner-mapped-v1`.

For kappa=.005,tau=.02 on the same strength6 map, all three grids also reached t=6. Finest H RMS is 1.02021e-7 and H point maximum .00288981. The fine-pair H/M/Theta/Z RMS orders are 4.766/3.517/4.868/4.310. However, bulk field order 3.325, H/M point orders 2.803/2.009, the H point threshold and scri pole threshold fail. The gradient peak difference is 6.1%, but the matched physical-node gradient discrepancy is 15.5%, above 10%. The new `assess_telegraph.py` records these failures explicitly rather than admitting the case from its RMS alone. The N4097 continuation of the resolution sequence is running from initial data through t=6 in `telegraph-inner-k005-fine-v1`.

The optional harmonic-mean mapped KO operator has a verified dr-energy identity and a dissipative spectral bound independent of the smallest map Jacobian. Its full solver suite passed (52 tests); tests of strongly clustered maps include nonzero flux, free constraints, RK-stage rays and exact restart. This is a numerical dissipation variant, not new continuum constraint damping. A bitwise comparison also confirms that the original mapped physical-KO path remains unchanged. See `evidence/telegraph-local-ko-implementation-v1` and the derivation in `telegraph-formulation.md`.

The stronger inner-C7 coarse pilots do not improve monotonically: their first cell shrinks, but their transition coarsens r around .001. `evidence/telegraph-grid-geometry-v1` illustrates relative spacing throughout the domain. A separate exponential-inner map retains roughly constant relative spacing across the inward front path. The three-grid t=6 study of that map is complete in `evidence/telegraph-exp-map-v1`. It still fails: finest H RMS 1.01615e-5, H point maximum .00696867, bulk field order 3.373, momentum RMS order 3.496, and scri pole 1.87977e-6. The maximum per-time matched gradient disagreement is 7.70%, which passes that particular check; the peak-only discrepancy is 6.60%. Its largest H error is associated with the map transition, so small near-puncture cells alone are insufficient.

The full radial sampled invariant audit also records a maximum mass discrepancy of about .007874 on N2049 near r=.999023 at t=.2 for both maps. Small errors at interior extraction radii or at the horizon do not erase this near-scri discrepancy. No accuracy admission is made from those smaller interior numbers.

## Diagnostic extension and dispatch recovery

The exponential-map trajectories are being continued from exact frozen-source checkpoints through t=10 to assess late resolution before committing a finer long run. The N513 continuation finished in `telegraph-exp-extension-v1/n513`. A misspelled second parent path terminated that dispatcher after the valid first child completed; the original launch record, failure receipt, and completed checkpoint are preserved. No completed trajectory was restarted. The corrected N1025/N2049 continuations are under `telegraph-exp-extension-v2`. The launcher now validates all requested parents and source/library hashes before dispatch; its regression test verifies that a missing later parent starts no partial batch.

Active handles and precise times must be revalidated from the corresponding launch records and live processes. The t=10 extension is diagnostic because the t=6 spatial audit fails. Independent timestep/characteristic-time, quadrature, dissipation, and adjacent-parameter convergence controls remain required. The goal is active, not complete.


## Extension results and normalization audit

All three exponential-map runs reached t=10. At the final time, the H point maxima are 1.04816/.277343/.0231198 for N513/1025/2049. The finest accepted H RMS maximum remains 1.01615e-5, set earlier in the trajectory. The final physical-lapse gradient peaks are 34211.6/35017.4/35136.5, with about .34% agreement between the finest two grids. The finest gradient peak is at r=4.07551e-7, with relative cell spacing about .01792. Inward minimum log(J) at t=10 is -17.5588/-17.5542/-17.5544. This is substantial coordinate compression, not a measured continuum crossing. The late H point error still fails the unchanged threshold. Evidence: `evidence/telegraph-exp-extension-v1`.

The gradient audit now normalizes each spatial discrepancy by the fine-grid gradient peak at that same time, then takes the maximum over the complete trajectory. The prior normalization by the largest peak over the whole trajectory could conceal earlier relative errors. Its original acceptance files are retained as `acceptance-global-normalization-v1.json`; current `acceptance.json` files supersede them. For the exponential map, the full-trajectory discrepancy is 7.70% (peak-only 6.60%) through both t=6 and t=10. The inner-C7 values remain 15.48% and 6.10%. No overall admission changes: all these sequences still fail other spatial criteria.

Two N4097 refinements are running from initial data: inner-C7/physical-KO through t=6 in `telegraph-inner-k005-fine-v1`, and exponential/local-KO through t=10 in `telegraph-exp-fine-v1`. Their numerical source/library bindings were checked against the corresponding completed grid sequences before launch. These runs must finish and be assessed before their results can support any claim. Independent timestep, quadrature, KO and adjacent-parameter controls remain unrun. The goal remains active and incomplete.

## Separating a late coordinate-compression model

The completed exponential-grid extensions were also compared with the leading AF shift model. With beta=b1(t) r+O(r^2) and c_-= -beta+O(r^2), this model predicts log(J_aff)=-integral b1 dt from the ray activation time t=1. At N2049, t=10 gives measured minimum log(J)=-17.5544 and model log(J_aff)=-10.0520. Over t=8..10 the measured decrease is 3.71131, compared with 3.04174 from the endpoint shift model (ratio .81959); N513/1025 give .81978/.81960. Thus substantial late compression is compatible with radial coordinate contraction, while additional compression beyond that model persists.

This is a diagnostic comparison, not an exact decomposition along the actual rays: b1 comes from the endpoint, the minimum seed may change, and its integral uses saved-time trapezoidal quadrature. It cannot prove absence of a continuum shock, replace independent characteristic timestep convergence, or explain away the Hamiltonian failure. The reproducible script and plots are `analyze_telegraph_compression.py` and `evidence/telegraph-affine-compression-v1`.

## Completed N4097 main-duration refinement

The inner-C7/physical-KO sequence N1025/2049/4097 has completed t=6. Bulk and endpoint field orders are 4.29892 and 5.04918. Finest H/M/Theta/Z RMS orders are 4.96691/3.85099/4.98320/4.55175. Finest accepted H RMS is 3.26211e-9; the H point maximum on r<.8 is 1.85851e-4 with order 3.95876. Both H magnitude criteria now pass. The complete-time matched gradient discrepancy is 3.20585%, and the peak-only discrepancy is .669762%. The final gradient maximum is 135.544 at r=.000435721.

This sequence remains rejected: momentum point-error order 3.49377 falls below 3.5, and scri pole 1.05043e-7 exceeds 1e-7. Clock, nullness, and physical-node positivity pass. The required t=10 convergence and independent controls are not established by this t=6 result. The full assessment and polished plots are `evidence/telegraph-inner-k005-fine-v1`; no thresholds were changed.

The exponential N4097 t=10 run remains active. Independent exponential-grid N2049 timestep controls were declared in `telegraph-protocol-exp-time-controls-v1.json` and launched as `telegraph-exp-time-controls-v1`, one worker: CFL=.025 followed by .0125, with baseline .05 from the completed extension. They hold all continuum parameters, map, KO and quadrature fixed. Numerical source/library bindings were verified against the completed N2049 baseline. These are required temporal diagnostics even if the pending spatial refinement fails; they cannot override failed spatial criteria. At this checkpoint the dispatcher/worker PIDs are 64116/64117, alongside the N4097 dispatcher/worker 63566/63586. Revalidate the live launch records before further action. The historical suspended jobs remain suspended, and the goal remains active.
