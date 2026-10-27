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


## Independent temporal-analysis verification

`analyze_telegraph_time.py` now verifies the frozen source hashes, recursive numerical Python import closure, both compiled libraries, completed checkpoint ancestry and complete exact saved-time coverage before comparing dt/dt2/dt4. It reports individual fields and aggregate bulk/endpoint differences, full-domain H/M/Theta/Z differences, physical-lapse gradient differences, auxiliary flux, ray position/log(J), and invariant mass. It retains raw timestep-to-spatial differences and observed orders rather than presenting them as certified error bounds. A pending or failed spatial result cannot be replaced by a successful timestep comparison.

Five analysis tests passed in the newly registered CTest 53. They include an independent analytic fourth-order error sequence, a deliberately failed time-to-space ratio, changed grid/mass-support rejection, changed inputs/source/coverage rejection, and an exact cubic derivative check for the high-precision mass audit. The completed baseline's 201 report times and frozen numerical dependencies were verified; the running half-step control has matching numerical dependencies and is correctly rejected by the complete-run analyzer until it finishes. Evidence is `evidence/telegraph-time-analysis-validation-v1`. The previous 52 solver tests remain as previously recorded; this analysis-only change did not rerun the full evolution suite.

## Near-scri invariant-mass arithmetic audit

At t=.2, `audit_telegraph_mass.py` recomputes mass from the saved N1025/2049/4097 states using exact rational seven-node derivative weights and 60/80-digit Decimal arithmetic. Independently evaluated geometric and canceled mass formulas agree, and increasing precision from 60 to 80 digits changes the results by less than 1e-40. Across the sampled last four physical nodes, differences from the saved binary64 mass diagnostics are below 8.4e-10, much smaller than the observed mass errors. Therefore arithmetic in the extraction formula does not explain those errors. This audit does not isolate roundoff already accumulated during evolution from truncation error.

At the shared physical radius r=.9990234375, absolute mass errors fall from .164392 to .00787427 to .000278132. The last physical node moves closer to scri under refinement and remains less accurate: N4097 has error .00522034 at r=.999755859375. The mass formula contains an Omega^-3 factor, so small boundary-state errors can be strongly amplified. This finding motivates continued scrutiny of endpoint discretization and convergence; it does not justify dropping those points. Evidence: `evidence/telegraph-scri-mass-arithmetic-v1`.

## Constraint-extraction stencil audit

Re-differentiating the identical completed states with 7/9/11-point stencils leaves the late H peak essentially unchanged. For the exponential N2049 state at t=10, maxima are .023119780057/.023119779629/.023119779656 at the same r=1.58811e-6. For inner-C7 N4097 at t=6 they are .000185850876/.000185851649/.000185851708. Original seven-point diagnostics reproduce bitwise, including all diagnostic fields.

The first combined audit correctly rejected a differing `radial_grid.py` snapshot: the earlier C7 run predates the exponential-map addition. The reproducible auditor now uses a separate interpreter with each case's exact frozen numerical imports and libraries, rather than assuming current source equivalence. It verifies frozen bindings and loaded module hashes. This preserves the different numerical histories while allowing independent re-differentiation of each saved state.

The higher-order extraction does not cure the observed late Hamiltonian error. This supports treating it as error already present in the evolved state, without claiming that this audit distinguishes spatial evolution error from temporal error. All original accepted-state maxima and gates remain unchanged. Evidence: `evidence/telegraph-constraint-extraction-v1` and `audit_telegraph_constraints.py`.

## Higher-order numerical pilot: useful and failed changes

The new optional width9 differentiation and KO8 dissipation are documented in `telegraph-formulation.md`. They change numerical operators only. KO8 has a verified mapped energy identity and isolated dissipative spectral bound, with seventh-order interior truncation. The AF coefficient column and actual-scri equations remain in place. All 53 tests passed after the implementation changes. The old width7/KO6 solver also reproduces the complete prior N513 exponential trajectory bitwise at all 121 saved times through t=6 and in the final compensated checkpoint. Evidence: `evidence/telegraph-high-order-implementation-v1`.

The initial high-width implementation checks exposed an ill-conditioned N33/b16 physical-r transition stencil: Python/compiled rate differences exceeded the existing absolute tolerance, and the corresponding evolution hit the coupled-speed CFL guard after one step. These failures are retained. High-width b16 cross-implementation tests use N129, where the map transition is resolved enough for the original tolerance; other map/width combinations still include N33/65. The N33 strong-map run is now an explicit negative guard test. No timestep guard or nonlinear acceptance criterion was relaxed.

The declared N513 factorial pilot completed all four cases through t=6. Width7/KO6 gives H RMS .00125074, H point maximum on r<.8 of .232120, and scri pole .000143414. Width7/KO8 improves these to .000588774 and .155049 with the same scri pole. This is improvement, not acceptance.

Width9 is rejected with the current scri treatment. With either KO order it develops a scri pole residual around 5590 and H RMS around .203 despite reaching t=6. The saved Hamiltonian maximum moves to scri around t=3. Completion without a process failure is not numerical success. The complete factorial histories, stage extrema, plots and manifest bindings are in `evidence/telegraph-high-order-pilot-v1`.

The selected numerical follow-up retains width7 and tests KO8 at N513/1025/2049 through t=10, with unchanged continuum relaxation parameters. The N513 case is being continued from its verified t=6 checkpoint under `telegraph-ko8-extension-v1`; the two finer initial-data runs are declared in `telegraph-ko8-convergence-v1.json` and `telegraph-protocol-ko8-convergence-v1.json`. They use one worker after the short coarse continuation. The existing KO6 N4097 run and its independent timestep controls continue from their own frozen snapshots. No candidate has passed the complete robustness protocol.

## Completed first temporal control and outer-resolution preparation

The N2049 KO6 dt/2 run completed t=10. Its maximum differences from the baseline over all 201 exact report times are 4.03948e-12 in aggregate bulk fields, 2.33236e-12 in endpoint fields, 5.59519e-10 in pointwise H, and 2.77578e-9 in the physical lapse gradient. The ray position difference is 6.36774e-11 and log(J) difference is 3.31217e-5. The latter is appreciably more sensitive than the fields and still requires the dt/4 check. Source/library bindings and complete ancestry were verified. This two-level comparison does not establish temporal order or replace the required time-to-space error assessment. Evidence and its reproducible analysis command are in `evidence/telegraph-exp-half-time-v1`.

The KO8 coarse continuation completed t=10, with H point maximum .831774. The independently launched N1025 run also completed t=10 with H point maximum .0812050, compared with .277343 for KO6 at that resolution. N2049 KO8 is still running. These are early refinement results, not a completed three-grid assessment.

A separate outer-resolution map `inner-exp-scri` is implemented and passes the three targeted telegraph/analysis tests. It retains the inner nodes and width7 while halving the asymptotic spacing at scri; its positive Jacobian and C7 join are derived in the formulation note. The earlier main-run scri maximum is reached around t=.2, so a paired short fine-grid test can isolate this issue. `telegraph-scri-map-pilot-v1.json` and its protocol are prepared before launch: paired maps at N513 through6 and N1025/2049 through .5. This pilot is not yet launched; it waits for an existing worker slot. The active KO6 N4097, KO6 dt/4 and KO8 N2049 runs retain their original frozen numerical sources. All goal criteria remain unchanged and incomplete.

## Completed finest KO6 extension

The N1025/2049/4097 exponential KO6 sequence has completed t=10. Bulk/endpoint field orders are 4.32812/4.97990. Finest H/M/Theta/Z RMS orders are 4.74464/4.63544/4.87912/4.85319, and H/M point orders are 4.65861/4.70341. The finest H RMS is 3.79034e-7 and point maximum is .000915382, both below the unchanged thresholds. The complete-time matched gradient disagreement is .869641% (peak-only .575356%). All these spatial checks pass through the extension duration.

The scri pole residual remains 1.0504318e-7, above 1e-7. Thus this candidate still fails the spatial audit despite its now-convergent lapse front and Hamiltonian errors. Independent controls and adjacent-parameter robustness also remain incomplete. Evidence: `evidence/telegraph-exp-fine-v1`.

## Completed KO8 refinement and scri-resolution pilot

KO8 on N513/1025/2049 has also completed t=10. Bulk field order is 5.14486, H/M RMS orders are 5.26378/4.92365, and H/M point orders are 5.18128/5.24459. Finest H RMS is 1.40515e-6, but H point maximum .00223801 and scri pole 1.87977e-6 fail. Matched gradient disagreement is 2.99938%. The stronger spatial order and reduced H error motivate finer testing; they do not admit this three-grid sequence. Evidence: `evidence/telegraph-ko8-convergence-v1`.

The paired outer-grid pilot is complete. The scri pole maxima for the original map versus `inner-exp-scri` are .000143414/.0000334697 at N513 through6, .0000206120/.00000283285 at N1025 through.5, and .00000187977/.000000141437 at N2049 through.5. The refined map has no observed delayed coarse boundary failure through6, and its inner H point maximum is unchanged to about 1e-12. Fine runs stop at .5 and cannot establish late stability or goal completion. The next main candidate retains width7, KO8, and this refined outer map, pending the ray-interpolation accuracy work below. Evidence: `evidence/telegraph-scri-map-pilot-v1`.

## Restricted boundary-spectrum diagnosis

A frozen outer-block linearization around analytic wormhole initial data, restricted to r>=.9 with fixed lapse-clock variation, has a growing width9 mode: largest real eigenvalues are approximately .970/2.636/5.567 for N129/257/513. Halving the finite-difference perturbation amplitude preserves this sign (N513 changes by about .004). Width7 has negative largest real parts in both the original and refined-scri maps. This supports the observed width9 boundary failure and the choice to retain width7.

This is not a full-system stability proof. Extending the restricted eigenvector by zero into the inner domain leaves a relative full-operator residual about .146 at N513 for the growing mode; interior feedback is not negligible. Nonnormality and nonlinear effects are also outside this eigenvalue test. The full recorded residuals, spectra comparison and mode profiles are in `evidence/telegraph-boundary-spectrum-v1`, with `audit_telegraph_boundary_spectrum.py` as the reproducible script.

## Characteristic temporal accuracy: a remaining failure

With the completed N4097 reference, the N2049 dt/2-to-spatial difference ratios are 5.56e-8 for bulk fields, .000108 for endpoint fields, 2.52e-8 for H, and 1.12e-5 for ray positions. However, the log(J) ratio is .224876: a temporal difference 3.31217e-5 versus spatial difference .000147289. This fails the 10% comparison target. The geometry's small time error does not certify ray-map time accuracy. Evidence: `evidence/telegraph-exp-time-space-v1`. The original linear-interpolation dt/4 run is still active and should not be restarted.

The current characteristic velocity is continuous piecewise linear, but its derivative jumps at cell crossings. An optional monotone C1 cubic Hermite interpolant, with log(J) driven by the derivative of the same interpolant, is the proposed next numerical diagnostic control. Its pre-implementation protocol is `telegraph-protocol-ray-interpolation-v1.json`. It must preserve bitwise geometry/flux evolution and default linear behavior, pass independent interpolation/RK/restart checks, and then demonstrate temporal accuracy. It is not yet implemented or accepted. SciPy is unavailable in both system and bundled Python; an independent oracle can be installed in an isolated test location or implemented and checked against analytic cases. The official PCHIP reference is linked in the protocol. No change to the Einstein or telegrapher equations is proposed by this diagnostic repair.

## C1 ray interpolator implementation and controls

The optional PCHIP diagnostic is implemented, retaining linear mode as the default and control. All 54 CTests passed. Independent SciPy 1.13.1 checks cover values and derivatives on uniform, sinh and strongest refined-scri grids, including monotone data, extrema, plateaus, random data and near-node samples. Actual nonzero-flux RK-stage ray evolution agrees with the Python/SciPy oracle. Geometry, B and their compensation are bitwise identical between modes. Matched checkpoints restart exactly, and the default linear mode reproduces the previous frozen library bitwise for 256 steps on N129.

Two test-development failures are retained in `evidence/telegraph-ray-implementation-v1`: an invalid synthetic terminal-state test retained compensation from unrelated interior rays (fixed by matched fresh state), and a derivative oracle tolerance ignored cancellation of order-1e8 secants on tiny mapped cells. The latter now uses an explicit machine-epsilon conditioning term for derivative evaluation, with separate value checks. Evolution and scientific acceptance thresholds are unchanged.

The declared local pilot runs both interpolation modes at N513 through t=3 with CFL .05/.025/.0125, plus PCHIP N1025 at .05. It uses width7/KO8, the refined-scri exponential map and unchanged kappa=.005/tau=.02. This tests diagnostic timestep sensitivity; it is not a substitute for t10 three-grid and independent-control validation. Dispatcher PID67875 / session45966 preserves source and library snapshots. The original KO6 dt-quarter PID65897 remains active from its own frozen snapshot.

## Completed interpolation pilot: active-ray improvement, terminal-exit failure

All seven declared t3 runs finished. Complete baseline snapshots have byte-identical geometry, B, shift, diagnostics (including the deliberately undefined scri mass slot), and sampled gauge speeds between linear and PCHIP modes. The final geometry/flux compensation is also bitwise identical at all three timesteps. Independent numerical bindings, exact report times and source/library hashes are verified by the temporal analyzer.

For all rays, the baseline-to-quarter-step maximum log(J) difference drops only from .000205182 to .000149440; PCHIP's raw time-to-space ratio is .253519 and fails the unchanged .1 target. Successive timestep differences give observed log(J) orders .314 for linear and .916 for PCHIP. Therefore the PCHIP-only pilot is not admitted, and the prepared t10 refined-scri/PCHIP matrix remains undispatched.

Separating error by ray status diagnoses the remaining failure, without changing acceptance. For rays active in both compared runs, the maximum difference falls from 9.87506e-5 to 9.88614e-7, about a factor100. The all-ray maxima belong to rays that have already reached r=1: PCHIP's largest error is plus-family ray107 at t=1.7, with both positions exactly1. There are no differing exit statuses at the saved times. The current RK stages stop rays outside the domain and clamp accepted positions, but do not locate the exit within the timestep; the accumulated log(J) therefore retains a first-order event-time error. This explains why smooth interpolation helps active trajectories without fixing terminal records.

Evidence is `evidence/telegraph-ray-interpolation-v2`, produced by `analyze_telegraph_ray_pilot.py`. The first analysis attempt (`...-v1`) retains a NaN-equality assertion failure; bytewise comparison verifies the arrays were identical, including the undefined scri mass value. The completed result contains all-ray, active, exited and status-mismatch histories rather than discarding terminal errors.

The next declared diagnostic repair is `telegraph-protocol-ray-exit-v1.json`: optional continuous-RK exit localization and evaluation of log(J) at the same event time, with matched terminal freezing. It requires independent exact-flow, stage, bitwise no-feedback and restart tests before repeating the t3 pilot. It is not implemented yet. This error is in passive ray tracking and cannot account for the Hamiltonian errors in the evolved geometry. The original KO6 quarter-step control (PID65897, dispatcher64116/session59457) continues uninterrupted; last observed t=8.4. No full robustness claim is made.

## Dense ray exits implemented and verified

The previous goal turn was concrete progress: it isolated terminal-event errors after smooth interpolation had reduced active-ray timestep sensitivity. The optional `--ray-exit dense` repair is now implemented according to the declared protocol. It follows the first boundary event in the classical RK4 cubic continuous extension and evaluates log(J) at that same fraction of the step. Endpoint-polynomial continuation is bounded to one boundary cell, and all already-terminal ray states remain frozen. The full eight-field geometry and B update are unchanged.

All 55 CTests passed. Exact affine and time-dependent affine flows, both boundaries and seventeen initial positions per direction, give exit-time and terminal-logJ orders between 3.81 and 4.19 across dt=.08/.04/.02/.01. Finest errors are below 9.1e-11. Tests also verify continuous third-order conditions, first exits followed by return, independent coupled Python/SciPy RK stages with real scri exits, bitwise geometry/flux agreement, exact freezing, and restart with mixed terminal/active rays. Legacy modes (linear and PCHIP) match their previous frozen library bitwise over 256 steps including actual terminal exits. Changing the exit flag across a CLI restart is rejected. Evidence: `evidence/telegraph-exit-implementation-v1`.

The repeated seven-case t3 pilot is active under `telegraph-ray-exit-v1`, dispatcher 69081/session 2107. Both interpolation modes now use dense exits, with identical grids, continuum parameters, KO and timesteps to the preceding pilot. No long candidate run has been dispatched yet.

## Completed independent KO6 t10 timestep sequence

The original N2049 quarter-step run finished t=10 without interruption (2658.57 seconds). All dt/dt2/dt4 histories and the N4097 spatial reference have verified frozen numerical bindings and complete report coverage. Bulk and endpoint field temporal orders are 4.00012 and 4.00026. Baseline-to-quarter differences are 4.29191e-12 and 2.47807e-12, corresponding to time-to-space fractions 5.9123e-8 and .000114963. Pointwise H differs by 5.71585e-10 versus spatial difference .0222252; its timestep differences have reached a nonmonotonic numerical floor and do not demonstrate an order, but are much smaller than spatial error.

Every measured raw time-to-space ratio passes .1 except the legacy log(J) diagnostic: .254417, with baseline-to-quarter difference 3.74728e-5 and spatial difference .000147289. Its observed successive-difference order is .617771. Thus the completed control confirms that evolving-field timestep errors are negligible at the tested settings, while the old characteristic diagnostic remains inaccurate. It does not admit the candidate or excuse its failed scri gate. Full evidence: `evidence/telegraph-exp-time-controls-v1/review.json`.

## Exit-aware pilot passed; full-duration combined candidate dispatched

The repeated t3 pilot completed all seven runs. PCHIP plus dense exits gives baseline-to-quarter all-ray log(J) difference 9.88614e-7, versus spatial difference .000589463: ratio .00167714 (0.1677%), below the unchanged 10% threshold. Its observed successive timestep-difference order is 2.29611. Every measured time-to-space ratio in this pilot passes. Linear interpolation with dense exits remains substantially less accurate (all-ray difference .000101918, observed order .611881), confirming that both derivative continuity and exit localization matter.

The new all-ray PCHIP difference is about 151 times smaller than with legacy exits. The active-ray difference is unchanged, and the exited-ray difference drops to 9.86900e-7. All 427 saved profiles across the seven paired old/new trajectories have byte-identical geometry, auxiliary flux, shift, diagnostics and sampled speeds. Frozen bindings are independently verified. Thus this repair changes passive ray integration only. Evidence and visually checked plots are in `evidence/telegraph-ray-exit-v1`.

The combined candidate uses width7/KO8, inner-exp-scri b16, PCHIP and dense exits, with kappa=.005,tau=.02,nu1,M=1/3,AF-regular coupling and transition .24. The N1025/2049/4097 t10 matrix is running under dispatcher 69510/session 85050 (first child N1025 PID69512). The independent N2049 half/quarter-step sequence is running under dispatcher 69515/session 31608 (first child half-step PID 69531). Both have frozen source/library snapshots and one worker each. Revalidate the actual child PIDs before observing or continuing; dispatcher launches later cases sequentially.

Original KO6 dt-quarter PID65897 is finished, not paused or missing work. Historical suspended jobs remain untouched. No candidate is admitted yet: the t10 combined spatial/boundary/ray/time results are pending. The independent quadrature, KO-amplitude and adjacent fixed kappa/tau controls are declared in `telegraph-protocol-scri-pchip-robustness-v1.json` and three input matrices, but remain undispatched until the main results are assessed. They must use the main frozen numerical snapshot. All original acceptance thresholds are preserved.

## Independent-control and characteristic evidence verification

The previous goal turn made implementation and experimental progress. While the combined candidate's long runs continue, `analyze_telegraph_control.py` now verifies the pending quadrature/KO comparisons against a fixed spatial reference. It requires complete trajectories, matching frozen numerical dependencies and executables, exact report times, and only the declared numerical parameter to vary. Quadrature uses the predeclared raw difference threshold; a KO difference report cannot admit a candidate without its separate three-grid study. Zero spatial denominators and failed ratios remain explicit.

A shared profile validator now rejects nonfinite fields everywhere, including the radial region between the bulk and endpoint field-norm masks. Only the deliberately undefined actual-scri mass entry is allowed to be nonfinite. This check is used by temporal, spatial and numerical-control analysis. All 61 exact report-time difference metrics from the completed dense-exit pilot reproduce unchanged. Five existing temporal-analysis tests and six new control/ray-analysis tests pass in two targeted CTests; the evolution kernel is unchanged and its previous 55-test validation remains recorded. New tests include altered on-disk source/executable rejection, missing reports, running status, uncontrolled physical parameters, false-pass examples, zero denominators and nonfinite unplotted fields. Evidence: `evidence/telegraph-control-analysis-validation-v1`.

`analyze_telegraph_ray_convergence.py` adds verified, separately labeled inward/outward ray fans, active-neighbor spacing, terminal counts, all-ray spatial differences and (when three grids are available) observed orders. Boundary coincidences are distinguished from active coincidences, and order reversal is rejected as unresolved integration. Its independent synthetic error-scaling test recovers second order and does not admit any PDE result. Neither positive exp(logJ) nor finite-grid ordering is presented as a continuum crossing theorem.

## Combined candidate: two completed grids and half-step control

N1025 and N2049 have completed t=10. Their accepted H RMS maxima are 5.4005570e-5 and 1.4054072e-6; H maxima on r<.8 are .0812050 and .00223801. Scri pole maxima are 2.8328521e-6 and 1.4143655e-7. Clock error is zero, nullness is at floating-point roundoff, and all-stage positivity checks pass. N2049 still exceeds the final H-point and scri-pole magnitude thresholds; the third grid is required before any spatial admission.

Over these two completed trajectories, the maximum all-ray position difference is 1.63474e-7 and the maximum log(J) difference is 6.62559e-5. At t=10 all 81 inward rays remain active and all 81 outgoing rays are terminal. The minimum inward log(J) values are -17.5545021/-17.5544475; minimum adjacent inward spacings are 8.32223e-11/8.32268e-11, with no active coincidences or ordering reversals. This is strong finite-grid compression, not evidence of a resolved characteristic crossing. The visually checked paired plots and complete histories are in `evidence/telegraph-combined-two-grid-rays-v1`; the earlier single-grid report is retained separately.

The N2049 half-step control also completed t=10. Compared with its baseline, the maximum bulk/endpoint field differences are 4.03948e-12/2.33087e-12, H point difference 5.71069e-10, lapse-gradient difference 3.25031e-9, ray-position difference 4.27325e-13 and log(J) difference 7.68000e-8. Source/library bindings and all 201 report times are verified. This two-level result does not establish temporal order or the required middle-to-fine time/space ratio. Evidence and reproducible script: `evidence/telegraph-combined-half-time-v1`.

Current workers were revalidated with ps: finest N4097 PID 70295 under dispatcher 69510/session 85050, and quarter-step N2049 PID 70889 under dispatcher 69515/session 31608. Last observed times were 2.1 and .25, respectively. The half-step child PID 69531 is finished. Do not restart these trajectories on observation timeouts. All remaining quadrature, KO and adjacent-parameter matrices are still undispatched, as declared; no acceptance criterion has changed.

## Requested lapse status plots

The user requested a current status update and actual lapse profiles during the active study. `plot_telegraph_lapse_status.py` now produces global conformal-lapse evolution, physical-lapse evolution on a logarithmic inner radial axis, and raw resolution overlays with difference panels. It checks frozen numerical sources/libraries and matching configurations, reconstructs alpha independently from the stored AF logarithmic state, and reads only immutable selected snapshots at or before each manifest's captured progress. It never treats the partial fine trajectory as a completed convergence run.

The plot capture has N1025/N2049 completed through 10 and N4097 available through 4.4. At t=2.5, the maximum physical-lapse differences in the displayed .03<=r<=.22 window are 3.17133e-5 (1025–2049) and 3.49524e-7 (2049–4097). At t=10, the two completed grids differ by 3.64080e-6 over 1e-8<=r<=3e-5. A definite nonmonotone inner bump remains, with peak physical lapse about .012 near r=6e-7; it is shown explicitly. Close agreement of two profiles does not certify the late front, its derivatives, or the Hamiltonian constraint. The fine t10 result is still pending.

Both PNG/PDF figures were visually inspected. The raw profile hashes and captured manifests are preserved in `evidence/telegraph-lapse-status-v1`. The global figure uses conformal alpha, which remains finite at scri; the inner plots use physical alpha/Omega. No profiles were smoothed or temporally interpolated, and no acceptance threshold changed.
