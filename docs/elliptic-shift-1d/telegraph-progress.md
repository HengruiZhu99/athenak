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

## Requested constraint histories

`plot_telegraph_constraint_status.py` produced visually checked PNG/PDF histories in `evidence/telegraph-constraint-status-v1`. Full-domain radial RMS norms of H, M, Theta and Z use exactly the evolution weights r^2 dr/dx; these are conformal diagnostics, not proper-volume norms. Every plotted norm and interior H maximum was independently recomputed from saved profiles and checked against history, with frozen source/library checks and profile hashes retained. The first render failed because Matplotlib requires braces around mathcal arguments; this label-only issue was corrected and rendering passed.

N1025 and N2049 cover t=10; the captured N4097 trajectory stops at t=5.1 and is explicitly partial. Sampled H RMS maxima are 5.40030e-5, 1.40528e-6, and 1.52805e-8 over their respective available intervals. These sampled maxima are not the all-step maxima. At t=10, N2049 H RMS is 9.66971e-8 but its pointwise H maximum is .00223801, above the .001 acceptance threshold. The companion full-grid and r<.8 maximum plots expose this late localized error. The Z norm also continues growing late in time. Full-time fine-grid convergence and remaining robustness controls are still pending; no numerical evolution or thresholds changed.

## Constraint localization relative to the apparent horizon

Following the user's question about interior exclusions, `analyze_telegraph_constraint_location.py` reuses the immutable constraint-history capture, verifies its profile and numerical-source hashes, and independently recomputes apparent-horizon positions and masses against the saved history. All original full-domain acceptance diagnostics remain unchanged. The exterior-only maximum is explicitly supplemental. The figure in `evidence/telegraph-constraint-location-v1` was visually inspected.

At N2049 and t=10, the Hamiltonian maximum .00223801 occurs at r=1.46454e-6, whereas the apparent horizon lies at r=.107650308. The exterior maximum is 1.04931e-7. About99.906% of the squared radial H norm lies at r<r_AH, but only23.887% lies at r<.01; the extreme near-puncture pointwise peak and the dominant weighted-norm contribution need not be colocated. The puncture endpoint's analytic diagnostic is zero; nearby nonzero-radius errors are retained. The full-domain H failure is not waived because the exterior is accurate.

The maximum sampled apparent-horizon mass errors are 8.78766e-7 and1.22579e-8 for the two completed grids; the partial fine trajectory through5.1 has maximum1.26652e-10. These different-duration maxima are not a three-grid convergence estimate. Small horizon mass errors cannot override constraint failures. Apparent-horizon extraction does not locate the event horizon.

The previous turn's scientific work was a verified wait on live workers followed by answering the user's diagnostic question. This turn adds new constraint-localization and invariant evidence. Live PIDs70295 and70889 were revalidated, reaching t5.45 and3.65; both remain running. No new evolution workers were started. Full three-grid/time assessment still awaits their completion, and the planned robustness matrices remain undispatched.

## User-requested stopping point and completed three-grid assessment

The user requested wrapping up after the finest grid finishes and updating the existing LaTeX document. No remaining robustness matrices were launched. The prior all-controls assessment waiter was cancelled before analysis; its cancellation receipt is preserved. A stopping-point monitor (PID73329, session24581) waited for the actual finest worker, then ran the three spatial/ray analyses. It completed successfully.

All N1025/2049/4097 main trajectories finished t10. The unchanged spatial gates all pass: bulk/endpoint field orders5.88810/5.62955; finest H/M/Theta/Z RMS orders6.52310/6.48344/6.61712/6.49048; H/M point orders6.61081/6.72154. Finest accepted H RMS1.52811075e-8, H maximum on r<.8 2.28987291e-5, scri pole4.21225385e-9, clock0, null4.44089210e-16, strictly positive physical alpha/chi. Matched lapse-gradient discrepancy is0.103294%, with peak discrepancy0.0359178%. Evidence: `evidence/telegraph-scri-pchip-convergence-v1`.

Characteristic position/logJ spatial orders are4.71856/3.58878; middle-to-fine peak differences6.209e-9/5.507e-6. At t10 all81 inward rays are active, all81 outward rays terminal; minimum fine inward logJ=-17.554446120844084, active adjacent spacing8.32269579e-11, no sampled ordering reversals or active coincidences. This is finite-grid compression, not a continuum noncrossing theorem. Evidence: `evidence/telegraph-scri-pchip-rays-v1`.

The quarter-step worker70889 and dispatcher69515 were suspended at the requested milestone; ps confirms T states. Captured manifest/checkpoint time8.35, step1368064, checkpoint SHA2565ad81949cf334ea8b51a80245ebf6588c6aa144af090bf28dc07e367d75a1e6c. Its original manifest remains untouched and says running; the authoritative external pause receipt is `evidence/telegraph-wrapup-v1/status.json`. Do not treat it as completed or restart/resume without user authorization. Historical suspended PIDs32790,32792,47342,47992 remain untouched. Full independent temporal order, quadrature, KO-amplitude and adjacent-parameter controls remain incomplete.

The existing PRD-style paper was revised in place with a preserved pre-edit copy. It retains all historical failures and adds the spherical conformal telegrapher equations, AF regularity, frozen characteristics/damping, fixed mapped numerics, passive-ray repairs, five portable figures, a numerical table, and the incomplete-control scope. The final23-page PDF has13 figures and4 tables; all pages were visually inspected. No original protected mathematical/citation/reference spans were removed. Numerical claims and generated table macros are tied to verified evidence. The late nonmonotone lapse bump remains explicitly shown. The source bundle and integrity audit distinguish successful spatial evidence from the unachieved robustness objective.


## 2026-10-26: appendix baseline and user-requested long puncture pilot

Moved the original Figure6 to AppendixA (now Figure14), preserving the failed pre-telegrapher evidence, and emphasized the completed three-grid telegrapher study. Started a separate verified N1025 continuation from t10 toward t500 using exact frozen source/binaries. Dispatcher PID75767, evolution PID75769; root output/elliptic-shift-1d/telegraph-long-puncture-500-v1. Other suspended runs remain paused.

At the user's direction, the long pilot's primary budget now excludes r<r_AH(t), normalizes RMS over exterior weights, and prioritizes apparent-horizon mass. Puncture mass is diagnostic only. The amended protocol is telegraph-long-puncture-exterior-budget-v1.json. At t20: exterior H RMS1.79432e-7, exterior Hmax5.87584e-6, and relative horizon-mass error6.42771e-8; the full initial transient is retained. A fixed portable t<=20 snapshot and updated24-page paper preserve this interim evidence. The target500 is not reached at the manuscript snapshot.


## 2026-10-26: terminal unfloored trial and configurable physical lapse floor

The original long run failed at t43.836474609375 (897771 steps); last saved profile43.8. Dispatcher finished is not simulation success. Exterior H/M/Theta/Z RMS at43.8 are1.79432e-7/2.43681e-6/4.99927e-8/5.21150e-7; relative AH mass error1.18005e-7. Frozen-kernel stage replay finds lapse underflow at positive-radius node2, r3.628948e-9, stage4 log(alpha)=-1109.306. The telegrapher logarithmic source dominates the negative rate. Evidence directories: telegraph-long-puncture-terminal-v1, telegraph-lapse-stop-diagnosis-v1, telegraph-lapse-floor-probe-v1.

Implemented input lapse_floor (CLI --lapse-floor), default1e-10 physical alpha/Omega, zero disables. Every RK stage and accepted step is floored at ordinary nodes, with AF coefficient and scri clock preserved. Restart binds the parameter, and historical matrix inputs explicitly disable the new floor. Seven targeted test groups pass. Native regression from an isolated failed-checkpoint copy completes one time unit with the floor, while zero floor reproduces the failure; evidence telegraph-lapse-floor-native-regression-v2 and telegraph-lapse-floor-implementation-v1.

Fresh genuine wormhole N1025 trial targets500 under output/elliptic-shift-1d/telegraph-lapse-floor-long-v1; input inputs/elliptic_shift_1d/telegraph-lapse-floor-long-v1.json, dispatcher80846 and child80847. It remains running at this update (saved t15.7 checked). The existing30-minute heartbeat now monitors this trial with the exterior/AH budget and preserves the old failed run. Other paused trajectories remain paused. No long-time-stability claim yet. The updated25-page manuscript has15 figures and4 tables, includes terminal exterior diagnostics and lapse-underflow diagnosis, and retains the pre-telegrapher baseline in AppendixA.


## 2026-10-26: floored trial terminal assessment and repair authorization

The fresh floored run failed at84.798291015625,1736669 steps, runtime1528.921s. Last saved84.75;1696 profiles. Both dispatcher80846 and worker80847 are absent. Final exterior H/M/Theta/Z RMS1.79432247e-7/2.43681080e-6/4.99926502e-8/5.21149969e-7; relative AH error2.10449126e-7, initial-transient peak2.63629888e-6. Outer floor radius6.29212027e-7 remains far insideAH0.10765008. Saved scri clock/null0, pole4.59077e-13.

Frozen-kernel replay reproduces coupled-speed guard at stage1,node3,r5.53026834e-9: beta3.93017191e7, mapped Courant9.93706483e11, dt/tau0.00244140625. Lapse is at physicalfloor1e-10, chi7.71983747e-14. Floor maximum log correction rises from5246.372 at last profile to11950.571 terminally. This is a distinct later inner failure, not repair to t500. Evidence: telegraph-lapse-floor-terminal-v1 and telegraph-lapse-floor-stop-replay-v2. The26-page paper now contains16 figures and4 tables, with separate floored evidence and preserved unfloored data.

During assessment the user explicitly requested fixing this long-time puncture failure and checking conformal weighting against the earlier Z4c implementation. This authorizes new diagnosis and controlled repair trials; paused quarter-step/historical runs remain untouched. The currently implemented lapse correction already carries chi, while the flux source has no extra chi. The af-regular window is1 near the puncture. The earlier origin/project/telegrapher_lapse branch similarly uses inverse conformal factor oopsi4 in the lapse's contracted flux derivative. Numerical repair work is ongoing separately from the terminal assessment.


## 2026-10-26: ongoing repair controls after user request

Old branch origin/project/telegrapher_lapse uses oopsi4=pow(chi_guarded,-4/chi_psi_power) in the contracted flux derivative, with no extra chi in the flux relaxation equation. The current spherical correction already carries chi; af-regular w=1 at the puncture. No missing leading chi factor was found.

The failed floor checkpoint has a neighboring node4 conformal lapse7.59e12, while the speed-guard node3 remains floored. Positive logarithmic RK kicks can amplify additive flux sources exponentially. Added optional lapse_update=total: integrate conformal alpha directly at positive-radius RK stages, apply physicalfloor, store logarithm afterward; retain AF/scri variable treatment and account for reconstruction roundoff in physical compensation. This does not change continuum gauge equations. Independent RK oracle/restart tests pass. An isolated t84.75 profile probe (zero compensation, not exact restart) fails at84.88222656; a badly degraded state is not recovered. Fresh total-update-only pilots fail at17.27578125 (N257) and42.60048828125 (N513).

Added optional puncture_stencil=positive: one-sided bulk derivative/advection stencils exclude the r0 asymptotic-coefficient slot; every positive-radius node remains evolved. Coefficient endpoint remains an independent log diagnostic, with overflow-safe reconstruction that does not exponentiate its unused coefficients. Endpoint isolation and polynomial/native-Python tests pass. This closure plus total-lapse RK delays the N257 failure to32.90703125, but is not sufficient. All failed cases remain under their original output roots.

Current candidate adds inner_total_derivatives_radius=.001. Inside half this radius, derivatives of reconstructed chi, alpha, A_r, Khat and Theta are computed directly instead of differentiating the singular reference-reduced chart. A quintic blend recovers the historical chart derivatives by the stated radius. Corresponding upwind transport corrections use the same variables. This is a derivative discretization choice, not constraint projection or a continuum gauge change. It retains chi weighting and the unchanged elliptic solve; no lapse upper cap or shift limiter is used. Fresh N257/N513 trials target500 under output/elliptic-shift-1d/telegraph-total-derivatives-pilot-v1. Current investigation has not established a repair. Paused historical controls remain paused.


## 2026-10-26: normalized lapse attenuation duration candidate

The direct-total-derivative control failed early (N257t4.74140625,N513t5.67431640625) and is disabled in the current candidate. A disclosed gauge modification multiplies the existing chi-weighted lapse flux by s=(1+epsilon)*alpha/(alpha+epsilon*A), A=(1+r^2)/2. This preserves s=1 on the reference lapse and bounds the extra logarithmic source coefficient as alpha approaches zero. Flux relaxation remains unchanged; coupled speed is updated consistently to sqrt(chi*(F+s*kappa*w^2/tau)).

N257 exploratory attenuation runs failed at57.1580078125 (epsilon1e-4) and73.5423828125 (epsilon1e-3), with already unacceptable exterior error and approximately7% AH mass error by31.25. They are negative evidence. All failed controls are archived in evidence/telegraph-regularity-repair-controls-v1.

Active fresh N513/N1025 trials use epsilon1e-4, total-lapse RK, positive-node stencils, floor1e-10, and the original chart derivatives (inner_total_derivatives_radius0). Root output/elliptic-shift-1d/telegraph-attenuated-duration-v1; dispatcher89526, workers89528/89529. Both target500. At the latest recorded check N513 reached90.25 and N1025 reached23.05, with bounded stage Courant and only tiny floor corrections. Nine targeted test groups pass; receipt evidence/telegraph-regularity-repair-tests-v1. Input and predeclared primary N1025 acceptance limits are recorded in telegraph-attenuated-duration-protocol-v1.json. This remains an experimental repair, not a completed stability result. The existing30-minute monitor now tracks these exact trials and the updated repair scope. Do not resume paused historical controls or duplicate active runs.


## 2026-10-27 UTC: duration gate and fresh monitoring

Verified process commands and frozen numerical bindings for workers89528/89529. Latest checked saved times: N513t270.75, N1025t69.0. N1025 relative AH error1.75112e-7, exterior RMS(H,M,Theta,Z)=(1.79431e-7,2.43679e-6,4.99923e-8,5.21146e-7). N513 remains finite but its relative AH drift5.83992e-4 is substantially larger; it is an exploratory duration control, not an accuracy pass. Current monitor/latest.json retains full process identities, profile hashes, floor activation and boundary data. Both trials continue unchanged toward500.

Added a predeclared-budget assessor and seven passing gate tests. Independently validated the entire N1025 saved trajectory through60; budget passes at that cutoff, duration verdict pending. Fixed a collector manifest/hash snapshot race exposed by the assessor and retained the rejected export under tmp. Current evidence: evidence/telegraph-attenuated-interim-t60-v2/n1025. Main paper remains the terminal floor-only evidence snapshot until the new duration trials have an assessed outcome.


## 2026-10-27 UTC: repaired trial passes the historical floor-only stop time

The current N1025 trial passed the floor-only failure time84.798291 without changing parameters or relaxing a guard. At the independently checked saved state t86.3, relative AH mass error is2.14166e-7, exterior(H,M,Theta,Z) RMS=(1.79431e-7,2.43679e-6,4.99923e-8,5.21146e-7), and the maximum recorded mappedCFL remains0.200000000008. The sampled physical lapse over0<r<=1e-4 spans4.61814e-7 to1.79677e-3. N513 is still live at338.5, with relativeAHdrift7.33719e-4; its survival is not promoted to the primary accuracy result. Actual worker identities and frozen bindings were checked. The monitoring receipt is evidence/telegraph-attenuated-interim-t60-v2/past-old-failure-monitor.json; its time is explicitly86.3 and is separate from the t60 full-trajectory export. Target500 and its terminal assessment remain pending.


## 2026-10-27 UTC: interim repair manuscript and portable package

Updated the paper with the explicitly modified attenuation gauge, total-lapse stages, positive-radius stencils, seven failed fresh controls, and a fixed validated N1025 export through90. All1801 saved states pass the declared exterior budget to that cutoff; final relativeAH error2.22518e-7. Captured stage extrema through92.65 are labeled separately. Duration verdict remains pending. The new Figure16 shows raw lapse ripples and exterior diagnostics; the historical pre-telegrapher baseline is now Figure17 in AppendixA. No convergence verdict is transferred from the old unattenuated three-grid study to the modified candidate.

Final PDF has29 pages,17 figures,5 tables. Source/claim audit and full rendering completed; no undefined references or overfull boxes. REVTeX deferred-float warnings remain with all floats present and inspected. The181-file source package passes CRC and all SHA checks; a clean extracted build reproduces29 pages, identical PNG output, and the pending t90 assessment. No numerical parameters or running-state source bindings changed. Paused historical experiments remain paused.


## 2026-10-27 UTC: N513 reaches500; principal N1025 still running

N513 completed500 normally,5,120,000 steps,2488.08s. All10,001 saved states validated under warnings-as-errors. Final/max relativeAHerror0.00109005921; late exterior constraints nearly steady. This is exploratory duration survival with0.109% mass drift, not principal accuracy acceptance. Evidence and raw-profile plots: evidence/telegraph-attenuated-n513-terminal-v1. Actual worker89528 exited; fine worker89529 is verified live at134.15, relativeAHerror3.22185e-7. Current monitor records terminal coarse assessment and continuing fine trial. No restart or parameter change.

Collector now sorts numeric report indices through10001 and masks the AF coefficient before exponentiation for positive-radius minima. These analysis-only changes prevent late-export failures without altering any physical sample. Paper PDF remains the validated t90 snapshot; its source package includes the updated collector/plot helper. The final primary result will be incorporated when N1025 terminates.

## 2026-10-27 UTC: full saved-history checks through400

The same N1025 worker89529 is verified live at408.6. Independent collector exports through250,300,350,and400 pass every declared accuracy limit and saved-profile completeness check. The t400 export contains8001 states; its initial-transient maximum relativeAH error remains2.64272e-6. Late exterior RMS maxima remain(H,M,Theta,Z)=(1.79450e-7,2.44246e-6,4.99959e-8,5.21288e-7). Target500 is still pending. No source or parameter change, restart, or duplicate evolution was made.

A read-only floor-counter audit distinguishes the initial physical-lapse clamp from later total-RK clipping. Initial flooring changes80 nodes with maximum conformal-lapse change4.99996e-11; the later absolute-correction counter5.73769e-15 excludes this initial operation. See evidence/telegraph-attenuated-floor-accounting-v1 and tst/elliptic_shift_1d/review_telegraph_floor.py. This reporting clarification will be included in the terminal manuscript revision.
