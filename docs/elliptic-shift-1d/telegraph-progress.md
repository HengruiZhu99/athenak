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
