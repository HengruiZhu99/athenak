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

## Live follow-up at this checkpoint

- `telegraph-fine-front-v1`: N4097, kappa=.001, tau=.02, full initial data through t=3.
- `telegraph-long-v1`: exact-source checkpoint continuations of the three kappa=.001 grids through t=6, one at a time.

Each dispatcher records PID, child PID, command and terminal status in its `launch.json`. Revalidate process handles before deciding whether to wait or restart. These runs are not presumed completed. The t=10 extension, independent dt/shift/KO controls, and parameter-robustness checks remain required. Do not mark the goal complete from the t=3 results.
