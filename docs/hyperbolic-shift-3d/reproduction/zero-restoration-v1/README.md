# Candidate A without the extra singular shift restoration

User clarification: use the ORIGINAL native A initial slice (C-infinity, 4M–8M), not the separately studied polynomial 4M–12M slice. Fresh N16 dt=.0005 and N24 dt=1/3000 to coordinate time10/3 are authorized after boundary audits and native GPU checks. Branch codex/zero-restoration-puncture preserves A and the completed wider-collar trial. No old state may be restarted in the new formulation. No automatic science/timestep/floor changes.

The only equation change removes -4 W C (beta+x)/Omega. W, direct-outer mu, auxiliary D, lapse restoration, inner damping, constraint damping, KO8 and original scri operators remain. Distinct native targets control_zero_restoration use checkpoint magic HYP3D_MPI_NZ1, existing168-byte/scheme3 layout. Geometry and lapse parameters named restoring in tensor_scri are unrelated and are preserved. Numerical source and binaries must be frozen per experiment.

## Limiting equation and scope of compatibility

Let L=dt-beta^j partial_j, v^i=beta^i+x^i and W=1 on the outer plateau. Since L x^i=-beta^i, the retained equation gives L v^i=-v^i+mu D^i, mu=(3/2)chi(alpha^2+4 Omega^2 alpha). This retains finite relaxation; it removes only the extra singular source. In the blended interior the original inner shift equation remains weighted by1-W.

At scri, impose alpha=1 and beta^i=-s x^i, s=sqrt(chi q), q=gtilde^{ij}x_i x_j, at fixed coordinates. Define n^i=gtilde^{ij}x_j. Using the ACTUAL geometric RHS G (including penalties and angular dissipation), qdot=-n^i G_gij n^j and sdot=(q G_chi+chi qdot)/(2s). The required shift rate is T^i=-sdot x^i. Thus

    D^i = (T^i - beta^j partial_j beta^i + x^i) / ((3/2)chi).

All three components are dependent because all three shift components are prescribed by nullness plus zero coordinate-tangential shift. The old +4 x^j partial_j(beta^i+x^i) restoring-limit contribution is absent in both CPU and GPU closure and their compatibility diagnostics. Taking a regular limit of the removed source would be inconsistent: it is identically absent in the interior.

This algebraic construction ensures the shift equation and differentiated boundary gauge agree for the supplied geometry rate and shift jets. It does NOT by itself imply compatibility with the auxiliary PDE. Let E^i=D^i-Lambda^i. The unchanged equation is L E^i=-D^i. A smooth solution must also satisfy the differential trace condition

    dt D^i - T_Lambda^i - beta^j partial_j(D^i-Lambda^i) + D^i = 0.

Here dt D is the derivative of the algebraic trace along the full solution, not an independently adjustable boundary rate. Arbitrary manufactured spatial data need not satisfy this time-compatibility condition. Tests must distinguish discrete operator consistency on regular data from existence of a solution satisfying all trace conditions. No well-posedness claim follows from roundoff nullness, CPU/device agreement or survival.

For the frozen shift–Lambda principal subblock, eliminating D gives advection plus transverse speed sqrt(mu q) and longitudinal speed sqrt(4mu q/3). At scri beta^i x_i=-s and mu=3chi/2: coordinate normal speeds are s(1±sqrt(3/2)) (two transverse pairs) and s(1±sqrt(2)) (one longitudinal pair). There are three incoming shift modes; the E transport speed is s, outgoing. Removing the lower-order source does not alter this subblock's principal speeds. This count is not a proof for the coupled lapse/metric/Z4 system or a boundary energy estimate. Prescribing beta and reconstructing D are one compatibility boundary treatment, not evidence that an independent outgoing E condition may be freely imposed.

## Audits and execution gates

Local completed: independent bulk-source oracle; nontrivial CPU/Serial-device closure with changing geometry, nonzero shift gradients and angular dissipation (absolute shift compatibility1.66e-16, differentiated null2.57e-16, old limit magnitude.1866, geometry speed rate.0352). Regular manufactured N16/24/32 refinement shows decreasing D-trace errors (p2 pair orders5.72/5.82); the nonconstant twist metric retains original surface orders2.970/2.985 and first interior row2.976/2.989. These are operator tests, not evolved exact solutions. The necessary differential auxiliary trace condition is derived above, but a coupled boundary energy/well-posedness proof remains unresolved; no claim that arbitrary initial jets satisfy all time-compatibility conditions. Actual GPU checks are pending. Preserve known original-scri truncation-order limitations rather than incorporating first6/KO10 repairs. Distinguish local Serial device kernels from actual SYCL/GPU results.

After these checks: separate frozen Aurora build, short12-rank GPU physics parity, then fresh N16 and N24. Existing account/debug-scaling, one node12ranks and5GB guard; inspect both queues/saved submission before every submit. Preserve/hash artifacts before retirement. Only fully verified healthy same-formulation walltime checkpoints may continue automatically. Numerical failure is terminal and must be localized.

Compare N16 to preserved native A; N24 SG1 used text initialization and is not an exactly matched native baseline. Assess t0 native/text differences and use a fresh matched native-A N24 control if needed. Record fixed-cutoff/full/candidate histories, horizon gaps and accepted masses, scri residuals, shift gradients/deviations, D behavior and separate boundary/sign counters. Matching admitted endpoints and common cutoff, radial plots and N16→N24 trends do not establish fourth order with two resolutions.

Read-only history additions for NZ1 record fixed coordinate cuts .12/.16/.25 and max Euclidean/Frobenius norms of beta+x, its gradient, D, gradient D, and D-Lambda. Fixed cuts are labeled coordinate cuts, not assumed horizons; endpoint analysis must choose a common radius outside measured horizons. Baseline checkpoint formats and evolution arithmetic are unchanged.
