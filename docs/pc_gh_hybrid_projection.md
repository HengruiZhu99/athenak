# Puncture-local projection with propagating reductions

2026-09-05. Status: implemented research candidate; initial CUDA correctness
gates pass, puncture and full qualification remain pending.
This records the accepted hybrid plan and its mathematical limits. The underlying
regular advective extension, Einstein residual identities, and r>0 principal
analysis remain those in `pc_gh_regular_extension.md`.

## Accepted experiment

The document-grounded interview chose both finite relaxation and actual
projection, FD6 first with FD2 controls, a full reset in the projection core after
each complete step, and a physical mask-width sweep. The initial arms are C1,
C4, C16 (constant rates), R4/R16 (outer rate 1, inner 4/16), and P1 (rate 1 plus
tapered projection). Units are M=1. The default radii are (1/8,1/2), with
(1/16,1/4) and (1/4,1) reserved for survivor sensitivity tests. The large mask is
diagnostic and does not confer qualification by survival.

Keep the cell-centered transfers, KO=.3, moving-puncture gauge, and algebraic
enforcement fixed; disable GH gauge projection. Retain the original failure
reproductions and compare candidates under the same coordinate-time ceiling
.0125 M. Independent halved-step controls are required before promotion.

The skills installed from pinned upstream revisions are Matt Pocock's grill-me
and grilling at 3cca18b368ae95cdbdebbff572ccafa662551015 and neverholiday's
grill-with-doc at 27f7de47e7481f255443fe56763bfbd52bfaa215. The user explicitly
ended the interview and authorized implementation. No further interview approval
is required for this campaign.

## The exact map

Hold the 22 non-reduction variables fixed during projection. Let G contain the
33 stored p/Q/L/B components, and define the targets with the existing finite
differences:

    T_p_i = D_i w
    T_Q_iab = D_i gtilde_ab
    T_L_i = 2 (w D_i rho + rho D_i w)
    T_B_ia = D_i beta_a.

For an inactive direction all its targets are zero. With E=G-T and the existing
C-infinity mask union P=1-product_n(1-P_n), apply G+ = G-P E on active cells.
The explicit P=0 and P=1 branches preserve untouched exterior values bitwise
and make the full reset identical to the historical global target assignment.

At fixed configuration E+=(1-P)E exactly before subsequent transfer. The stored
diagnostic Ralpha=L-2(w D rho+rho p), although different from E_L, obeys the same
scaling because p and L receive the same mask. Q trace enforcement is not repeated
afterward: D gtilde is generally not exactly trace free under a discrete chain
rule, and a second trace reset would change the derivative target again. Both
residuals must be measured, not silently reconciled by alternating projections.

In the continuum, all targets are exact gradients (T_L=d(2w rho)). Hence

    curl(G+) = (1-P) curl(G) - dP wedge E.

The taper term is homogeneous in reductions but can create curl. This formula
is not applied as a discrete product rule. For any fixed linear discrete curl C_D,
the exact identity is instead

    C_D(G+) = C_D(G) - C_D(P E)
             = (1-P) C_D(G) + P C_D(T) - [C_D,P] E.

The factorized discrete L target can have nonzero curl. Ghosts not yet projected,
boundary refresh, independent AMR interpolation, and algebraic corrections must
be represented explicitly in the actual state used by C_D. An immediately
projected core does not imply zero curl or reduction residual after exchange.

## Einstein consistency, propagation, and temporal limits

For a smooth continuum Einstein solution in the established gauge on r>0,
all defining reductions and curls vanish. The projection is then the identity;
its spatial derivatives also vanish as corrections. It leaves the Einstein
solution unchanged and leaves the already audited propagation/damping system
between projections unchanged. On finite grids, derivative and algebraic
identities disagree by truncation error; this is precisely why convergence of
corrections and exterior observables is required.

Between maps, the full nonlinear subsidiary equations of the existing extension
apply, including the spatial gradient of the finite relaxation coefficient.
At a map, the jump identities above close the constraint update at fixed
configuration. This supplies a hybrid continuous/discrete description, not a new
smooth FO-GH PDE or a uniform puncture-point hyperbolicity theorem. The map adds
no divisions by evolved fields. It does not regularize an invalid state:
post-RK checks remain before projection.

A full reset every step is not finite-rate relaxation as dt tends to zero. At
a point with fixed 0<P<1, the effective discrete rate is -log(1-P)/dt and becomes
unbounded. The P=1 core is an exact discrete constraint enforcement. Thus the
entire map must be checked under dt refinement; the RK3 label does not establish
third-order accuracy for the composed method. We do not claim an energy estimate
for the AMR composition from the contraction of E alone.

## Interfaces and measurement

`reduction_projection_profile=global|smooth_core` defaults to global and is
enabled by the existing `project_reduction_constraints` switch. Geometry and
centers use the existing reduction-mask parameters. Mask allocation/validation
also supports projection-only and research diagnostic use. Moving mask centers
share the RK tracker update and are refreshed from completed-stage positions
before projection, without advancing them twice.

`research_dt_ceiling=0` disables the new ceiling. A positive finite value is the
actual coordinate-time ceiling, accounted for before the mesh's CFL multiply.
Existing characteristic and relaxation stability restrictions still apply.

`hybrid_monitor=true` enables reduction monitoring and writes a separate
`.hybrid.csv`. It records all/core/taper/exterior/face-interface maxima,
locations, and coordinate-volume L1 integrals at each existing operation bracket.
The face-interface stratum means active cells within the finite-difference radius
of a face with a different-level neighbor; it does not label diagonal-only
neighbors. Empty strata have block=-1 and zero norm. Diagnostics never suppress
the existing unmasked strict state or constraint checks.

Operation 100 contains actual p/Q/L/B correction norms from the discrete map.
For each full step, its L1/dt is the average correction magnitude per unit time;
summing its L1 over completed steps gives cumulative variation, including across
restart segments when duplicate checkpoint times are removed. Other operations
retain their distinct immediate and post-refresh constraint measurements; a
difference of norms is never labelled a norm of a vector correction or a flux.

## Qualification sequence

Run the independent ghost-inclusive 33-component projection oracle, exact flat
and shifted waves, fixed/moving pulses, restart, and MPI controls on Della CUDA.
Reproduce the saved C1 uniform failure near 8.38 M and C16 M/256 core failure
near 4.89 M. Screen the six arms to 12 M on uniform M/8 and 6 M on the saved
M/256 core hierarchy. Survivors receive the agreed three-resolution, three-mask,
interface-location, and halved-step comparisons before binary work.

No strict failure, resolution-growing puncture divergence, unexplained
refinement-growing curl, unresolved exterior convergence, or nonconverging
correction is a pass. Mask dependence must decline under resolution and fall
within the measured resolution uncertainty. Uniform M/8--M/12 controls are not
matched M/256 core controls. Only qualified candidates advance to the established
head-on gate beyond 73.8 M and toward 100 M; prefer finite relaxation if equally
qualified. Preserve every failed run; do not escalate KO/rates or modify transfers
to turn this campaign into an uncontrolled search.

## Initial evidence (2026-09-05)

The serial CUDA build and MPI CUDA build compile on Della; the local compilation
check also passes. All 26 independent projection cases pass across FD2/FD6,
1D/2D/3D, zero/full/taper/global masks, and overlapping centers refreshed after
a prescribed tracker displacement. Worst component discrepancy is 3.584e-15;
worst discrete curl-identity discrepancy is 4.685e-14. Three two-rank CUDA
projection oracles (global/taper/overlap) also pass.

All six corrected flat-space controls pass. All six N32/N64/N128 shifted-wave
ladders pass the original all-sector checker with minimum order 1.8 and exact
tolerance 1e-12; no threshold was changed. KO=.3 and the same timestep ceiling
are used across the candidate comparisons. These results do not qualify
puncture stability, mask independence, or waveform accuracy.

Preserved setup failures: the first symbolic matrix identity needed expansion
before equality comparison; the first flat inputs selected FD6 but retained two
ghosts and were rejected before evolution. Corrected four-ghost inputs pass.
The first Slurm submission was rejected for an explicit partition; job 13479747
then failed at shell setup because the module environment references PS1 under
nounset. Neither launched an evolution. Corrected reproduction jobs are
13479856_0/1; pulse jobs are the 13479875 array. Scientific failure and setup
failure are kept separate in the evidence.

The preserved remote root is
`/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317` on Della. Local evidence
is under `qualification-runs-20260905/hybrid/`. Each run records its executable
hash, source parent and diff, input, hardware, and exact command. The parent is
9f674aec with the implementation patch; the later local feature commit does not
retroactively change the build provenance.

The first diagnostic schema inadvertently reused operation 8, already assigned
to physical boundary updates, for jump corrections. The dynamics and projection
cadence were unaffected. The jump verifier rejected the resulting invalid event
sequence. The corrected schema uses operation 100. The old CSV is preserved and
can be recovered unambiguously by selecting only correction rows inside the
operation-3 before/after projection bracket; the verifier checks final-stage and
unique-step consistency rather than silently accepting duplicate events.

The complete hybrid monitor additionally refreshes GH/ADM and algebraic diagnostics
at each transfer/RK bracket. It reports the minimum metric eigenvalue, w, rho,
and alpha by region, alongside GH Cperp/Z, H, alpha-weighted momentum, determinant,
trace-A and trace-Q residuals. For quantities named `min_*`, the CSV `max` column
contains the stated minimum (with its location); `coordinate_l1` holds the signed
coordinate-volume integral of that scalar. No inner-radius mask censors these
samples. The projection physical-boundary bracket has operation 9. Projection
operation 3, refresh operations 4/5/9/6, and jump operation 100 allow the immediate
map and subsequent derivative-target changes to be distinguished. Refreshing
constraint diagnostics changes no evolved field and retains strict checks.
