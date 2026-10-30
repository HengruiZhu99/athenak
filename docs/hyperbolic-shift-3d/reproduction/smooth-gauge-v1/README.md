# Broader gauge-weight comparison: N16 then N24 through 3M

The user authorized a matching N16/N24 test of a smoother, better-resolved gauge
blend, and explicitly limited this comparison to3M. Both cases start from the
same original generator inputs as their donor runs. No restart or direct-variable
continuation is used. This is a separate equation experiment, not a replacement
for the frozen N32 campaign.

Only the telegrapher coupling weight w and its analytic derivative change.
The original w=S8((0.8-r)/0.24), S8(t)=t^8/(t^8+(1-t)^8), is replaced by
S9(t)=126t^5-420t^6+540t^7-315t^8+70t^9, with exact zero/one extensions.
Its derivative is630t^4(1-t)^4; w_r is this times -1/0.24. Symmetry is used
when evaluating the upper half to avoid endpoint cancellation. This degree-nine
polynomial has four continuous derivatives at the endpoints. The original
rational blend has more endpoint differentiability but is sharply concentrated;
the intended improvement here is a broader resolved transition, not higher
formal differentiability.

The interval0.56–0.8 and midpoint0.68 remain fixed. The separate harmonic collar
on0.8–0.9 is unchanged, so w times collar remains identically zero. Omega, the
reference geometry, initial slices, bulk coefficients, initial shift, mesh,
angular dissipation, penalties and timestep are unchanged. This is a gauge
coupling change, not a remapping of the spacetime slices at initialization.

Use n16 dt=.0005 then n24 dt=.0003333333333333333, each to coordinate time1=3M,
with the same log-chi/log-lapse RK and KO scheme as the original0–3M comparison.
The physical lapse floor remains1e-10; no new chi floor is introduced into DT1.
Preserve horizon lmax12/ntheta24/tolerance.001 and diagnostic interval.05.
Complete candidate horizons are retained as provisional exterior cutoffs while
preserving the original acceptance flags. This matches the earlier diagnostic
replay and prevents missing exterior histories from finder-tolerance failures.

The separate target is hyperbolic_shift_3d_mpi_smooth_gauge. Its checkpoint magic
HYP3D_MPI_GW1 binds the fixed polynomial profile and rejects DT1/AC1 files. The
header stays152 bytes with the original DT1 configuration blocks. The existing
targets default to the original weight. A matching diagnostic-only checkpoint
reader is hyperbolic_shift_3d_checkpoint_constraints_smooth.

Plan: campaigns/n16-n24-smooth-gauge-v1/plan.json. It runs N16 then N24 in one
allocation, one node/12 MPI ranks/GPU tiles, CompactBinaryMerger/debug. Native
walltime reserve180s and worker budget3200s are retained. The controller restricts
this phase to fresh N16 then N24 inputs, exact3M targets and the matched dt and
diagnostic settings. Preserve the5GB task footprint cap, current N32 inputs,
and all source/binary provenance; archive and verify obsolete data before purge.
Do not overlap debug allocations or silently modify the running N32 experiment.

Compare time histories with the original candidate-diagnostic replay and radial
constraint integrals at3M using the same r=0.0753 cutoff and bands. Record horizon
quality, null closure, lapse/floor diagnostics and any numerical failure. Do not
infer convergence from two resolutions or from a change of gauge alone. Stop
after the matched3M comparison; no20M smooth-weight continuation is authorized.
