# Investigation of DO1 constraint growth

The user requests cause investigation and modified N16 tests, possibly moving
the gauge transition outward; N24 may resume only after N16 succeeds. Previous
turn made progress: DO1 N16 was verified to10M and its poor constraints measured.
Current authoritative state: N24 job9017333 was running, then cancelled under
this new priority (PBS terminal F,exit143, before native output/worker status).
The old20-minute N16/N24 continuation monitor was deleted. Terminal lock was
released; cancelled campaign records are locally preserved. No N24 continuation
is allowed merely because a new N16 reaches its endpoint.

## Measured localization

Read-only native diagnostic targets load the exact IT1 and DO1 admitted10M
checkpoints with their respective schema, halo closure and volume kernels.
SHA256 before/after confirms unchanged checkpoints. Common exterior cutoff
r=.1 is outside both candidate radii. No horizon finder is rerun and no radial
interpolation is used for the norms. Cumulative volume integrals are differenced
to obtain radial-band norms. All commands, binary hashes and source checksums
are in execution.json; profiles.py restores the stored reference offsets.

At10M on r>.1:

| Norm | IT1 | DO1 |
|---|---:|---:|
| Hamiltonian RMS | .00810299 | .23426143 |
| Momentum magnitude RMS | .01244080 | .32180363 |
| Radial momentum RMS | .01319300 | .67087103 |
| Theta RMS | .00041284 | .01688544 |

65.99% of DO1's full-domain Hamiltonian squared integral lies in .16<r<.30;
only .083% is at r>.9. The same .16--.30 band contributes .229% in IT1.
DO1 develops large oscillatory Theta near .16--.35. Thus the exterior growth
is not an artifact of choosing a smaller horizon, and the final error is
concentrated around the Cartesian/shell transition rather than scri. Final
spatial concentration does NOT establish where the instability first originated.

## Controlled next test: DG1

DO1 changed both gauge blending and constraint damping relative to IT1. Its
smoother .2--.9 transition kept weak physical Cauchy damping active farther out
and removed the former 8A enhancement. Simply moving both transitions farther
out would confound gauge placement with even weaker damping near the overlap.
First run DG1: retain DO1's COMPLETE gauge, including its .2--.9 weight and
principal symbol, and restore exactly IT1's damping coefficient interpolation.
Only HyperbolicDamping changes. The damping weight is Smoothstep9((.56-r)/.36),
with the original .56--.8 donor damping enhancement. Inner physical damping,
outer damping and all gauge equations retain their previous endpoint limits.

This is a causal ablation, not an assertion that damping is already proven to
be the cause. If restoring it suppresses the growth, the weakened constraint
subsystem was a material contributor. If growth persists, investigate gauge
forcing/propagation and test a later gauge transition while holding damping
fixed. A moved transition should preserve the exact inner gauge and reach the
unchanged outer gauge before scri; measure its physical extent and resolution,
rather than assuming compactified coordinate width is sufficient.

DG1 target hyperbolic_shift_3d_mpi_damping_ablation has unique magic
HYP3D_MPI_DG1,header168,scheme3. Fresh original N16 wormhole text at t=0,
dt=.0005,target10/3=10M. Direct chi/lapse RK/KO,existing floors/sign logging,
mesh,FD4,KO,angular epsilon2,scri penalty1,horizon lmax12/ntheta24/tolerance.001,
diagnostic interval.05 are unchanged. No old checkpoint conversion. Physical
oracle tests check unchanged full gauge and restored damping at off-diagonal,
nonradial states; CPU/device parity and N16-only campaign policy tests passed.

N16 success requires more than finite evolution: compare the measured history
and common-domain10M norms with IT1, check whether the post4M growth is removed,
retain horizon quality flags and check horizon mass/radius against the baseline.
Do not advance to N24 automatically based solely on native/worker completion.
The initial DG1 run is diagnostic; review evidence before declaring a successful
formulation or choosing the next modification. The full objective remains active.

At r=.30,.35,.40 on the initial CMC reference, the old/new Theta damping-rate
ratios are8.36,15.15,13.29. This is a coefficient comparison, not an observed
growth-rate fit. The damping-only DG1 test separates this change from gauge
placement. Short local DG1 evolution reached coordinate.001 in two steps;
a DG1 restart reached.0015 in three total steps, with null residual<=1.12e-15.

All123 controller/worker/campaign tests passed. The new campaign helper rejects
N24 plans and incompatible DO1/IT1/GD1 restart schemas. Frozen source for DG1:
35004000d25e34767069a8da1d2ad45bdc42c3c1. Later documentation does not alter
the built source snapshot.
