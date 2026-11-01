# Direct inner-telegrapher to outer-hyperboloidal transition

The user authorized fresh N16 followed by fresh N24, each toward 10M. DO1 is a
separate opt-in target, hyperbolic_shift_3d_mpi_direct_outer. IT1 and all older
scientific targets remain available. Start both resolutions from the original
wormhole generator text, never from an older evolved solution.

## Formulation

This implements the candidate audited in ../inner-telegrapher-v1/characteristics/.
The inner physical gauge and mass-scaled constants match IT1 and the reference
project/telegrapher_lapse b99e27f branch on r<=.2. Let
s=Smoothstep9((.9-r)/.7). Interpolate COMPLETE local gauge RHSs with s, reaching
the existing outer gauge at r>=.9. No derivative of s is added. The outer branch
uses mu=1.5*chi_bar*F everywhere, F=alpha_bar^2+4*Omega^2*alpha_bar, no Euclidean
B/lapse coupling and no gradient forcing in B; B still advects and relaxes with
tau=.02. The CMC reference sources and original restoring collar .8--.9 remain.
Thus the previous mu=4 plateau and old lapse/B transition are absent. The
legacy gauge_blend report field names the retained coefficient definition;
inner_gauge=telegrapher_direct_outer_r020_090_v1 identifies the active DO1 RHS.

Damping uses the same s: physical inner kappa1=.03, kappa2=0 to donor outer
kappa1=4,kappa2=-.99, interpolating the coefficients multiplying Theta/Z.
The old 8*A inner damping correction is absent. The geometric RHS's unwanted
-3*Omega.normal*Theta term is cancelled as before. Other off-constraint geometric
terms, grid, FD4, KO, coupling, projection and scri closure are unchanged.
This is not full equivalence to the production Cauchy Z4c kernel.

Both chi and lapse evolve as signed physical-variable RK candidates with KO
from t=0. Retain physical lapse floor 1e-10, positive chi_bar floor 1e-8 inside
r<.03, and finite nonpositive chi rescue everywhere including boundary
interpolants. Record negative/zero candidates before clipping; signs alone do
not terminate. Nonfinite arithmetic and invalid metric geometry still fail.

The principal coefficients multiply AFTER blending. For Q=g^{ij}n_i n_j,

    lapse_speed_squared = [chi*((1-s)*F+s*2*Omega*alpha_bar)
                            +2*s^2*max(Omega^2*chi,1e-5)]*Q
    longitudinal_shift_speed_squared = (4/3)*[(1-s)*1.5*chi*F+s]*Q

This preserves the audited proposal, including its known exact inner collision
counterexample. It is not a proof of global strong hyperbolicity. Independent
compiled symbols agree on 401 random/reference states to 8.89e-16 maximum
matrix-entry difference. The physical gauge oracle checks nonradial gradients,
off-diagonal metrics, guarded chi, damping, transition and outer/scri identity.
CPU/device partition parity and tensor-scri tests passed. See checks/.

## Execution and restart protocol

N16 dt=.0005, N24 dt=1/3000; dt/dx=.0444444444444444. Target coordinate10/3=10M.
Preserve horizon lmax12/ntheta24/tolerance.001, diagnostic interval.05,
angular epsilon2, scri penalty1 and complete-candidate exterior diagnostics;
retain horizon acceptance flags and missing values. N16 then N24, one
CompactBinaryMerger/debug node at a time,12 MPI ranks/GPU tiles, SYCL/PVC+MPI,
make -j64. Home task footprint must stay below5,000,000,000 bytes including
output/atomic checkpoint reservations. Preserve unrelated jobs.

DO1 uses HYP3D_MPI_DO1,168-byte header,PositiveConfig.scheme3 and exact floor
configuration. N16 checkpoint length79,725,224; N24 length234,413,736. Old
IT1/GD1/GW1/AC1 checkpoints are incompatible. direct_outer_campaign.py admits
only fresh original generator data for segment01, then verified DO1 restarts.
Native target10M and all scientific coefficients are frozen for this campaign.

After a terminal job collect the full campaign directory, retaining campaigns/
layout locally; verify SHA256 and byte lengths of every worker artifact. For a
healthy walltime stop or completion run the frozen helper:

    python3.11 launcher-direct-outer-v1/direct_outer_campaign.py verify --root ROOT --plan PLAN

It checks scheduler/worker/native status, frozen bindings, admitted schema,
configuration/dimensions/dt/time/steps, finite payload and forward progress.
Release ONLY the matching terminal root lock, once. If healthy below10M, create
a fresh sequential same-N plan with --restart VERIFIED_CHECKPOINT --sha256 HASH,
preserve/bind its horizon seed sidecar, bind the unchanged manifest, check all
user debug jobs, submit once and update direct-outer-10m-state.json. Never
submit a duplicate queued/running/ambiguous job. Worker3200s,command3100s,
native exit reserve180s. Never resume a numerical failure, corrupt/unadmitted
state or change dt/floors automatically. Resolve infrastructure failures first.

After N16 completes or numerically fails, start N24 from its own original data.
After both terminate compare histories against IT1 and earlier baselines, sum
segment-local candidate/floor counts (including rejected attempts), retain
separate boundary counters and horizon flags. Commit/push concise evidence and
plots, report results and delete the monitor. Completion alone does not establish
accuracy or convergence. Copy/hash-verify large evidence before remote purge;
retain required restart inputs until successors have verified outputs.

Local puncture smoke evolution completed two steps to coordinate .001 and a
DO1 restart completed the third step to .0015. A two-step Minkowski evolution
also completed; maximum null residual was1.12e-15 across these checks. A native
IT1 restart attempt was rejected with schema/configuration/admission mismatch.
All121 controller/worker/campaign tests passed. These short tests establish
implementation/setup consistency, not long-time stability or convergence.

## Frozen Aurora campaign

Scientific source e1ec5803bd5c0694b91f019e016407386cd7f538.
Manifest source-direct-outer-v1-manifest.json SHA256
7f7dd0783f042e7786c1332bff604d710879f27a8453b9fcc0f689e3332e24a3.
Binary build-sycl-mpi-direct-outer-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_outer
SHA256640fab47d261055f90aeb5d41d8c5090352a2da7282f5f37f99703cdd529c902.
Built with oneapi/release/2025.3.1, SYCL/PVC+MPI, make -j64; all1486 copied
Kokkos source files match the previous frozen manifest. Source manifest excludes
Python bytecode/cache files. Later test/documentation commits do not change this
frozen runtime source or binary.

N16 job9016957 is submitted, plan
campaigns/n16-direct-outer-fresh10m-v1-segment01/plan.json,
SHA2561415b8558768b8ce21f92bde4db67026bbb959d06987b4373bb94b0eaa3c65a7.
N24 campaigns/n24-direct-outer-fresh10m-v1-segment01/plan.json is prepared,
unsubmitted. Follow remote direct-outer-10m-state.json and saved submissions.
Launcher: launcher-direct-outer-v1; verify files against provenance.json and
use its direct_outer_campaign.py/control.py. N16 binding measured3,296,782,529
bytes, with1,475,931,888 bytes headroom after output and atomic reservations.
All user debug jobs were checked before submission; unrelated held capacity
jobs remain unchanged. Hourly monitor aurora-direct-outer-n16-then-n24-to-10m
is active for healthy continuations and sequential N24 dispatch. Submission
alone establishes no duration result.

Original input hashes:

- N16 inputs/puncture-continuous-n16-smooth-v1.txt:
  35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca
- N24 inputs/puncture-continuous-n24-smooth-v1.txt:
  6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914
