# Direct chi and lapse retry

The user authorized this trial after the direct-lapse/log-chi replay failed
at9.0975M. It starts from the verified admitted3M physical state, not a failed
checkpoint. Both chi and conformal lapse perturbation alpha-A now use linear
compensated RK4 updates and linear KO. The telegrapher PDE, dt=.0005, n16,
mesh/overlap/interpolation, angular epsilon2, scri penalty scale1 and horizon
settings remain as in the preceding run. This tests both the RK and KO variable
choice; it is not a pure RK comparison or a demonstration of Cauchy equivalence.

The physical lapse floor remains1e-10. Positive chi below1e-8 is floored only
inside r<.03. To honor the user's instruction that finite negative candidates
must not terminate evolution while retaining floors, finite chi<=0 is recorded
and reset to1e-8 everywhere, including the surface and its interpolated ghost
boundary values. Finite negative lapse is recorded before its existing floor.
No upper clamp is introduced. Nonfinite arithmetic and invalid positive-definite
metric geometry remain failures. Chi and lapse are positive after flooring;
this is not evolution with a signed physical metric.

Candidate sign counts, minima, positions and first coordinate times are recorded
inside existing bulk RK kernels and gathered once at the segment end/failure.
The surface uses the same counters; point_id=-(surface_point+1) distinguishes
surface rows. Counts include rejected attempts and reset per segment. Volume
chi-floor counts include positive below-floor clips; separate surface floor
counts equal its negative/zero candidate events. Scri interpolated chi negative
and zero counts/minimum are separately accumulated over all halo calls (including
diagnostics, repeated points and repeated ranks); they are not unique-cell counts.
No-event candidate rows are omitted. Nonfinite candidates are not negative counts.

## Checkpoint and restart

Target: hyperbolic_shift_3d_mpi_direct_chi_lapse.
Schema HYP3D_MPI_AC1,168-byte header, PositiveConfig.scheme3 at byte144,
chi floor/radius1e-8/.03 at byte152. The importer
`tst/hyperbolic_shift_3d/prepare_direct_chi_lapse_restart.py` accepts only the
verified DT1 source SHA256
628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31.
It preserves physical volume/surface fields exactly and resets volume lapse,
volume chi and surface chi compensation because their units change from log
to linear. Other compensation remains unchanged. The imported AC1 input SHA256 is
29e8e82cd143903430f1d207fc2b8cb53328b4b731d83612f38fd63192fd080b.
This explicit scheme distinguishes the broader finite-nonpositive rescue policy.

Only terminal healthy walltime stops may continue. Before a continuation, verify
checkpoint against report and worker SHA256/size: AC1 magic,n16,nf31,kind2,
admitted1,dt.0005,time in[1,20/3),scheme3,chi floor/radius1e-8/.03,
length168+16*(31*cells+surface),matching cumulative steps and time. Preserve all
other configuration bytes and frozen source/executable hashes. A horizon-seed
sidecar may be used only if bound to that exact checkpoint. Preserve horizon
quality flags and sum segment-local candidate/floor counts.

Release the matching terminal root lock using the frozen controller, check all
user debug jobs, create a fresh sequential plan with experimental_policy
n16_direct_chi_lapse_retry_to_20M, bind the validated restart and frozen source,
then submit. Use CompactBinaryMerger/debug, one node/job at a time,12 MPI ranks
and GPU tiles; target20M (coordinate20/3), horizon lmax12/ntheta24/tolerance.001,
diagnostic interval.05. Build SYCL+MPI using make -j64. Keep task footprint below
5,000,000,000 bytes. Copy and hash-verify large evidence before remote purge;
retain inputs until successors no longer need them. Do not resume numerical or
storage failures or ambiguous scheduler outcomes. No additional fresh failure
replay is authorized by this trial. On failure or verified completion collect,
commit/push evidence, release terminal lock and delete the monitor. Reaching20M
alone does not establish exterior accuracy or convergence.


## Local validation

All53 controller tests and the chi/lapse floor/sign unit cases passed. Two-rank
n16 Minkowski completed two steps; puncture completed3M to3.003M, and split
versus uninterrupted checkpoints (including compensation) were bitwise identical
with identical dt. An AL1 input was rejected by the AC1 executable. These smokes
used horizon lmax1/ntheta4; the Aurora trial retains12/24.

A synthetic final-compensation perturbation produced negative chi(-.999883) and
lapse(-.990604) at cell4095. Both were logged, floored and the step completed.
This verifies reporting and continuation through finite negative candidates;
its deliberately corrupted physical state is not a stability/accuracy result.
The first synthetic invocation collided with its own input checkpoint name and
was refused before evolution; distinct input/output names resolved that harness
error. An expected incompatible-checkpoint rejection emits no report; the harness
was corrected to capture its error log. Scientific runs were not repeated for
these harness fixes. See local-test-results.json and local-tests.py.


Frozen scientific source:5b379718c344c23ed4b5f859c5cbc2b4bd9ce7a3.
Remote root:/home/hzhu/hyperbolic-core-shell-20261028.
Source:source-direct-chi-lapse-v1; manifest:source-direct-chi-lapse-v1-manifest.json.
Controller:source-direct-chi-lapse-v1/tst/hyperbolic_shift_3d/aurora/control.py.
Binary:build-sycl-mpi-direct-chi-lapse-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_chi_lapse.
First plan:campaigns/n16-direct-chi-lapse-to20m-v1-segment01/plan.json.
Minkowski's two-step maximum field drift was6.51063e-15; maximum surface and
ghost null residuals were6.66e-16 and1.11e-15. These are roundoff-scale short
checks, not a long-time convergence result.


## Submission

SYCL/MPI build passed in277.88 seconds using make -j64. Job8990504 was submitted
on one CompactBinaryMerger debug node with12 MPI ranks/GPU tiles, target20M.
All-user debug queue inspection found no competing job before submission. The
30-minute monitor aurora-n16-direct-lapse-to20m now follows this AC1 trial and
continues only verified healthy walltime checkpoints. Exact source/binary/plan
hashes, storage reserve and monitor policy are saved beside this file. The task
footprint at binding/submission remains below5GB. Submission is not evidence of
long-time stability; no20M result is claimed here.
