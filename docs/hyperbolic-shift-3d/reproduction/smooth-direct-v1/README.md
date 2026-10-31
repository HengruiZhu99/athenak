# Fresh smooth-gauge direct-variable campaign

The user requested direct signed chi and alpha evolution with floors, from fresh
wormhole data at t=0: N16 toward20M first, then N24 toward20M. This supersedes the
GW1 log-variable continuation. N16 GW1 failed at9.1575M; N24 job9013517 was
cancelled by request, not classified as a numerical failure. Its terminal PBS
records and partial diagnostics are preserved. Neither3M nor failed checkpoints
are inputs to the first segments of this new campaign.

Target hyperbolic_shift_3d_mpi_smooth_direct combines the existing AC1 direct RK
and physical-variable KO treatment with smoothstep9 gauge blending on0.56–0.8
and complete-candidate exterior diagnostics. GD1 is a distinct168-byte checkpoint
schema with PositiveConfig.scheme3 and chi floor/radius1e-8/.03. No old target
changes its equations or checkpoint identity. Finite signed candidates are
counted before clipping; they do not terminate evolution. Physical lapse floor
is1e-10, positive chi floor1e-8 applies inside r<.03, and finite nonpositive chi
is rescued everywhere including boundary interpolants. Nonfinite arithmetic and
invalid metric geometry remain failures. Surface lapse remains fixed by scri
closure. No chi or lapse logarithmic RK/KO is used in GD1.

Both grids start from the original wormhole generator text, with SHA256:
- N16:35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca
- N24:6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914

Keep N16dt=.0005,N24dt=1/3000,dt/dx=.0444444,target coordinate20/3=20M,
angular epsilon2,scri scale1,original harmonic collar, horizon lmax12/ntheta24/
tolerance.001 and diagnostic interval.05. Preserve candidate-horizon quality
flags; reaching20M alone does not establish accuracy or convergence.

Use smooth_direct_campaign.py to create fresh segment01 from --initial, or
later sequential segments from --restart and independently verified --sha256.
It rejects GW1/AC1, unadmitted/configuration-mismatched/nonfinite checkpoints,
changed initial data and3M starts. Freeze smooth-direct-science.json beside the
launcher after the SYCL/MPI build and include it in launcher provenance.
For each terminal healthy walltime stop: collect the full campaign, copy and
hash/length verify all artifacts locally, run helper verify, release only its
matching terminal lock, then create/bind/submit the next segment once. Never
resume numerical failures; advance from N16 failure/completion to fresh N24.
On storage/integrity/scheduler failure, diagnose before retrying. Do not change
science, timestep or floors between segments. Sum segment-local candidate/floor
counters, which include rejected attempts; keep boundary interpolation counters
separate. After both terminate, compare histories, preserve gaps, commit/push
results and delete the hourly monitor.

Use CompactBinaryMerger/debug, one node/job at a time,12 MPI ranks/GPU tiles,
worker3200s,command3100s,native exit reserve180s. Check all user debug jobs before
submission. Build SYCL/MPI with make -j64 using the recent oneAPI configuration.
Keep task footprint below5,000,000,000 bytes including output/atomic reservations;
copy and hash-verify large evidence before purge. Stay quiet on unchanged monitor
states; notify progress, continuation, completion, failure or required action.
