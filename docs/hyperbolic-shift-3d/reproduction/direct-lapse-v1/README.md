# Direct lapse retry

The chi-floor run failed at 9.156M with positive-infinite lapse in the eight
nearest puncture cells. This experiment tests the user's proposed direct-alpha
update instead of logarithmic lapse RK evolution. It retains the existing
telegrapher PDE, log-chi RK/KO, chi floor1e-8 at r<.03, scri closure, interpolation,
dt=.0005 (dt/dx=.0444444), n16, RK4 and geometry/projection policy. The lapse
perturbation alpha-A (static A=(1+r²)/2) now advances directly at each RK stage
and final compensated update; finite candidates are floored at physical
alpha>=1e-10, equivalently conformal alpha>=1e-10*Omega. Compensation is cleared
when a final lapse update is clipped. Nonfinite candidates are still rejected.
Lapse KO now acts on the stored linear lapse perturbation. Positive interpolation
and unused logarithmic lapse chart construction remain as in the control.
This is a combined RK-variable/KO-variable comparison, not an isolated RK-only
change. No claim of long-time stability or equivalence to Cauchy Z4c is made.

## Main branch audit

Fetched HengruiZhu99/athenak main at c8d5c33e39089f005e416bf839652f6796fa6e82.
In src/z4c/z4c_update.cpp, ExpRKUpdate linearly combines stored fields and their
RHS at each explicit RK stage. src/z4c/z4c_calcrhs.cpp lines535-547 computes the
lapse RHS using advective 1+log/harmonic slicing and optional slow-start lapse;
lines570-582 applies KO to all stored variables. No lapse floor was found in
main's src/z4c. The optional floor in z4c_tasks.cpp lines362-384 applies to chi;
it is disabled by default in z4c.cpp. Therefore direct lapse evolution agrees
with main's variable choice, but our lapse floor is the retained donor policy,
and our telegrapher gauge is not main's slicing equation. Similar gravitational
bulk equations alone do not establish the same puncture stability.

## Checkpoint and campaign policy

The new executable is hyperbolic_shift_3d_mpi_direct_lapse. AL1 checkpoint magic
HYP3D_MPI_AL1 has a168-byte header; PositiveConfig.scheme=2 at byte144 identifies
direct lapse RK/KO, with chi floor1e-8/radius.03 at byte152. Existing schemes
retain their original identities and default behavior. The explicit importer
accepts only the verified admitted DT1 checkpoint at3M with SHA256
628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31.
Physical fields and non-lapse compensation are unchanged; volume lapse
compensation is explicitly reset because its old units belong to log(alpha).
No failed checkpoint is admitted or reused.

Aurora target is20M from the verified3M state with n16, dt=.0005, original
penalty strengths, diagnostic interval.05, horizon lmax12/ntheta24/tolerance.001.
Use CompactBinaryMerger/debug, one node at a time,12 MPI ranks/GPU tiles,
SYCL+MPI, make -j64, and home footprint below5,000,000,000 bytes. Freeze a fresh
source/build and bind hashes before submission. Continue only healthy walltime
checkpoints; numerical failures must stop and be reported. Large remote evidence
must be copied and SHA256-verified locally before purge.
