# Complete intrinsic point kernel: CPU verified, CUDA failure open

`intrinsic_rhs.hpp` evaluates all 50 rows of the pinned intrinsic candidate.
`intrinsic_sources.hpp` contains the ten complete configuration sources and
lower triangular chart rate. `intrinsic_jet.hpp` applies a first directional
chain rule to those complete functions and to geometry, so true primary jets
are not replaced by independent p/l/S/B when differentiating composites.
The auxiliary rows retain both advective derivative and true shift-gradient
stretching, then add dF and -lambda*(G-dx). The lapse row uses true d(alpha).

The switch uses the prescribed z0=0.1, z1=0.5 plateaus. Inside, it evaluates the
logistic of 1/(1-u)-1/u using a sign-dependent stable exponential. Both plateaus
return exactly constant values and zero directional derivative. No old gauge
switch or prescribed-source theorem is imported. The caller supplies coordinate
lambda, eta, kappa separately. Lapse-scaled lambda=alpha*gamma_R can be passed,
but no parameter parsing, grid timestep, or damping-policy integration exists.

The fundamental point kernel contains no division by w, rho or alpha. It uses
chart exponentials and the smooth switch. Geometric derivatives are evaluated
by forward automatic differentiation. This is a continuum point-jet kernel,
not a discrete chain-rule guarantee: materialized-source versus analytic FD
chain rules, nonlinear RK defects, KO and interface injections still require
explicit measurement when integrating it into the solver.

CPU checks compare all rows, configuration values and three directional jets
with the pinned Python implementation at 24 nontrivial off-reduction states,
and all entries of eight 50x50 principal matrices at oblique normals. The 432
compiled points pass a frozen 2e-11 normalized tolerance; maximum source/jet
error is 1.73e-15. A separate 128-case physical-metric Ricci/Hessian/Codazzi
oracle compares K,C,Ahat[5],Z[3] on reduction with nonzero GH fields and passes
2e-12 at 4.52e-16. Its physical inverse powers occur only in the independent
finite-radius oracle, not in the candidate kernel.

CUDA compilation and execution completed, but the original combined diagnostic
returns NaNs throughout the nonconstant outputs. This is FAIL. Compute Sanitizer
memcheck reports zero errors; this does not rule out all undefined behavior or
code-generation problems. A smaller two-kernel probe and a combined diagnostic
with extra before/after stores produce finite sampled values. The instrumented
combined output agrees with CPU on all 432 points to 1.28e-15. The smaller probe
has a discrepant K row despite finite output. These diagnostic outcomes do not
resolve the original failure and are not grounds to promote the CUDA kernel.
The source equations were not changed to fit a CUDA result.

All remote processes for this checkpoint are terminal. Source snapshot and
original executable remain at
`/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-rhs-001`.
Original source tar/manifest predate additional probe/debug CMake targets.
The original binary is `build/intrinsic_rhs`; probes are separate executables.
Next action: isolate the diagnostic-layout dependence under controlled compiler
and initialization checks, preserving original and instrumented outputs. Do not
claim a compiler bug solely from the present observations.

Reproduce the CPU check with the standalone CMake directory used by the map,
then run `check_intrinsic_rhs.py --binary ... --reference .../candidate.py
--output ...` and `check_intrinsic_physical.py --binary ... --reference-dir ...
--output ...`. The reference path must point to pinned 7ef9c61c. The scripts do
not invoke the reference modules' mains or write their result directories.
`check_intrinsic_backend.py` compares downloaded GPU output against the frozen
CPU input/reference, requiring identical input hashes. Optional CMake targets
`intrinsic_probe` and `intrinsic_debug` retain the CUDA diagnostic experiments.

No evolution mode, restart layout reader, grid operator or physical campaign
uses this new kernel yet. Legacy equations, projections and task timing remain
unchanged by this checkpoint. Full subsidiary/characteristic/Fourier and coupled
numerical qualification remains required; matching matrices is not itself a
bounded-projector or finite-time stability proof.
