# Intrinsic finite-difference and RK3/KO checkpoint

The new reusable `FiniteDifferenceRHS` in
src/pc_gh/intrinsic_finite_difference.hpp reads all 50 intrinsic fields, forms
true derivatives using AthenaK's existing Dx<2/3/4>, calls the complete intrinsic
point RHS, and adds the existing normalized KO operator to all 50 rows.
It reads only active spatial directions. The caller must provide synchronized
stencil data, dimensions 1 through 3, and a nonnegative KO amplitude. This
routine does not project fields or implement boundaries, MPI exchange, restart
conversion or timestep selection. It is not yet called by production mesh tasks.

For spatial order 2r, the centered modified wave number is
2/h times the sum of c_j sin(j theta), with coefficients
(1/2), (2/3,-1/12), or (3/4,-3/20,1/60). For p=r+1 the KO symbol is
q=-epsilon sum_d sin(theta_d/2)^(2p)/h_d. The C++ consumer uses the unchanged
Diss template and sign (-1)^(p+1)/2^(2p) normalization from pc_gh.cpp.
No legacy operator or task timing changed.

## Compiled stencil evidence

The actual new consumer is compiled on CPU and A100. Plane-wave perturbations
in every state column, two phases and two signs produce the full complex
linearized operator. The 60 cases cover FD2/4/6, dimensions 2/3, spacings
(0.125,0.2,0.3), KO 0/0.3 and five mode scales. In 2D, the unused third-direction
phase is deliberately nonzero, so unintended derivative reads are detectable.
These are infinite-grid plane-wave functors, not populated halo arrays.

Both backends pass all 12000 point calls. The maximum normalized matrix error
against J+i*sum(k_eff*P)+q*I is 3.259e-11; the frozen finite-amplitude tolerance
is 2e-8, accounting for nonlinear centered-difference and roundoff errors.
Identical-input backend disagreement is 6.662e-16. The base J/P matrices are
those already extracted from the compiled point kernel and independently
verified by the exact Minkowski oracle; their archive hash is recorded.

## Coupled time integration

For the same 60 operators, the three driver.cpp SSPRK3 stages agree with
I+dt*A+(dt*A)^2/2+(dt*A)^3/6. The timestep is 0.2*min(h)/sqrt(2), matching the
flat maximum characteristic speed for this fixture, not an intrinsic-mode
production timestep policy. Full 50x50 matrix powers at common t=0.5 and three
timesteps agree with exp(t*A) with minimum observed order 2.9721. Sampled
spectral radii from the exact Fourier roots with modified k and KO shift are
at most 1. The discrete reduction closure C*RK=R(dt*(q-lambda))*C holds to
1.462e-16 normalized. Full propagator norms are retained in cases.json.

The RK analysis uses the verified J/P-based discrete operator rather than
amplifying finite-amplitude stencil-estimation noise. This is a coupled uniform
Fourier check; it is not a separate scalar CFL argument or a compiled mesh
RK-stage test. It retains the known homogeneous Jordan drift: radius one does
not mean that primary errors cannot grow algebraically. Uniform KO commutes
with the reduction derivative here. Nonconforming transfer, physical boundaries,
variable fields/rates and nonlinear chain-rule defects remain unqualified.

## Reproduction

Configure analysis/pc_gh_clean_reduction/compiled with a Kokkos installation and
ATHENA_CONFIG_DIR pointing to an existing compatible AthenaK build directory.
Build target intrinsic_stencil. The first attempted build preceded CMake
regeneration and had no target; that setup error log is retained. Regeneration
and both CPU/CUDA builds then succeeded.

```sh
python -W error analysis/pc_gh_clean_reduction/check_intrinsic_stencil.py \
  --binary /absolute/path/to/intrinsic_stencil \
  --matrices /absolute/path/to/verified-fourier-matrices.npz \
  --output /absolute/path/to/new-output-directory
```

The matrix archive is the previous Fourier check's `matrices.npz` (J0/P0).
`--run-only` saves the compiled batch without local exponential analysis.
Replay on Linux adds `--replay /absolute/path/to/saved-batch`; it requires
identical regenerated input bytes and records the saved binary hash without
executing the supplied binary. The CPU batch took 0.975 seconds; Linux analysis
of that saved batch took 1.694 seconds. A100 execution plus analysis took
2.411 seconds. Linux numpy/scipy avoid the previously recorded local expm
warning; all analysis runs use warnings-as-errors and one CPU thread.

Compact results, full case measurements, build logs/configurations, source
manifest, backend comparison, temporal-order plot and hashed external raw
inventories are in qualification-runs-20260907/pcgh-clean-reduction/intrinsic-stencil-001/.
The remote source hashes were verified before running the new executable.

Next: connect the intrinsic layout to mesh storage/tasks with explicit restart
identity and coherent 50-field transfers, then qualify actual mesh stages and
interfaces before smooth/forced and puncture evolution gates.
