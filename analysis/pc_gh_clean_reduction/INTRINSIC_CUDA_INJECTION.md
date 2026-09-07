# Integrated CUDA and semidiscrete injection checkpoint

This checkpoint changes analysis/evidence only. The production equations and
operators remain those in faec722ce9c8db95d4cb3454953fd2d1e636e203.
All 337 remote production-source manifest entries were compared with that local
source after the repeated-restart parser correction. Kokkos is pinned to
08ceff92bcf3a828844480bc1e6137eb74028517; all 1455 tracked dependency files were
verified before the build. The CUDA Release build enables MPI, AMPERE80 and
Kokkos bounds checks, using nvhpc 24.11/CUDA 12.6.

## Executed integrated tests

The dedicated remote root is
`/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001`.
The original and parser-corrected binaries are retained there, respectively:

* `before-repeat-fix/athena`: SHA256
  `b495cb95b552b2d49b502247997d05101b3bf67e9f441242e66b18373e753233`.
* `build/src/athena`: SHA256
  `f68e44a6a6c8df449dc3bcb194fc672fd807c1a02089b5e0ddfc20d42b1e5aa1`.

The original controller and parser-fix follow-up both exited zero. FD2/4/6,
2D/3D, anisotropic uniform periodic meshes pass the independent nonlinear
FD/KO/RK3 oracle at maximum normalized error 1.111e-16. The corrected binary
also passes exact Minkowski, restart and 14 unsupported-path/output controls.
The stronger synthetic restart assigns an independent oblique mode to each of
the 50 fields. All six cases have bitwise identical serial/multiple-block and
two-rank fields and ghosts at cycles 0, 1, 2, 3; both repeated restart and
two-to-one-rank restart continuations are bitwise equal. These are two processes
sharing one A100 40 GB, **not** a multi-GPU test.

`compare_intrinsic_backends.py` exports float64 payloads with original restart
hashes and compares all stored cells, including every ghost layer and corner,
after checking identical block locations. CPU/CUDA maximum normalized difference
is 3.330396224302806e-16 over 24 seeded snapshots (tolerance 2e-12).
The raw compressed export remains outside Git under
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/intrinsic-mesh-cuda-results-001/`.
The committed `intrinsic-mesh-cuda-001/final-artifact-manifest-001.json` inventories
788 remote files, including restart and binary hashes. Compact JSON, controller
logs, inputs and build configuration are committed alongside it. Runtime timings
are retained per process in `runs.json` and in evolution logs.

## Sanitizer negative evidence and isolation

The same corrected binary completed the three-step, eight-block seeded 3D FD6
case under Compute Sanitizer, but default MPI initialization reported eleven
CUDA_ERROR_INVALID_CONTEXT API errors in UCX (exit 99). Changing only
`OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp` reduced this to one error in
the UCX OSC initialization path, also exit 99. Both full logs are preserved.
Adding `OMPI_MCA_osc=pt2pt` gives exit zero and `ERROR SUMMARY: 0 errors` for
the same binary/input/restart. This establishes a clean single-rank kernel
memory check with that MPI configuration; the default UCX sanitizer check
remains failed. It is not a clean sanitizer result for multi-rank GPU transfers.
All arms use `compute-sanitizer --tool memcheck --error-exitcode 99`, bounded
by a 300-second timeout. Available device memory before this check was
36937 MiB; no unrelated process was modified.

## Nonzero semidiscrete reduction/curl injection

`check_intrinsic_injection.py` calls the actual compiled PointRHS on smooth
off-reduction states and independently assembles periodic centered FD and KO
operators on the complete global array. It does not read production `u_rhs`
or measure individual task/stage operation increments. The production mesh
consumer is independently covered by the nonlinear mesh oracle above.

For phi=(w,rho*w,s,beta), E=G-Dphi, and Omega=dG, it records signed vectors

    I = Gdot - D(phidot) - Lie_beta(E) + lambda E - KO(E),
    J = d(Gdot) - Lie_beta(Omega) + lambda Omega
        + d(lambda) wedge E - KO(Omega).

Here phidot_alpha=rho*wdot+w*rhodot includes the actual tangent of the nonlinear
lapse, with KO included in both primary derivatives. It also retains
rho*Dw+w*Drho-D(rho*w). These defects need not vanish: centered differences do
not obey the continuum product/chain rules. All ten family component maxima
and full signed arrays are retained. The family ordering is w,alpha,s[5],beta[3].

The frozen fixture uses a 2D periodic oblique trigonometric field, amplitude
0.02, background (w,rho)=(0.6,1.4), lambda=alpha, eta=2, kappa=1, anisotropic
spacing, N=16/32/64, and KO=0/0.3. All six order/KO arms pass the predeclared
minimum observed order of FD order minus one. Measured ranges across both
refinement pairs and both KO values are:

| FD order | Reduction injection order | Curl injection order | Lapse chain/product order |
|---|---|---|---|
| 2 | 1.951–1.988 | 1.902–1.981 | 1.958–1.969 |
| 4 | 3.900–3.975 | 3.871–3.973 | 3.916–3.959 |
| 6 | 5.830–5.958 | 5.824–5.962 | 5.868–5.947 |

At FD6/N64/KO0.3, max |I|=1.66325e-9 and max |J|=3.26872e-10.
This is smooth snapshot truncation convergence, not physical evolution,
temporal convergence, exact reduction preservation, or interface stability.
The compiled kernel sources were verified unchanged against the archived
base-geometry manifest; the exact tested CPU executable is retained and hashed
in `intrinsic-injection-001/manifest.json`.

## Reproduction and remaining work

Use the archived build/test controller scripts for the integrated tests.
After a seeded decomposition run, export/compare with:

```sh
python3 -W error analysis/pc_gh_clean_reduction/compare_intrinsic_backends.py \
  --root CUDA_SEEDED_RUN --export states.npz --output export.json
python3 -W error analysis/pc_gh_clean_reduction/compare_intrinsic_backends.py \
  --root CPU_SEEDED_RUN --compare states.npz --output comparison.json
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_injection.py \
  --binary BUILD/intrinsic_rhs --output NEW_INJECTION_RUN
```

Gate 1 is still partial: intrinsic refinement, physical boundary transfer and
core reconstruction are absent. Production physical diagnostics and signed
stage/operation budgets must be integrated before serious evolution. Gate 2
smooth curved/forced evolution and independent space/time ladders remain open;
puncture and binary gates have not been promoted.
