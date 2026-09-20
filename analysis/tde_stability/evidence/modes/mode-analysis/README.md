# Full-step discrete vacuum mode experiment

This experiment identifies growing modes of the **implemented complete RK3 map** for a stationary vacuum Schwarzschild R0=M trumpet. It does not establish continuum ill-posedness or a stable production configuration.

## Isolation and reproducibility

- Detached worktree: `/Users/hz0693/research/TDE/athenak-mode-analysis`, based on `1b93538a`.
- Only source change: `src/main.cpp`, saved as `isolated-hook.patch`.
- The hook imports all 25 residual fields, including physical ghost cells, **after** ordinary initialization and refreshes full metric/ADM caches. It then executes the unmodified task graph, stencils, RK, boundary closure and projection. No state clipping or reset occurs between RK steps.
- Raw shape: 25 x 24 x 24 x 24 double values; active indices [4:20] in each spatial direction. B auxiliaries remain zero because the telegraph option is inactive.
- Immutable baseline binary: `athena-mode-analysis`, SHA256 `1aa1c3cae182341469eff2239fa79123b9fbf3227ad37327bf7f9f45d544e0a2`.
- No production files, Aurora jobs or queues were changed. The hook was subsequently also applied, at root's request, to the independent lapse-damping experiment worktree.

Ordinary checkpoint reload was deliberately not used as the map: initialization re-extrapolates ghosts, whereas the preceding RK step ends with projection after ghost extrapolation. These operations need not commute. The complete-state hook avoids adding an extra boundary operator.

`mode_operator.py` implements subprocess-based maps and central finite differences, retains call provenance and rejects nonfinite output/invalid metric diagnostics or wrong cycle/time/dt. `validate_operator.py`, `calibrate.py`, `arnoldi_modes.py`, `validate_modes.py`, `validate_linear.py`, and `plot_modes.py` reproduce the checks and figures. Use the Codex dependency Python with NumPy/SciPy/Matplotlib and `OPENBLAS_NUM_THREADS=1`; each map uses OMP2.

## Scheme and gate checks

Single 16^3 block, box [-2,2]^3 M, dx=.25M, sixth-order spatial stencil, RK3, dt=.075M, adapted gauge f=1/G=2, kappa1=.1/kappa2=0, KO=.5, characteristic zero-rate RHS closure; original cubic residual ghost extrapolation. Matter feedback, inner excision and outer sponge are disabled. All experimental operator flags are off.

- Zero stays bitwise zero through two full RK steps, including all fields and ghosts.
- Fresh pulse: F(F(v)) versus F^2(v) is bitwise identical.
- Rescaled late-control state: F^2(F^2(v)) versus F^4(v) is bitwise identical.
- Calibration is necessary: peak input 1e-6 gave ~8e-4 full-state response sensitivity, dominated by high-order ghost extrapolation of floating-point cancellation. Peak 1e-3 central differences converge to a few parts per million when varied to3e-3 and3e-4. The active-state differences are substantially smaller.

The Arnoldi seed is the finite t=500.025M checkpoint from the prior G=2 run, used only as a direction and rescaled before every map. Its original full perturbation is never evolved directly by this experiment. Both signs are used to remove leading even nonlinear response.

## Measured growing modes

A 32-vector Arnoldi basis of the 40-step/3M finite-difference map found the following two independently verified real modes:

| Cubic ghost mode | 3M multiplier | growth/M | e-fold time/M | direct 3M relative residual | one-step relative residual |
|---|---:|---:|---:|---:|---:|
| faster angular mode | 1.149643599 | .046483994 |21.5128|9.7e-7|5.2e-7|
| mode matching late control |1.113160363|.035734381|27.9843|2.2e-6|5.5e-7|

An uninterrupted nonlinear 100-step/7.5M evolution at input peak1e-4 matches predicted shape and amplification to3.1e-5 and7.0e-5 relative. Active-cell one-step Rayleigh rates reproduce both growth rates; the result is not solely a ghost-inclusive norm artifact. Halving dt to.0375 with the independent experimental binary's damping flag off preserves these modes (see `scaled-old-mode-response.json`).

Active Theta peaks in the innermost puncture cells; active metric components are strongest toward outer cube edges/corners. Fourth ghost corners have much larger extrapolated amplitudes. These are **global discrete eigenmode shapes**, not evidence that the physical instability was first injected at whichever field maximum is largest. `growing-mode-slices.png` and `.pdf` show the active slice z=.125M; color amplitudes use arbitrary eigenvector normalization, and the dashed circle is the horizon section.

The first-stage active Euclidean projection of the RHS onto each mode gives:

| contribution /M | faster | slower |
|---|---:|---:|
| volume | .04886413 | .03687452 |
| KO | -.00185364 | -.00062520 |
| characteristic correction | -.00115199 | -.00051203 |
| algebraic projection increment /dt |+.00062550|-.00000287|

Their sums reproduce the measured growth to the expected RK difference. The positive volume contribution and negative instantaneous boundary correction do **not** prove that boundaries are irrelevant: the complete mode and volume stencil inputs depend on the boundary operator.

## Matched linear ghosts

Changing only `z4c/extrap_order=2` retains almost identical active mode shapes (absolute cosine .99989 and .99984):

| mode | cubic growth/M | linear growth/M | reduction |
|---|---:|---:|---:|
| faster |.0464840|.0455908|1.9%|
| slower |.0357344|.0318548|10.9%|

Direct eigen-residuals are9.3e-7 and2.1e-5; one-step checks and input-amplitude convergence agree. Thus linear ghosts reduce amplification, but do not cure this vacuum instability. This is a matched G=2 operator experiment, separate from the earlier G=1 linear-ghost evolution tests.

## Limits

This is a validated finite-difference approximation to the complete discrete tangent map, not automatic differentiation. Only one block, one CPU process and one resolution/domain are included. The Krylov subspace is finite and seed-dependent; it does not establish the entire spectrum, nor does absence of a positive Ritz value certify stability. No conclusion about SMR/AMR, MPI/GPU or physical-star stability follows. The extracted modes are useful stronger regression perturbations for candidate repairs, alongside fresh lapse pulses and exact vacuum tests. Spin backgrounds remain gated on passing vacuum stability.

## Signed physical constraints

`athena-constraint-probe` (SHA256 `5ea4a8d1dba9b294b96881992c91aa62d0a59bd343fb43e0fd7cea5dd8903b56`) adds only an export after the map: refresh ADM, run existing `ADMConstraints<4>`, write double `u_con`. It leaves the residual state bitwise unchanged, including nlim=0 imports. `measure_constraints.py` evaluates central ±mode differences, so the fixed background Hamiltonian/momentum discretization error cancels. H and all covariant M components are signed, not norms squared. The conformal connection constraint linearization is Q^i = deltaGamma^i - d_j h_ij + .5 d_i tr(h), using the same sixth-order centered derivative and the conformally flat background. Vector RMS values use unweighted coordinate components and are not invariant integrated norms.

All three physical perturbations grow with each mode: H/M/Q multipliers match1.1496436 or1.1131604 to shape residual<=1.1e-5. Input amplitudes1e-3 and1e-4 give H derivatives agreeing to1.7e-8 relative, M to5.1e-10. Thus Theta growth accompanies signed physical Hamiltonian, momentum and connection-constraint growth. The faster H peak is near r=.545M and M peak near r=.217M, while Q peaks near an outer face. The slower H/Q are strongest near the outer boundary; its M peak is inside the horizon. See `physical-constraint-modes.json`, `physical-constraint-calibration.json`, and the compact NPZ arrays.

## Lapse-scaled damping follow-up

At root's request, the same complete-state hook was applied to the separate `athenak-lapse-damping` worktree. Its immutable binary is `athena-lapse-scaled`, SHA256 `5be3094897d5edd5106aa6cadc1faa5a5d25959d1a78ab2cc3774013686374c8`. This changes the lower-order constraint damping product from alpha*kappa1 to kappa1 when the new default-off flag is enabled. The root task owns that source change and its equation/source regression; this directory contains an independent complete-map test.

The default-off binary at half dt=.0375 reproduces baseline active growth rates to~3e-8/M. Enabling the flag and kappa1=.3 reduces but does not remove the two growing modes:

| scaled kappa1=.3 mode | multiplier per3M | growth/M | e-fold/M | direct relative eigen-residual |
|---|---:|---:|---:|---:|
| faster |1.111193723|.035144955|28.4536|1.4e-6|
| slower |1.075794885|.024353272|41.0623|1.6e-6|

These use80steps per map, dt=.0375,32Arnoldi vectors, original cubic ghosts, G=2 and all other baseline conditions. Changing central-difference input peak from1e-3 to3e-4 changes the response by3.3–3.5e-6 relative. Independent one-step active Rayleigh rates agree to2e-8/M. Signed physical H, covariant M, and Q perturbations grow with these eigenvalues too, to<=6e-7 relative shape error. **This configuration therefore fails the vacuum linear-stability gate**, despite reducing Theta/source amplitudes or delaying failure in a lapse-pulse evolution.

See `scaled-validated-modes.json`, `scaled-old-mode-response.json`, `mode-configuration-comparison.png`/`.pdf`. The initial old-mode projection test is retained separately: those shapes deform under the new operator, so its short-time projected gains are not used as new eigenvalues. A preparation-only command-line parameter error was retained in `scaled_k01/mode0_map80_plus_input-option-error`; it was corrected by declaring the new flag in a separate input, and is not an evolution failure.

No full Arnoldi result was computed for scaled kappa1=.1; only response of the two old modes was measured. Do not infer its new spectrum from the kappa1=.3 result or from the old-mode gains.
