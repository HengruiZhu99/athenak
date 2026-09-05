# Regular PC-GH reduction research harness

Read [the derivation and status](../../docs/pc_gh_regular_extension.md) before using
this option. Defaults preserve the legacy equations. The candidate is not yet
qualified by CUDA puncture or binary evolution.

Local symbolic/zero-step reproduction (Python with SymPy/NumPy):

```sh
cmake -S . -B build-regular-extension -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=../../analysis/pc_gh_regular_extension/production_oracle \
  -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF
cmake --build build-regular-extension -j8
```

Run `build-regular-extension/src/athena -i ABSOLUTE_PATH/oracle.athinput` from a new
evidence directory. Then run `run_symbolic.py NEW_OUTPUT --production EVIDENCE`.
The original independent 4D/Ricci suite remains `analysis/pc_gh_symbolic/run_all.py`.
These operations do not evolve test spacetimes locally.

Della deployment must use a fresh source/scratch location. Keep the old binary and
run trees untouched. Include the pinned Kokkos submodule and preserve the source
commit, working diff, and submodule hash. The earlier environment is
`/home/hz0693/athenak_env`; the existing Linux TwoPunctures installation can be
located in `/home/hz0693/TwoPuncturesC`. Do not copy the macOS static library into
a Linux build.

```sh
source /home/hz0693/athenak_env
python3 analysis/pc_gh_regular_extension/cuda_driver.py build \
  --kind oracle --build build-regular-cuda
python3 analysis/pc_gh_regular_extension/make_inputs.py NEW_INPUT_DIRECTORY
python3 analysis/pc_gh_regular_extension/cuda_driver.py run \
  --build build-regular-cuda --input INPUT --output UNIQUE_SCRATCH_RUN
```

The driver records GPU information, executable/input hashes, commit/diff, full
CMake cache, exact argument arrays, and each segment log. The custom problem
generator checks the actual execution space is CUDA. No result is labelled a
physics pass merely because a process exits cleanly. A direct vis1 run can use
`--wall-segment 00:55:00`; the Slurm template uses 15-minute segments inside the
previously used 20-minute `gpu-test` allocation. Inspect current Slurm availability
before submission. A Slurm wall stop may require `--resume` in a later allocation.

Build `--kind z4c` for the existing single-puncture control and `--kind binary` for
the TwoPunctures binary. The existing Z4c single-puncture pgen reinitializes on
restart: the runner refuses to silently continue it. This control needs a complete
allocation or an independently validated pgen restart fix. The PC-GH single and
binary restart paths retain the existing production implementation.

Qualification order and interpretation:

1. CUDA principal oracle, exact Minkowski, independent compact p/Q/L/B pulses.
   `verify_pulses.py RUN...` records all-component continuum errors, fitted decay,
   centroid speed, amplitude and domain bounds. Compare zero/nonzero rates, three
   resolutions, two derivative directions, finite-amplitude and stiff-rate controls.
2. Three-resolution shifted harmonic-wave tests with legacy, new zero-rate, and
   new nonzero-rate equations. These test consistency; they do not prove the new
   harmonic-gauge off-constraint hyperbolicity.
3. Isotropic one-puncture controls at dx=1/8,1/10,1/12M, both uniform and SMR, using
   matched RK4/CFL=0.1/KO=0.3. The uniform 16M domain is a finite-boundary control;
   the larger SMR domain separates boundary effects before any exterior claim.
   Compare regular field powers with resolution, exterior constraints/solutions,
   positivity, symmetry, and reduction/curl maxima. The inputs provide frequent
   restart/slice output. Inspect the saved Z4c and projected-PC-GH controls too.
4. Three-dimensional compact pulses cross a fixed refinement interface; compare
   with uniform controls and the operation-resolved injection logs. Intermediate
   ghost-fill norms must not be mistaken for synchronized continuum errors.
5. Only after stability, convergence, and useful damping are demonstrated, run
   the established 128M-domain adaptive head-on case through 73.8M toward 100M.
   The new primary input disables both GH and reduction projection, uses kappa=1,
   lambda=1, and unchanged KO=0.3. Matched kappa controls are necessary to attribute
   improvements. The saved old projected kappa=0 baseline is not a one-parameter
   damping comparison.

SSH access was restored through the user's multiplexed connection. CUDA oracle,
flat, compact-pulse and shifted-wave gates have now been run; see the dated
qualification log. Puncture screening and controls are in progress. None of these
early gates establishes puncture or merger qualification.

`run_collection.py` runs only an explicitly selected input group and stops on a
failed process. `summarize_pulses.py` and `plot_cuda_pulses.py` aggregate the pulse
resolution ladder, retaining the underresolved coarse points. Native puncture
profiles use `analyze_native_puncture.py`: no Cartesian interpolation or nonfinite
mask is used, and the actual slice cell-center offset is retained. Native binary
fields have float32 output precision; analysis promotes them to float64. Full
volume bounds and operation maxima remain the production double-precision logs.
