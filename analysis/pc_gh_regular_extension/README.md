# Regular PC-GH reduction research harness

Read [the derivation and status](../../docs/pc_gh_regular_extension.md) before using
this option. Defaults preserve the legacy equations. The candidate is not yet
qualified by CUDA puncture or binary evolution.

The optional `reduction_profile=smooth_core` candidate uses bounded finite
inner relaxation, with fixed physical `reduction_core_radius` and
`reduction_taper_radius`. `reduction_inner_rate` is the core rate; the existing
`reduction_rate` remains the outer rate. `reduction_follow_trackers=true` advances
3D black-hole mask centers with the field RK stages. Defaults retain the constant
profile. Read the exact subsidiary/curl and tracker qualifications in the
derivation before using this option.

`make_smooth_controls.py QUALIFICATION_DIRECTORY NEW_DIRECTORY` prepares the
fixed/moving pulse, gauge-wave, oracle and single-hole inputs. The smooth pulse
verifier uses independent characteristic quadrature and the uncontracted curl
columns added to `pulse.hpp`; it does not fit a single rate to a variable-rate
packet. `verify_moving_wave.py` compares both tracker paths with the analytic
normal flow of the shifted gauge wave. Build and run numerical qualifications
on Della CUDA; the local smooth build recorded in this session is compilation
validation only.

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

`read_complete_history.py` preserves all restart segments and reports duplicate
timestamps and superseded overlap. `cuda_driver.py --restart-from CHECKPOINT`
starts a distinct, documented experiment; `--resume` continues an unchanged
experiment. Research source snapshots include new untracked problem generators.

`exterior_self_convergence.py` compares three matched Cartesian exterior samples;
`plot_native_core.py` separately compares actual closest cells and radial maxima.
`summarize_amr.py` retains all 33 pulse components and the bracketed norm changes.
`decompose_amr_pulse.py` supplements the total error with seeded/generated family
contributions. Full 3D component CSV files remain on Della to avoid duplicating
many gigabytes during routine analysis.

`frozen_reduction_source.py` implements the independently differentiated nonlinear
configuration source. `sample_restart_sources.py` uses the verified vacuum PC-GH
checkpoint ABI, with explicit second/fourth/sixth-order stencils; its samples are
not global bounds. Build `--kind subsidiary` for `nonlinear_reduction_oracle.cpp`.
Run its zero-step input at multiple physical spacings and pass the output run
directories to `verify_nonlinear_reductions.py --output REPORT.json`. All 33
reduction rows are checked on off-GH/off-reduction/nonzero-curl jets in all three
gauge-switch regions. The independently differentiated configuration RHS has
second-order truncation error, which must converge without an adjusted threshold.

Current results include stable second-order single-puncture evolutions to 20M,
declining exterior/AMR errors, and failure of the sixth-order puncture controls
near 8M, including uniform and smaller-step controls. The merger gate remains
closed while these failures are investigated; see the qualification log for
the numerical values and the unresolved puncture limits.
