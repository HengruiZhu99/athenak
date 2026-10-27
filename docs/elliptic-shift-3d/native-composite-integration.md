# Experimental native composite Task integration

This opt-in path runs the full 3D composite tensor engine through the actual AthenaK Driver and Task callbacks. It is an initial vacuum Minkowski integration milestone. Shell cells and their multigrid hierarchy remain global host-owned engine state, rather than native MeshBlocks. It does not complete native distributed, device, AMR, wormhole, or late-time validation.

## State and time integration

`NativeCompositeEvolution` owns `CompositeTensorEvolution`, `NativeCompositeStageState`, and `NativeCompositeFields`. The native core block has physical conformal fields in `u0`; the canonical global coefficients include the Cartesian core, configurable cubed-sphere layers, scri surface, independent Lambda, and telegrapher B. The owner checks the entire native mirror and gathers active fields through the field adapter before every stage. External native changes fail explicitly rather than being erased on scatter.

The opt-in Task queue calls BeginStage, full global RHS, AdvanceStage, physical ADM conversion, full-grid timestep selection, and independent constraint diagnostics. RK coefficients come from the live Driver; `CompositeTensorEvolution::Step` is never called by the native owner. Beta is solved before each RHS and accepted output, and is excluded from RK register mixing. All 22 coefficients, including independent Lambda, the six metric components and B, evolve freely. The queue bypasses the Cauchy lapse/shift evolution, floors, algebraic constraint projection and Cauchy boundary routines. Default Cauchy behavior retains its original queue.

The native core plus all its ghosts use `ToPhysicalADM`: psi4=1/(Omega² chi), physical lapse=alpha/Omega, physical metric=psi4 tilde-g, and physical curvature=psi4[Omega Atilde+tilde-g(Khat+2Theta)/3]. Conformal and physical ADM gauges have separate storage. No Omega floor is used. Scri itself is an explicit Omega=0 engine surface and uses the existing tensor limiting closure and prescribed lapse clock.

## Configuration and current restrictions

See `inputs/z4c/composite_minkowski.athinput`. Enable `z4c/composite_elliptic_experimental=true` with the dedicated `problem/pgen_name=z4c_composite_minkowski`. `composite_radii=.65,1` configures two shell layers; one or more layers are accepted within the bounded cell/coarse-solve limits. `composite_initial_lapse_amplitude` defaults to zero and has magnitude at most 1e-3. `composite_initial_angular=true` adds the engine's nonspherical lapse perturbation on a new run only. Restarts restore state and bind the original pulse metadata. `composite_output_every` limits full CSV snapshots while retaining every-cycle diagnostics.

The owner requires double precision, Kokkos Serial, one rank, one static cubic native Cartesian block, vacuum, RK3/RK4, order4, chi_power=-4, separate ADM gauge, and ng=3. This is the native order4 ghost contract; research drivers used ng=2, with the same differentiation stencil where two or more ghosts exist. Native MPI, GPU/device execution, multiple native core blocks, native AMR/regridding, matter, trackers, extraction/CCE, floors, other initial slices, old hyperboloidal/cubed-sphere modes, and native midstage restart are rejected. The native AMR negative control encounters the pre-existing odd-ghost AMR guard before the owner. The owner bounds cells at 50,000 and rejects incompatible native core/ghost geometry.

## Timestep selection

`composite_dt` is a ceiling, not a timestep guarantee. The owner evaluates accepted-state geometry without advancing it (`RHS(...,0)`), then takes the minimum over every global active cell of c/(N v), with c=min(native cfl_number,0.4). The characteristic speed is exactly the engine's sum over mapped directions of |J beta| + sqrt[chi (J tilde-g-inverse J) (alpha²+4 Omega² alpha+telegraph support²/4)]. It also includes every scri radial-spacing penalty limit c h/(pole_sat+regularity_sat) and the telegrapher relaxation limit 0.004. Native Mesh timestep multiplication by its CFL number is accounted for once. Each advancing stage retains the engine's full-grid speed, scri and relaxation guards; a stage that violates them fails explicitly.

## Accepted-cycle restart

Native restart files have a paired `.composite` sidecar containing global volume/scri coefficients, all RK registers, algebraic-shift warm start, live Driver RK arrays, pulse/chart/mesh metadata and cycle/time state, with an FNV64 integrity checksum. Restore rebuilds solver hierarchies, invalidates the solved shift, solves again, and verifies the native physical mirror against the sidecar. Full decoded values and registers, not merely output checksums, are compared in tests. A checksum is integrity protection rather than authentication. This first format uses local binary ABI/endian representation, supports accepted-cycle checkpoints only, and stores the sidecar's absolute path in the native restart header. Relocation requires overriding `z4c/composite_restart_file`; source/binary identity is recorded in external manifests rather than enforced by the local checkpoint tag.

## Reproduction and evidence

From the repository root:

```sh
python3 tst/elliptic_shift_3d/build_native_compatibility.py --output NEW_BUILD_DIRECTORY
python3 tst/elliptic_shift_3d/run_native_composite_smoke.py --extended-controls --output NEW_CONTROL_DIRECTORY
python3 tst/elliptic_shift_3d/analyze_native_composite.py NEW_CONTROL_DIRECTORY
```

The runner allows at most 180 seconds per owned process and two build workers, and preserves stdout/stderr. It never modifies the original 1D checkout or signals its jobs. The default test input runs two RK4 cycles at dt=1e-4 on N6 with one/two shells. Nonzero radial and nonspherical pulses use amplitude1e-5; an extended equilibrium control runs 40 cycles to t=.02 at dt=.0005. These are affordable integration and short-time scientific controls, not convergence or late-time black-hole results.

The pre-CFL implementation and all15 controls are frozen under `evidence/native-composite-pulse-build-20261027` and `evidence/native-composite-pulse-controls-20261027`. All expected controls passed, including bitwise full volume/scri and decoded full-register restart equivalence for equilibrium and both pulses. Extended equilibrium H_RMS=2.01749e-15, M_RMS=4.79605e-15, maximum state deviation=1.14366e-14, original shift RMS<=1.65458e-13, and scri tensor pole<=8.97125e-14. Both pulse cases produce nonzero B, off-diagonal metric, independent Lambda and Z. Independent native BIN physical-ADM comparisons include all ghosts and agree within float32 output precision. Negative controls retain ADM alias, native AMR, corrupted sidecar, edited native restart mirror, changed RK and changed pulse metadata failures.

The post-CFL build is frozen under `evidence/native-composite-cfl-build-20261027`: actual athena target build passed in7.009s, athena -h exited0, and all413 compiled-source files match the archived hashes. All15 post-CFL expected controls passed, with extended equilibrium42.394s. All34 complete volume/scri CSV snapshots across equilibrium, pulses and extended equilibrium are bitwise identical to their pre-CFL counterparts. A separate full-grid selection control on the normal geometry with native CFL=.001 and ceiling=.0005 selected4.86111111111111e-6 and advanced two cycles to9.72222222222222e-6; independent original-equation and boundary checks pass. Its controls are under `evidence/native-composite-cfl-controls-20261027`; extra timestep/default-Cauchy controls and deliberate failures are under `evidence/native-composite-cfl-extra-controls-20261027`. A deliberately thin-shell .98,1 stress case failed its initial elliptic solve at240 iterations, original RMS7.15201e-10 against1e-11, before time advancement. This unsupported numerical case is preserved and is not claimed to validate thin-shell CFL selection. A stale gauge-wave input with an unregistered pgen is retained separately as a failed test attempt; the registered default Cauchy linear-wave8³ two-cycle control passed with opt-in disabled.

The first native output attempt wrote initial diagnostics before the native main changed into its output directory. That IO-order bug was corrected by deferring initial diagnostic emission to Driver initialization, and the initial artifacts/failure provenance remain under the first smoke evidence. No equation or tolerance was changed to repair it.

## Remaining scientific and architecture gates

The engine still uses a host canonical state and engine-owned shells. Native shell MeshBlocks, general native AMR/distribution/device operation, restart/regridding at those capabilities, wormhole initialization on this native path, and coupled black-hole accuracy remain open. The separate wormhole research pilots retain failed scri/constraint gates. The native equilibrium/pulse results do not establish wormhole regularity, late-time front convergence, scalable performance, or general moving-puncture support. No spherical projection, beta puncture pin, lapse/chi floor, or attenuated 1D source has been imported.
