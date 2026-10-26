# Cubed sphere exterior for regular Z4c

The Cartesian bulk now has a fixed exterior made of six cubed sphere blocks. The bulk retains its octree and active-cell AMR transfer. All evolution, interpolation and diagnostic kernels use Kokkos device views. This is a finite-radius regular-Z4c verification implementation; the hyperboloidal adapter still rejects this grid. Stable substantial angular hyperboloidal disturbances and a subsequent wormhole-to-trumpet evolution with an immutable Minkowski reference remain the scientific objectives.

The implementation starts from `24616eb8d52060414f234192548abbfc7758efb0` on branch `codex/cubed-sphere-boundary`. The [assessment handoff](hyperboloidal-assessment-handoff-20261010.md) and its failed native evolutions remain relevant. A spherical mesh boundary removes the Cartesian cut sphere and its ghost search, but it does not regularize conformal poles or establish a stable coupled boundary condition.

The [independent refinement and native-failure controls](cubed-sphere-refinement-20261012.md) reproduce the old failure, demonstrate boundary sensitivity, and expose resolution imbalance at the cube/shell interface. An opt-in CPU Serial mapped conformal research probe now preserves one immutable Minkowski reference and the old wave-map/geometric formalism. This separate probe does not extend production hyperboloidal AMR or GPU support.

## Geometry and evolution

The core is the centered cube `[-L,L]^3`. The exterior blocks are ordered `+X,-X,+Y,-Y,+Z,-Z`. Each has logical coordinates `(s,u,v)` in `[0,1] x [-1,1]^2`. For the positive X block,

```
d = (1-s)*L + s*R/sqrt(1+u*u+v*v)
(x,y,z) = (d,d*u,d*v)
```

The other blocks permute axes and change the dominant-axis sign. The inner faces match the Cartesian cube and the outer faces tile the physical sphere of radius `R`. The map requires `R>sqrt(3)*L`, with stricter checks on ghost geometry. This first version uses a simple algebraic map rather than an equiangular or stretched map.

The 25 Z4c fields retain their global Cartesian components. This includes the contracted conformal connection: it is not treated as a vector under a patch coordinate transformation. Only differentiation directions change. The ordinary Z4c RHS, ADM constraints and Weyl calculation use the inverse Jacobian and the full inverse Hessian:

```
partial_a f = J[A,a] D_A f
partial_a partial_b f = H[A,a,b] D_A f + J[A,a] J[B,b] D_A D_B f
```

Upwind selection uses the full logical velocity `J[A,a]*beta[a]`. Dissipation acts in logical directions with an inverse-map scale. The timestep bounds light, lapse and Gamma-driver speeds in each logical direction and includes the adaptive bulk. Changing the input CFL changes the timestep; it is not canceled by another hard-coded limit.

The shell is a separate pack allocated with the ordinary Z4c class. Its temporary Cartesian tree provides allocation only; its neighbor topology is never used. Bulk and shell RHSs read the same RK stage and use the same RK3 weights and times. Exact harmonic-wave outer ghosts use the appropriate stage time.

## Interfaces and AMR

Ghosts use tensor-product six-point Lagrange interpolation from active donor cells. A donor never reads another ghost cell. Exact coincident Cartesian points use direct copy. Subtracting a donor anchor before interpolation preserves constant fields exactly.

The donor plans rebuild when the bulk AMR sequence changes. Regression cases refine and coarsen both the central region and all eight corners touching the shell. Uniform bulk interiors retain the native neighbor exchange. For adaptive runs, the extension fills every fine bulk halo directly from the current leaf hierarchy after native prolongation, and supplies exterior coarse ghosts before prolongation. This resolved invalid newly allocated halos observed in the first coupled refine/coarsen test. The independent ordinary Cartesian AMR regression also passes; this coupling result does not establish a general defect in native AMR.

The shell stays at fixed resolution. There is no shell refinement, MPI donor communication or subcycling. Interpolation accuracy and the observed tests do not provide an energy estimate for the coupled operator. A characteristic penalty or SBP interface may still be required for longer or more demanding evolutions.

## Boundaries and failure capture

`outer_boundary=analytic` supplies exact harmonic gauge-wave or Minkowski ghost data. This is a verification boundary. `outer_boundary=sommerfeld` extrapolates ghosts and replaces the outermost active-cell RHS for Khat, Theta, A and Gamma with the ordinary regular-Z4c Sommerfeld subset, using the physical radial derivative and the existing speeds. It is a finite-radius approximation, with no claim of constraint preservation.

Neither option is a boundary condition at future null infinity. Hyperboloidal evolution needs a regular closure for the full coupled characteristic system at `Omega=0`. Simply placing the existing Sommerfeld condition there would change the problem and leave the singular source terms unresolved.

Every updated RK stage checks finite values, positive lapse and chi, and positive definite conformal metric before algebraic projection. A failure writes `cubed_sphere_first_bad_stage.csv` with the complete local stencil, physical coordinates, input stage, updated state, RK register, RHS, timestep, cycle and stage. A companion file saves the native Jacobian, Hessian and logical spacing, and the resolved parameters are retained. The stream is explicitly closed before termination. The failure regression verifies that all stencil rows are present and that the saved operands reproduce the failed RK update.

## Supported configuration

This version requires one rank, 3D vacuum regular Z4c, RK3, fourth-order spatial differences, `chi_psi_power=-4`, a centered cubic bulk and at least six cells per bulk block axis. It supports harmonic or 1+log lapse and the ordinary Gamma shift. Telegraph lapse, slow-start lapse, harmonic shift, matter, cyclic zoom, extraction, trackers and horizons are rejected. The initial-data entry point is currently the supplied cubed sphere test problem.

Restart input and restart output are rejected because native checkpoints would omit the shell. Ordinary output infrastructure still covers the Cartesian pack; the supplied final CSV includes both regions. Full shell field output and checkpoint support are prerequisites for production use. Enabling the extension does not relax any of the existing hyperboloidal GPU, AMR or single-block guards.

## Running the checks

A CPU build and the four bounded regressions use:

```
cmake -S . -B build-cubed-sphere -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_CUBED_SPHERE_TESTS=ON
cmake --build build-cubed-sphere -j8
ctest --test-dir build-cubed-sphere --output-on-failure
python3 tst/cubed_sphere/check_evolution.py build-cubed-sphere/src/athena \
  --suite convergence --output build-cubed-sphere/Testing/convergence
```

The input example is [inputs/z4c/cubed_sphere.athinput](../inputs/z4c/cubed_sphere.athinput). The regression suites save inputs, logs, metrics, source hashes and an immutable copy of the executable. `l2` and `linf` mean exact-solution errors for Minkowski and the harmonic gauge wave; for the angular pulse they measure deviation from Minkowski, since no exact evolved pulse is supplied.

The scalar suite evolves `u_t+x.grad(u)=0` with an exact outward Gaussian solution through the same donor plans and mapped upwind derivative. It tests refinement at `t=0.1` and passage out of the sphere through `t=4`. Its high-order interpolation is not positivity preserving. Radial characteristics retain their angles, so this does not by itself certify wave-packet traversal of patch seams. The scalar test checks transport consistency and bounded sampled evolution; coupled Z4c and scri need their own validation.

The Aurora GPU runtime build uses oneAPI 2025.3.1, SYCL, PVC architecture, O2 and precise floating-point arithmetic. One-node debug job 8923658 passed all four regressions on a selected Intel Max 1550 GPU tile in 2 minutes 40 seconds, including central and corner bulk AMR. The separate bounds-checked Debug AOT build was stopped after prolonged compilation in unused GRMHD kernels; GPU bounds-check validation is therefore still pending. The installed 2026 compiler failed in bundled Kokkos before reaching the new code. The actual configure commands, device-selection log and bounded one-node PBS script are preserved in [validation evidence](validation/cubed-sphere-20261012/README.md) and [run_aurora.pbs](../tst/cubed_sphere/run_aurora.pbs).

## Validation gates toward the hyperboloidal objective

1. **Accept the regular grid only within demonstrated limits.** Require derivative and exact-wave convergence, constant preservation through actual refine/coarsen events, and device execution with bounds checks. Repeat nontrivial AMR convergence and refine near the cube interface. Add full shell outputs, restart and MPI communication before broader production claims. The compiler reports register spills in the mapped-capable RHS; specialize Cartesian and mapped kernels and measure bulk performance before making throughput claims.
2. **Resolve coupled interface and boundary behavior.** Use small-amplitude disturbances of every Z4c sector, then the substantial angular pulse. Require bounded amplification as resolution increases, decreasing constraints and Weyl curvature, and comparable results with a genuine half timestep. The regular `t=2` pulse completes at shell N=16,24,32, with decreasing Hamiltonian, momentum, Z and Weyl errors; the finest bulk Hamiltonian L2 is still 0.06834 and its maximum is 3.07876. A true half-timestep coarse run changes Hamiltonian L2 by only 0.04323%, supporting spatial rather than dominant temporal error. Refine the core while holding the shell fixed, and then reverse that comparison. Locate the constraint peaks and compare a Cartesian control before attributing them to the interface. Add oblique packets that cross patch seams and repeat nontrivial AMR convergence. Declare absolute and pulse-normalized constraint/curvature error budgets before acceptance, alongside convergence and bounded amplification. Completion is not the acceptance criterion. Distinguish underresolution, gauge steepening, interface growth and outer reflection before changing arithmetic.
3. **Port the conformal adapter with reference preservation.** Cache one immutable Minkowski hyperboloidal reference at the actual mapped coordinates. Interpolate and transfer deviations, then add the exact reference; interpolating the varying background itself would destroy stationary balance. Transform the complete derivative jets with both J and H. Resolve a real Cauchy core in the bulk, record the actual minimum Omega, and establish stationary-reference convergence before applying disturbances.
4. **Construct a regular scri closure.** Replace the cut-sphere ghosts with a radial boundary closure for the full conformal system. Test the coupled linearized characteristic and constraint operator at faces and patch corners. Require a resolution-independent growth bound, convergent constraint/curvature errors and no resolution-worsening incoming or growing mode. The ordinary Sommerfeld subset and the scalar check alone do not satisfy this gate.
5. **Bind the oracle and arithmetic to actual failures.** On the new grid, save the first bad native stage and its exact stencil. Compare raw binary64 and multiprecision evaluations using the same saved inputs and atoms. Separate analytic-jet residuals from native finite-difference and boundary residuals at the actual mapped points and smallest-Omega collar. Change arithmetic only after identifying a consequential rounding error; passing unrelated manufactured identities is insufficient.
6. **Meet the original evolution gates.** Restore the intended substantial angular disturbance and certify any manufactured slicing remains spacelike over the tested interval. Require admissibility and convergent constraints/curvature at three resolutions and a true half timestep through `t=2`, then `6`, then `12`. Only after that, introduce a resolved single-hole wormhole interior and puncture gauge, retain the same Minkowski hyperboloidal reference, and require convergent horizon/mass diagnostics through the trumpet transition and multiple exterior crossings.
