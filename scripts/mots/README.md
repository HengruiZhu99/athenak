# Frozen VC Cartoon MOTS analysis

`fastflow/horizon_only=true` requires a restart and a fresh `-d` directory.
It restores the checkpoint mesh and fields, reconstructs ghost zones, derives
native ADM fields, searches at the saved time, and exits without the normal
initialization/evolution/final-output stages. It rejects custom boundary
callbacks and non-vacuum/non-VC/non-Cartoon configurations. Active fields are
compared bitwise before ghost initialization and after the search; time,
cycle, timestep and every leaf location must remain identical.

Build Athena normally with CMake (CUDA/MPI are supported). The IrisK library
is not needed to analyze a saved restart with built-in boundary conditions.
New analysis keys must be supplied via an `-i` overlay, since Athena does not
allow undeclared command-line parameters. The wrapper generates that overlay:

```
python3 scripts/mots/search_checkpoint.py --athena /absolute/path/to/athena \
  --checkpoint /absolute/path/to/input.rst --output /absolute/new/result \
  --lmax 16
```

Use `--launcher 'srun -n 2 -c 16 ...'` for MPI. Run the wrapper once; it launches
Athena ranks together. The manifest records checkpoint/executable/source hashes,
Slurm ID, command, timestamps and exit status. Analysis output is a child
`search` directory; the wrapper refuses to reuse an existing result directory.
For source exports without Git, copy the corresponding SHA into
`source_revision.txt`; per-source-file hashes are recorded regardless.

## Numerical contract

The native VC sampler uses tensor-product cubic interpolation of ADM fields
and separately computed native Cartoon metric derivatives. All derivative
nodes are active; cubic stencils become one-sided near block faces. Native
finite differences consume reconstructed ghosts. Finest-leaf ownership is
deterministic, and invalid/nonfinite sampling is reported without order fallback.
This approach has measured spatial error; Newton convergence cannot remove it.

Searches try preserved center/shape pairs, the origin, and refined local axis
lapse minima with a logarithmic radius bank. The default lower radius is four
local grid spacings. Tracking seeds persist in memory across failed searches;
current-slice success is separate. Frozen searches ignore restored success and
perform discovery afresh. No last-good seed is recovered from a failed legacy
restart carrier.

The solver uses radius-squared flow scaling, a conservative all-angle 10%
displacement limit, backtracking, and a pivoted dense numerical-Jacobian Newton
solve. Near a solution Newton is tried first; otherwise it is the fallback if
flow stalls. Both steps decrease the dimensionless direct-residual merit.
`mots_newton_switch`, `mots_backtracks`, `mots_displacement` (at most 0.1),
`flow_iterations_0`, `mots_radius_count`, `mots_radius_min`, and existing axis/
maximum-radius settings control the bounded search. No tolerance is changed
in response to failure.

At least `2*lmax+4` angular points are used; independent dense verification uses
`4*ntheta+3` Gauss points, plus an all-angle radius check including poles. The
m=0 basis enforces pole regularity. Surface CSVs store outgoing/ingoing expansion
on the dense grid. The history `direct_residual` is now epsilon_2 normalized by
area radius; `epsilon_inf` uses the same radius. `solver_converged` and `verified`
are distinct. Neither implies spatial validation or outermost status.

Every independently verified component can count as a detection. A mirror-pair
label additionally requires reflected coefficients and a conservative separating
bound. Failure of this sufficient disjointness test does not reject components.
Recentring replaces a successful candidate only if its replacement verifies.

## Qualification

```
cmake -S . -B build -DAthena_BUILD_UNIT_TESTS=ON
cmake --build build --target athena athena_z4c_cartoon_m0_fastflow_unit_test
ctest --test-dir build -R 'z4c_cartoon_m0_fastflow$' --output-on-failure
python3 scripts/mots/qualify.py --athena /absolute/path/to/athena \
  --output /absolute/new/qualification
```

The math test covers scaled/translated Schwarzschild discovery, a coordinate-
stretched Schwarzschild horizon, flat nondetection, failed-iterate diagnostics,
failed recentering, unequal tiny pair residuals and duplicate rejection. The
native-grid suite creates analytic checkpoints with zero evolution, checks
uniform spatial refinement and a surface crossing a static refinement interface,
and requires unchanged fields/time/topology in frozen analysis. The VC Kerr
fixture now initializes stored ADM ghosts before ADM-to-Z4c differentiation.

Acceptance targets remain epsilon_2 <= 1e-6 and epsilon_inf <= 1e-5. Spatial
resolution sensitivity is reported separately; the executable deliberately
never sets `spatially_validated=true`. CC/full-3D qualification, independent
shooting, general non-star-shaped surfaces and outermost identification are
outside this milestone. No search failure certifies the absence of a MOTS.
