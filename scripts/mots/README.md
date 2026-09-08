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
current-slice success is separate. Frozen searches ignore restored success as a detection, but try the saved
center/shape together as a guess before fresh discovery; angular coefficients
are padded or truncated when changing lmax. No last-good seed is recovered from a failed legacy
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
`4*ntheta+3` Gauss points, plus a dense radius check including poles. Expansion at the poles uses its
analytic m=0 limit and participates in epsilon_inf. The
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


## Angular continuation

`angular_continuation.py` runs bounded coarse-to-fine searches on the same
checkpoint, carrying the best evaluated center and coefficients into the next
stage with zero padding. Failed candidates remain guesses, never detections.
This is spectral continuation, not a full multigrid correction cycle. Each stage
stops at convergence, line-search failure, or its iteration budget (default 500).

```
python3 scripts/mots/angular_continuation.py --athena /absolute/path/to/athena \
  --checkpoint /absolute/path/to/input.rst --output /absolute/new/sequence \
  --levels 8 16 32 64 128
python3 scripts/mots/plot_angular.py --sequence /absolute/new/sequence \
  --output /absolute/plot/directory
```

Use `--initial-result /path/to/previous/result` to begin from an existing trial,
and `--ntheta-factor 4` to double the default solve quadrature (`2*L+4`).
The one-search wrapper also accepts `--seed`, `--seed-only`, `--ntheta`, and
`--profile-points`. A seed text file contains center_z, coefficient count, then
normalized real Y_l0 coefficients. Seed contents and SHA256 enter the manifest.
The seed count cannot exceed L+1; missing higher modes are zero padded.

The m=0 basis now uses a three-term Legendre recurrence differentiated twice
with respect to theta. This avoids the severe cancellation/overflow in the
factorial Wigner sum at high L and handles poles without division by sin(theta).
The generic 3D harmonics are unchanged. Tests cover high-L equator values,
axis derivatives, and the Legendre differential equation through L=128.

Every surface gets a dense-profile CSV and a `mots.mots_dense_<id>.json` containing
independent area-weighted residuals, including explicit pole checks. The
continuation requests 1,061 points (or the existing 4*ntheta+3 minimum, if larger).
Plots use the dense area and residuals, not search-grid residuals. The signed
companion uses symmetric log scaling; the primary figure uses absolute expansion.
