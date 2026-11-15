# Native scri recorder, schema 1

Implemented against `edbe24a3d` on `codex/hyperboloidal-binary-integration`.
This is an accepted-state data archive and native-coordinate projection. It does
not identify simulation coordinates with inertial Bondi coordinates. No evolution,
gauge, projection, boundary solve, timestep, KO coefficient, or mesh is changed.

## Configuration and lifecycle

Append `inputs/z4c_hyperboloidal/scri_output.athinput` to an existing input, choosing
an unused output number. Recommended settings are:

```ini
<output20>
file_type = scri
id = scri
dt = 0.01
nodal = true
modes = true
lmax = 8
unit_description = geometric code units; c=G=1; physical mass scale unspecified
```

`dt` is in simulation coordinate time. Choose cadence by temporal convergence for
the intended analysis; 0.01 is a storage example, not a guaranteed sufficient
radiation sampling interval. Standard `dcycle` may replace `dt`. File numbers,
`last_time`, initial and final output follow the native Driver. Repeated final
calls at an already written `(cycle,time)` in the same live topology are suppressed.
A topology change at that same time emits a new, explicitly numbered epoch. The last written pair
is persisted in restart parameters along with native numbering. Output blocks
are independent; their block number is included in filenames even if IDs match.

`nodal=true` always retains the entire support described below, independently of
`lmax`. `modes=false` disables modes and ignores `lmax`. `nodal=false` is a reduced,
nonreconstructible time series: it omits all time-dependent nodal fields, collar
fields, sampled jets and normal lifts. It still writes the topology metadata.
At least one must be enabled. No field-subset or collar-truncation switch exists.
Slicing, `gid`, and `ghost_zones` are rejected for this surface output.

`ScriOutput` uses `BaseTypeOutput::LoadOutputData/WriteOutputFile`. No RK hook or
independent scheduler is installed. Each due output transports only its actual
support donors through `MappedExchange`; the full volume is never copied to the
host. A separate endpoint-only `SurfaceExchange` samples and solves from these
values. It never calls `Project`, `Evaluate`, `FormResidual`, or diagnostic history
consumers, and never writes module state, residuals, RK RHSs or histories.
The normal solve is recomputed from the accepted state, not taken from an old
stage. All-rank participation includes ranks with no surface nodes.

Caches rebuild when `GetAMRLoadBalanceUpdateSeq()` changes. Repartition, refinement
and coarsening produce new epoch files. A restarted process creates a new epoch
at its first new record, even if the topology happens to match an earlier epoch;
that makes a continuation independently interpretable without searching a prior
process's metadata. Configuration is immutable during a native run. A changed
configuration starts another process/epoch.

Files are opened with exclusive creation. A rewind into an already populated
output namespace fails rather than overwriting an earlier branch or silently
appending duplicate samples. Use a fresh output directory for a branched restart.
Keep earlier archives if earlier history is needed. A completed file has an end
marker; interrupted files fail the reader. `load_record` rejects missing ranks,
duplicate ranks and mixed record identifiers. There is no crash-recovery rewrite
of a partially written record.

## Retained representation

All 31 stored slots are saved redundantly, including the 26 independent slots:

| Slots (zero based) | Native storage definition |
|---|---|
| 0 | conformal `chi` |
| 1–6 | determinant-one Cartesian metric, order xx,xy,xz,yy,yz,zz; diagonal minus 1 |
| 7–12 | trace-free conformal curvature, same symmetric order |
| 13 | `P+3`, where `P=K-2*Theta` |
| 14 | `Theta` |
| 15–17 | `Lambda_x,y,z`; independent connection variables |
| 18 | conformal lapse minus `(1+r²)/2` |
| 19–21 | `B_x,y,z`, lapse-driver flux auxiliary, with Euclidean reference raising |
| 22–24 | shift components plus `x,y,z` |
| 25–27 | `D_x,y,z`, shift-driver auxiliary; not interchangeable with Lambda |
| 28–30 | stored `J_x,y,z` source slots, not independent RK variables |

Slots 6 and 12 are algebraically dependent. Independent indices are
`0..27` excluding `6,12`. See the archived `fields` string for exact slot labels
and the archived mathematical identity/source hashes for the equation version.
Tensor components share the same Cartesian basis on every patch. Patch maps
transform derivative directions, not the component labels.

The raw `u0/s0` state already contains the reference values and storage offsets.
For the boundary interpolation, subtract the stored reference values in independent
slots, and additionally divide bulk chi residuals by the recorded residual weight.
Surface chi residuals are not divided. Nonindependent residual slots are zero.
The archive includes full reference Cartesian jets at every support node and
surface target, so restoration requires no local binary initial-data file and no
re-evaluation of the reference initializer.

For algebraic restoration, let `a=gxx`, `b=gxy`, `c=gxz`, `d=gyy`, `e=gyz`, and
`minor=a*d-b*b`. Then

```
gzz = (1 - 2*b*c*e + a*e*e + d*c*c)/minor
Azz = -((d*gzz-e*e)*Axx + 2*(c*e-b*gzz)*Axy
        + 2*(b*e-c*d)*Axz + (a*gzz-c*c)*Ayy
        + 2*(b*c-a*e)*Ayz)/minor
```

These formulas act on jets as well as values. `restored_endpoint_jets()` applies
the native reference/chi-weight restoration, algebraic expansion, and normal
lifts. `physical_values()` only removes storage offsets; its result is still in
the native conformal frame. It is not a physical radiation extractor.

### Endpoint support and normal derivatives

The authoritative endpoint is the separately evolved `s0` at local radial
coordinate 1. It is never a radial ghost. Each endpoint row uses the native seven
radial abscissae: the last six interior nodes and the true endpoint. In local
block coordinates the six are `(n-6+i+node_offset[0])/n`, `i=0..5`. The angular
stencil is the native five-by-five jet stencil. At seams and refinement interfaces
its nonlocal samples use the existing sixth-order donor interpolation. Thus the
stored support includes the exact unique donor union, including the modest
additional radial/angular donor support required at these interfaces, rather
than assuming six uniform radial planes are sufficient everywhere.

Each target's actual flattened row, its ordered donor keys and complete value,
first-derivative and second-derivative weights are archived. This preserves seam,
patch-subdivision and interpolation conventions exactly, without reimplementing
ownership searches in the reader. Duplicate support donors across rank files
are intentional. They do not enter quadrature. There are no volume dumps.

The anchor-difference summation convention is:

```
jet.f = first_donor_value
jet += sum_t weight[t] * (donor_value[t] - first_donor_value)
```

Reference restoration follows `SphericalPolicy::Restore`; at scri chi is
`(residual_chi_jet + reference_chi_jet/r²)*r²`, using Cartesian jet arithmetic.
Full `sampled_residual_jets` provide a redundant, independent round-trip target.

For each surface tile, save three shift-normal correction coefficients, or four
coefficients (shift plus lapse) for regularity/clock-source treatment. The angular
interpolant is the native six-by-six Lagrange polynomial in local tile coordinates;
its clamped start is `clamp(index-2,0,n-6)`. The normal lift is

```
z = (r-1)/dr
H(r) = dr * z * product(i=0..5, z+i+gap) / product(i=0..5, i+gap)
lift = H(r) * angular_interpolant
```

`dr` and `gap=1-node_offset[0]` are stored per face. This polynomial vanishes at
all seven radial sample nodes but has unit radial derivative at scri. Retaining
only the field values would lose it. The reader reconstructs its Cartesian jets
from coefficients and map derivatives; `normal_lift_jets` are a redundant check.
The same coefficients define the closure's lift in its last two radial layers.

### Geometry, conformal factor, units and time

For every support node, retain its GID, stored indices, owner rank, local
coordinates, Cartesian coordinates, inverse-map Jacobian and Hessian, map bounds,
chart/face/layer, equiangular flag, radial parameters and node offsets. Surface
owners are native outer leaves with nominal `j,k=0..n-1`; ghosts and independently
evolved angular overlaps are excluded from modal quadrature.

`J[A][a]=dq_A/dx_a`, `H[A][a][b]=d²q_A/dx_a dx_b`. A logical jet transforms as
`df/dx_a = J[A][a] df/dq_A`, with the standard Jacobian-product and Hessian terms
for its second derivative. This is a common Cartesian component basis, not a
patch dyad.

The prescribed Omega is static. Its profile parameters and value/gradient/Hessian
at every donor are stored. At scri and in the profile's outer branch,
`Omega=(1-r²)/2`, `gradient=-x`, `Hessian=-I`. The boundary equations explicitly
set Omega to zero at the true endpoint, even if the recorded Cartesian radius
differs from one by floating-point roundoff. The flat-inner branch and C6 transition
are defined in `conformal_profile.hpp`, with recorded parameters and source hash.
The prescribed stationary boundary gauge source's seven numbers are also saved.
No external initial-data file is needed to restore the archived geometry.

Times are exact native `Mesh::time` values and cycles are native integers.
There is **no accepted-state time derivative array** in this schema. In particular,
no `u_rhs` or `s_rhs` stage scratch is mislabeled as a time derivative. Reconstruct
time derivatives from a temporally resolved series of accepted states. Future
mixed/time derivative calculations must demonstrate temporal convergence and
account for irregular cadence and remeshing; the finite collar is insufficient
for replaying the entire bulk evolution/KO operator. No bandwidth or arbitrary
asymptotic order is promised. Spatial support reproduces the existing second-jet
endpoint closure; it does not validate higher asymptotic coefficients.

Coordinates and times are geometric code units (`c=G=1`), not Bondi coordinates or
seconds. The complete input, runtime mathematical identity (including reference,
gauge and residual parameters), source SHA256 manifest, compiler/configuration,
execution space, `sizeof(Real)` and user-provided unit description are retained.
An astrophysical mass-to-seconds/meters normalization must be supplied for the run;
its absence is recorded explicitly rather than guessed.

## Native-coordinate quadrature modes

For every one of the 31 stored components, the writer computes
`sum_q w_q f_q conjugate(Y_lm(theta_q,phi_q))` on actual owned endpoint nodes.
All `m=-l..l` are retained, with index `l*l+l+m`; values are real/imaginary pairs.
These are scalar (`s=0`), orthonormal complex harmonics with Condon–Shortley phase,
`theta=acos(z/r)`, `phi=atan2(y,x)`. No spin-weighted combinations or patch-local
tensor projections are provided. The coefficients retain the storage offsets
listed above and are labelled native modes at simulation time.

The measure is the fixed coordinate unit-sphere solid angle, **not** the area of
the evolved induced metric. For gnomonic cube coordinates `u,v`,
`dOmega=du*dv/(1+u²+v²)^(3/2)`. For equiangular coordinates first apply
`u=tan(pi*a/4)`, `v=tan(pi*b/4)` and multiply by
`(pi/4)²*(1+u²)*(1+v²)`. We use composite midpoint weights over each native leaf's
angular extent. They are not rescaled to force area `4*pi`; quadrature error must
converge with resolution. Mixed-resolution leaves count each angular region once.

Weights and harmonics are cached per epoch. Kokkos team reductions run on device;
only coefficient arrays are MPI-summed, and rank zero writes the global modes.
There is no all-rank full-surface gather. The nodal files remain rank-local.
Enabled modes require `0 <= lmax <= 128` and `lmax < Ncoarse`, where `Ncoarse` is
the minimum `2*n/(angular_leaf_width)` over both axes and all outer leaves. This
is a sampling admission limit, not an accuracy guarantee; N12 admits default 8.
Converge angular resolution for scientific use.

## Binary schema and reader

Files are siblings named
`basename.id.oBLOCK.NUMBER.rRANK.scri`; epoch files insert `.epoch` before `.scri`.
NUMBER and RANK use eight decimal digits. Each rank writes independently, including
an empty-rank record. Metadata is saved once per local epoch, not every sample.

The file starts with 8 bytes `SCRI0001`, then a native-endian uint32 `0x01020304`.
Each array has uint32 name length, uint32 type, uint32 ndim, ndim uint64 extents,
UTF-8 name bytes, then C-order payload. Type 1 is binary64, type 2 is signed int64,
type 3 is UTF-8 bytes (one-dimensional). A uint32 zero name length ends the file.
No struct padding or Kokkos memory layout is serialized. The reader recognizes
both byte orders and rejects unsupported versions, truncation and trailing data.
Integer topology identifiers stored in type-1 tables are exact int32-range values.

Epoch arrays:

| Name | Shape / content |
|---|---|
| `fields`, `parameters`, `initial_data_identity`, `source_manifest`, `build`, `execution_space`, `units`, `basis`, `kind` | text |
| `configuration` | n, origin, overlap, nghost, scri-treatment enum, residual-weight inner/outer radii, Real bytes, coarse angular resolution |
| `conformal_profile` | flat flag, flat radius, outer radius, plateau |
| `boundary_source` | alpha, shift_ratio, lapse_offset, lapse_linear, lapse_quadratic, shift_offset, scri_obstruction_reference |
| `surface_ids` | Q×5: gid, local face, k, j, owner rank |
| `surface_geometry` | Q×39: x[3], J[3,3], H[3,3,3] |
| `surface_reference` | Q×31×13: reference Cartesian jets |
| `surface_weights` | Q: coordinate solid-angle quadrature weights |
| `faces` | F×27: gid, dr, gap, then map fields |
| `bulk_ids`, `edge_ids` | B/E×5: gid,k,j,i,owner rank; edge i=0, edge j/k have no storage origin |
| `bulk_maps`, `edge_maps` | B/E×24: map fields |
| `bulk_geometry`, `edge_geometry` | B/E×42: local q[3], then x,J,H |
| `bulk_reference`, `edge_reference` | B/E×31×13 |
| `bulk_omega`, `edge_omega` | B/E×13 |
| `bulk_residual_weight`, `edge_residual_weight` | B/E; edge weight is 1 |
| `rows` | Q×3: begin,end,physical radial spacing; half-open term range |
| `term_ids` | T×2: boundary flag, donor ordinal into edge or bulk |
| `term_weights` | T×13: Cartesian jet weight |

A jet's 13 numbers are `f, dx,dy,dz, hxx,hxy,hxz,hyx,hyy,hyz,hzx,hzy,hzz`.
The 24 map fields are `chart_id,face,layer,lo[3],hi[3],half_width,inner_radius,
outer_radius,cube_transition,equiangular,outer_surface,node_offset[3],overlap_sides[6]`.

Accepted-state arrays:

| Name | Shape / content |
|---|---|
| `kind`, `epoch` | text; epoch is a relative sibling filename |
| `record` | int64[8]: cycle,file_number,mesh_sequence,rank,nranks,nodal,modes,lmax |
| `time` | binary64[1]: accepted simulation time |
| `surface_state` | Q×31 raw accepted native slots |
| `bulk_state`, `edge_state` | B/E×31 raw accepted donor slots |
| `normal_correction` | Q×3 or Q×4; shift xyz, optional lapse |
| `normal_lift_jets` | Q×3/4×13; redundant lift reconstruction check |
| `sampled_residual_jets` | Q×31×13; before reference restoration/lifting |
| `native_modes` | rank zero only, 31×(lmax+1)²×2 |
| `preparation_seconds` | local due-output preparation wall time, including epoch setup when applicable |

Example:

```python
from scri_reader import load, restored_endpoint_jets
state, epoch = load("run.scri.o20.00000000.r00000000.scri")
jets = restored_endpoint_jets(state, epoch)
```

`python3 vis/python/scri_reader.py FILE --verify` checks saved row reconstruction.
`load_record(paths)` additionally validates the complete distributed record.
The reader requires only NumPy. The C++ writer adds no HDF5 dependency.

## Validation and follow-up

Run the focused test with
`python3 tst/z4c_hyperboloidal/scri_recorder.py --build BUILD --output NEW_DIRECTORY`.
It builds against the actual AthenaK objects, tests nonzero manufactured states,
actual refine/coarsen/repartition, raw field/reference/geometry equality, endpoint
jets and lifts, read-only hashes, one/two/eight-rank agreement, empty scri ranks,
restart continuity, output scheduling, multiple blocks, nondefault/invalid lmax,
truncation rejection and enabled/disabled checkpoint equality. CTest also tests
analytic harmonic recovery, all m at l=3 and 8, leakage into every l<=8,
angular-resolution convergence, and patch subdivision for both map types.
See `scri-recorder-validation.md` for measured costs and limitations of the test.

Subsequent work is deliberately separate: derive and validate the coordinate-to-
Bondi inertialization; construct physical strain, news and asymptotic Weyl fields;
validate sufficient normal and temporal order for radiation and Coulombic terms;
compute asymptotic charges with declared conventions; implement BMS transformations;
fit early and late frames; and analyze memory and frame-transformed signals.
None of these physical analyses is performed or claimed by this recorder.
