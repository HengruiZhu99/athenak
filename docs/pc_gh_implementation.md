# PC-GH implementation

## Scope and authority

This document describes the puncture-regular production source updated on
2026-09-02, descended from the PC-GH branch at `1c67ad8f`.
The equations are those independently derived in `docs/pc_gh_derivation.md`.
No old FO-GH, Ref-GH, Z4c equation, reference geometry, controller, or generated
equation source is an authority for the PC-GH RHS.  Z4c-style cell-centered storage,
task integration, communication, and output mechanisms are reused only as software
architecture.

The current implementation is a vacuum solver.  It must not be presented as a
single-hole or binary production solver until the open qualification gates below are
closed.

## State and storage ABI

`pc_gh::PcGh` owns four fine-grid arrays (`u0`, `u1`, `u_rhs`, and `u_con`) and a
coarse state array when multilevel operation is enabled.  The restart/output ABI has
exactly 55 evolved components:

| Sector | Components | Count |
|---|---|---:|
| conformal geometry | `w`, symmetric `gtilde`, `K`, symmetric `Atilde`, `Z` | 17 |
| GH gauge/configuration | `Cperp`, `rho`, `beta` | 5 |
| first-order gradients | `p`, three symmetric `Q_k`, `L`, row-major `B_i^j` | 33 |

The ordering is defined once in `src/pc_gh/pc_gh.hpp`; output names and restart
registration use that ordering.  `Q` stores `Q_kij` as three consecutive symmetric
tensors, and `B` stores `B_i{}^j` row-major.

The 28 diagnostic fields contain GH, ADM, reduction, curl, algebraic, conformal-SPD,
physical-validity, regular-gradient, and RHS-family measures. History output integrates
19 squared diagnostics with the coordinate-volume element and records that volume separately.

## Runtime construction and task graph

The `<pc_gh>` block is vacuum-only and mutually exclusive with Z4c.  Construction
validates spatial order, ghost width, gauge choice, and the currently supported AMR
combination.  Supported centered derivative orders are 2, 4, and 6; the corresponding
stencil selectors are 2, 3, and 4.  Fourth-order PC-GH with multilevel transfer fails
closed because its high-order transfer has not been implemented.

Each explicit stage follows:

```text
receive setup -> copy U -> calculate RHS -> RHS boundary hook -> RK update
-> restrict/send/receive -> physical BC -> prolong -> algebraic projection
-> constraints / optional masked output conversion -> characteristic timestep
```

The simultaneous algebraic projection enforces `det(gtilde)=1`, trace-free
`Atilde`, and the derivative-consistent trace projection of `Q`; evolved `Z` is left
unchanged. The projection is applied after every RK stage. The local timestep uses
regular composites (`rho*w^2` for the physical family and `sqrt(2*rho*w^3)` for the
lapse family) and never reconstructs physical ADM fields.

Only strictly periodic physical boundaries are implemented.  A nonperiodic PC-GH
domain exits with an error.  The explicit RHS-boundary task is an insertion point for
a later independently derived GH characteristic boundary condition; it currently does
nothing on a periodic domain.

## RHS kernel

`src/pc_gh/pc_gh_calcrhs.cpp` contains one main Kokkos kernel and one optional KO
kernel.  The main kernel:

1. loads the 55 state components and inverts only the conformal three-metric;
2. differentiates first-order fields with centered `Dx` operators;
3. builds conformal Christoffels, Brown first-order Ricci, regular lapse/curvature
   composites, and GH constraints;
4. assembles the 22 primary/configuration RHS components;
5. assembles the 33 standard-order gradient RHS components by differentiating the
   configuration source functions.

It does not call `Dxx` or `Dxy`, form physical Christoffels or physical Ricci, invert
a four-metric, use reference geometry, or divide by a puncture field.
`verify_source_policy.py` scans every production file and specifically rejects such a
division in the preferred evolution, constraint, projection, and CFL files.

KO dissipation is an optional separate pass over all 55 fields.  The user-facing
`<pc_gh>/dissipation` is a finite nonnegative amplitude.  At setup it is converted to
the alternating sign and `2^(-2*fd_stencil)` normalization required by the raw
`Diss<FD_STENCIL>` utility, so every nonconstant Fourier symbol is nonpositive.  The
earlier direct use of the positive input made the second-order operator anti-dissipative;
Gate 5 caught that defect before trumpet evolution.

## Gauge implementations

### Harmonic

`gauge=harmonic` sets the regularized prescribed source to zero.  Exact Minkowski and
the nonlinear periodic harmonic gauge wave exercise this path.

### Prescribed stationary trumpet Gauge A0

`gauge=a0` reads `inputs/pc_gh/gauge_a0_m1.dat`.  The table contains 4097 uniformly
spaced log-radius nodes on `r/M in [1e-8,1e4]`.  It stores values, first log-radius
derivatives, and (for `A`, `chi`, and radial shift) second log-radius derivatives;
`K`, radial trace-free curvature, `h_perp`, and radial `h^i` store values and first
derivatives.  `generate_gauge_a0_table.py` regenerates the file byte-for-byte.

The loader rejects malformed rows, extra columns, nonuniform/nonincreasing nodes,
nonpositive mass, non-3D meshes, exact-center cells, and any current cell or ghost
point outside the open interpolation domain.  Device interpolation uses quintic
Hermite polynomials for `A`, `chi`, and radial shift and cubic Hermite polynomials for
the remaining targets.  The target `B_i{}^j` uses the radial shift for its isotropic
part and `alpha Atilde` for its trace-free radial part.  This keeps each additive
metric-gradient RHS term finite instead of differentiating a small difference between
independent interpolants.

Gauge A0 modifies only the configuration source functions and their explicit spatial
gradients.  It is prescribed from coordinates, mass, and center and therefore does not
alter the established GH principal part.  The `pc_gh_trumpet_a0` problem generator
initializes all 55 fields directly from the same interpolant, without an ADM
finite-difference conversion.

`audit_gauge_a0_cancellation.py` evaluates 387 named production temporaries and
additive RHS terms at 73 radii across the open table domain in binary64, long double,
and 100-digit arithmetic.  It fails if an additive RHS term develops a fitted divergent
inner power or if the bounded high-precision table residual exceeds its audit limit.

With `<problem>/frozen_operator=true`, the same problem generator perturbs every state
component and extracts dense 55-by-55 constant and sinusoidal-response Jacobians from
the production RHS.  The analyzer reports both the raw operator and its restriction to
the 50-dimensional tangent space of the actual metric, trace-free-curvature, and `Q`
projection.  This is a Serial diagnostic path, not an evolution mode.

### Direct and switched moving-puncture gauges

`gauge=z4c_mp` implements the default advective AthenaK Z4c moving-puncture
coordinates directly in PC-GH variables:

```text
D0 w      = w (rho*w*K - B)/3
D0 rho    = rho[-2*K - (rho*w*K-B)/3]
D0 beta^i = GammaTilde^i(Q) - Z^i - shift_eta beta^i
```

The default `shift_eta=2.0` corresponds to a unit mass. The production STANDARD
gradient equations evolve `p_i=partial_i w`, `L_i=2 partial_i alpha`, `Q`, and `B`
as exact spatial derivatives of their configuration equations; they are not
reconstructed through cancellation of separately differenced source terms. The `p`
and `L` equations are the explicitly denominator-free forms recorded in the derivation.

`gauge=z4c_mp_hyperbolic` adds
`S(rho*w^3) rho*w^3*gtilde^{ij}(rho*p_j-L_j/2)` to the shift. The default switch is a
cubic smoothstep from zero at `shift_switch_z0=0.1` to one at
`shift_switch_z1=0.5`; setup requires `0 < z0 < z1 < 4/7`. Its differentiated
STANDARD equation includes both the derivative of the metric-gradient term and the
`S'(z) partial_i z` product, with `partial_i z=w^2(L_i/2+2*rho*p_i)`. This variant has a conditional complete characteristic
basis on the domain stated in the derivation; setup does not claim or enforce that an
evolution remains in that domain.

The moving-puncture CFL estimate includes the physical, lapse,
transverse-shift, and longitudinal-shift families. It conservatively uses the direct
gauge upper factor for both variants.

The built-in `pc_gh_one_puncture` problem reuses the exact time-symmetric wormhole ADM
data from `pc_gh_bowen_york`, with `w=psi^-2`, `rho=1`, `p=partial w`, `L=2p`,
zero shift, and zero extrinsic curvature. Its MPI-capable diagnostics fail on the
first nonfinite state, RHS, constraint, characteristic speed, determinant, or
eigenvalue; on negative `w/rho`; or on a non-SPD conformal metric. They record all
field, constraint, transfer-change, and RHS bounds throughout the run.
`tst/inputs/z4c_one_puncture_control.athinput` makes the actual Z4c gauge
defaults explicit for a matched control.

## ADM conversion, communication, and outputs

`PcGhToADM` is absent from the normal evolution dependency chain. It reconstructs
`alpha=rho*w`, `gamma_ij=gtilde_ij/w^2`, and
`K_ij=(Atilde_ij+gtilde_ij*K/3)/w^2` only outside
`physical_output_inner_radius` when explicitly requested for output. Masked cells
receive an output-only flat extension and `pcgh_physical_valid=0`; those values are
never differentiated or fed back into evolution. `ADMToPcGh` is initialization-only
and initializes the first-order variables with the selected derivative operator.

The `pc_gh_bowen_york` audit pgen exercises that path with exact time-symmetric
isotropic Schwarzschild, the zero-momentum/zero-spin member of ordinary Bowen-York
data.  It fills ADM data on active cells and ghosts, uses the baseline pre-collapsed
`alpha=psi^-2`, sets zero shift and extrinsic curvature, converts to PC-GH, and
round-trips back to ADM. The puncture center must lie on cell faces so every active
cell has `r>0`. The defining `rho=alpha/w` conversion alone uses the configurable
initial-data guard and reports the unguarded global minimum plus activated-cell count.
Its final diagnostic calls the production RHS and
constraint kernels, compares against analytic continuum state/RHS values on a bounded
shell, and writes per-field maximum locations.

An independent three-precision script also audits the standard analytic momentum and
spin Bowen-York conformal extrinsic-curvature leading fields.  It does not synthesize
the regular elliptic correction, so the existing project/bbh TwoPunctures path remains
the authority for future constraint-satisfying boosted, spinning, and binary data.
That custom pgen currently relies on an external `twopuncturesc` header/library tree
which is not tracked by the required baseline or registered as a submodule.  A local
untracked symlink exists only in the original dirty checkout; it remains user-owned and
was not imported into this branch.  The custom pgen has therefore not been built or
advertised as PC-GH-capable here.

State communication, restriction/prolongation, load-balance migration, restart, tab
output, and history registration operate on the explicit 55-field array.  The PC-GH
run task list deliberately follows the Z4c ordering: update, conservative cell-centered
restriction, communication and physical boundaries, cell-centered prolongation,
algebraic projection, diagnostic/output conversion, and timestep selection. It calls the same
`RestrictCC(..., true)` and `ProlongateCC(..., true)` paths as Z4c rather than defining
formulation-specific multilevel transfer.  Their presence is plumbing evidence only:
dynamic-AMR reduction/curl injection and GPU backends remain unqualified. The static
four-level hierarchy has been exercised with 12-rank MPI at M/16--M/24.

One-puncture qualification history adds coordinate-volume RMS accumulators for the
full domain, `w^2>=0.0625`, and fixed radial exteriors `r>0.5M`, `r>M`, and `r>2M`.
The `ah` history label is a conservative spherical coordinate-radius enclosure plus a
configured buffer; it is not a surface integral and must not be presented as a dynamic
apparent-horizon mask. Horizon dumps first interpolate 18 regular conformal PC-GH
variables and reconstruct physical ADM values only outside the declared inner mask.
The flat dense-cube interior extension exists only for the external adapter and is
never copied into evolution storage. Apparent-horizon area, mass, shape, and solver residuals
remain outputs of the external AHFinderDirect executable, not quantities inferred by
AthenaK from the dump itself.

## Deliberate fail-closed limits

- vacuum only;
- no nonperiodic or GH constraint-preserving outer boundary condition;
- no Gauge A1 or Gauge B driver;
- equivalence to the older formulation is proved only for `w>0,rho>0`; the preferred
  polynomial evolution itself applies no puncture-field floor;
- no fourth-order multilevel transfer;
- no perturbed, boosted, spinning, binary, GPU, spectral/SAT, or
  performance qualification yet.

The 2026-09-02 single-hole campaign is classified `PARTIAL IMPROVEMENT`: both gauges
reach `6M` at M/16--M/24, but `rho` develops a negative inner power and an inner
maximum growing approximately as `N^1.12`; `L` and `B` also worsen with refinement.
Survival is therefore not promoted to puncture qualification.
