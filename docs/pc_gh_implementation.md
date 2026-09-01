# PC-GH implementation

## Scope and authority

This document describes the production source at commit
`7bde88a31884eefa7e89089c63754f4e07c29841`, descended from the required
`project/bbh` baseline `d3148a1b87c9b28008c92388055d6aebd56c381a`.
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
| conformal geometry | `chi`, symmetric `gtilde`, `K`, symmetric `Atilde`, `Lambda` | 17 |
| GH gauge/configuration | `pi`, `A`, `beta` | 5 |
| first-order gradients | `X`, three symmetric `Q_k`, `Y`, row-major `B_i^j` | 33 |

The ordering is defined once in `src/pc_gh/pc_gh.hpp`; output names and restart
registration use that ordering.  `Q` stores `Q_kij` as three consecutive symmetric
tensors, and `B` stores `B_i{}^j` row-major.

The 26 diagnostic fields contain GH, physical, reduction, curl, algebraic,
regularity-ratio, and RHS-family measures.  History output integrates 19 squared
diagnostics with the physical volume element and records the volume separately.

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
-> PC-GH to ADM -> constraints -> characteristic timestep
```

The simultaneous algebraic projection enforces `det(gtilde)=1`, trace-free
`Atilde`, and the derivative-consistent trace projection of `Q`.  The projection is
applied after every RK stage.  The local timestep uses
`|beta^i| + alpha sqrt(chi gtilde^{ii})` in each active coordinate direction.

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
a four-metric, or use reference geometry.  `verify_source_policy.py` scans every
production file under `src/pc_gh` and fails on those forbidden dependencies.

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

## ADM conversion, communication, and outputs

`PcGhToADM` reconstructs `gamma_ij` and `K_ij` only for existing ADM consumers.  The
reconstructed physical geometry is never differentiated by or fed back into the
PC-GH RHS.  `ADMToPcGh` provides the initial-data conversion path and initializes the
first-order variables with the selected evolution derivative operator.

State communication, restriction/prolongation, load-balance migration, restart, tab
output, and history registration operate on the explicit 55-field array.  Their
presence is plumbing evidence only: AMR reduction/curl injection, MPI execution, and
GPU backends remain unqualified.

## Deliberate fail-closed limits

- vacuum only;
- no nonperiodic or GH constraint-preserving outer boundary condition;
- no Gauge A1 or Gauge B driver;
- no claim at the exact puncture (`A=chi=0`); the analytic theorem is on `r>0`;
- no fourth-order multilevel transfer;
- no single-hole evolution, Bowen-York transition, binary, AMR, MPI-runtime, GPU, or
  performance qualification yet.
