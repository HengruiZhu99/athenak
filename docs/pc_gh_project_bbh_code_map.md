# PC-GH code map at the `project/bbh` base

## Provenance and audit boundary

This audit was performed before any PC-GH physics implementation. The authoritative
base is:

```text
repository: HengruiZhu99/athenak
branch:     project/bbh
commit:     d3148a1b87c9b28008c92388055d6aebd56c381a
date:       2026-04-30T13:27:27-04:00
subject:    improving the performance a bit
```

At worktree creation, local `project/bbh`, `project/bbh@{upstream}`,
`origin/project/bbh` from `git ls-remote`, and the new
`codex/pc-gh-from-scratch-20260901` worktree HEAD all resolved to that commit.

The pre-existing checkout at `/home/hzhu/Desktop/research/gr/athenak` was a dirty,
unrelated Ref-GH worktree. It was not modified. This audit did not open or use any
`src/fo_gh`, `src/ref_gh`, reference-geometry, reference-frame, controller, or old
gauge-subtraction implementation. Z4c is treated only as a software-architecture
template; its Einstein and gauge equations are not an admissible physics source.

The requested path `src/pgen/z4c/` does not exist at this base. Its current equivalents
are `src/pgen/z4c_*.cpp` and `src/pgen/tests/z4c_*.cpp`, specifically gauge wave, one
puncture, SpECTRE BBH, robust stability, TwoPunctures, boosted puncture, and linear wave.

## Executive map

The reusable architecture is a flat cell-centred state owned by a physics module,
shallow tensor aliases over that state, a three-queue numerical-relativity task graph,
generic cell-centred communication, RK registers, high-order first-derivative and KO
operators, generic ADM storage, and generic mesh restriction/prolongation machinery.

Several apparently generic paths are hard-wired to `pz4c`, Z4c variable indices, or a
boolean named `is_z4c`. PC-GH integration therefore requires small, explicit framework
generalizations. Aliasing PC-GH as Z4c would corrupt restart/output semantics and is not
an acceptable shortcut.

| Area | Current authority | PC-GH disposition |
|---|---|---|
| Flat state and RK registers | `src/z4c/z4c.hpp:44-89`, `src/z4c/z4c.cpp:65-119` | Reuse allocation and shallow-alias patterns; define an independent 55-field enum. |
| Tensor views | `src/athena_tensor.hpp` | Reuse `AthenaTensor`, `AthenaHostTensor`, and `AthenaPointTensor`. |
| RHS launch shape | `src/z4c/z4c_calcrhs.cpp:28-62` | Reuse templated stencil dispatch and `par_for` shape only; derive every PC-GH equation independently. |
| Finite differences | `src/utils/finite_diff.hpp` | Reuse `Dx<2/3/4>` and `Diss<2/3/4>`; production PC-GH must reject `Dxx` and `Dxy`. |
| RK update | `src/z4c/z4c_update.cpp:21-43` | Reuse after changing owner, variable count, and labels. |
| Task DAG | `src/tasklist/numerical_relativity.*`, `src/z4c/z4c_tasks.cpp:33-109` | Add a distinct PC-GH physics dependency and task range; mirror ordering. |
| CC exchange | `src/bvals/bvals*.{hpp,cpp}` | Reuse after generalizing the Z4c-named high-order-CC mode. |
| Physical BC fill | `src/bvals/physics/z4c_bcs.cpp` | Extrapolation mechanics may inform an early diagnostic BC; Z4c mathematics is not a GH CPBC. |
| AMR transfer | `src/mesh/{restriction,prolongation}.hpp`, `src/mesh/mesh_refinement.*` | Reuse operators, measure reduction/curl injection, and do not reset derivative fields automatically. |
| ADM storage | `src/coordinates/adm.*` | Reuse ADM arrays and determinant/inverse/trace helpers; generalize lapse/shift ownership. |
| Initial data | current Z4c pgens | Reuse ADM and TwoPunctures plumbing only; implement independent PC-GH conversion. |
| Weyl/waves/trackers/horizons/CCE | Z4c support objects and task wrappers | Reuse only components consuming generic ADM or cell-centred data; decouple ownership from `pz4c`. |
| Restart/output/load balance | `src/pgen/pgen.cpp`, `src/outputs/*`, mesh code | Add explicit PC-GH arrays, names, metadata, and counts. |

## State and allocation pattern

`Z4c` declares one contiguous component enum, named output variables, flat
`DvceArray5D<Real>` views for `u0`, `u1`, `u_rhs`, constraints, coarse state, and Weyl
state (`src/z4c/z4c.hpp:44-89`). The constructor sizes them over

```text
(max(nmb_thispack, nmb_maxperrank), nvar, ncells3, ncells2, ncells1)
```

and binds scalar, vector, and symmetric-tensor aliases with `InitWithShallowSlice`
(`src/z4c/z4c.cpp:65-119`). This is the correct PC-GH storage model:

```text
u0, u1, u_rhs: 55 components
coarse_u0:      55 components when multilevel
u_con:          separately named diagnostic families
u_weyl:         two generic wave components if retained
```

The PC-GH enum must encode exactly:

```text
chi(1), gtilde(6), K(1), Atilde(6), Lambda(3), pi(1), A(1), beta(3),
X(3), Q(18), Y(3), B(9) = 55
```

Shallow aliases can expose `Q_kij` as three consecutive symmetric rank-2 slices and
`B_i^j` as a nine-component unsymmetric object. `AthenaTensor` device field aliases stop
at rank 2, so `Q` and `B` need small wrapper structs or direct flat indexing.
`AthenaPointTensor` supports higher-rank per-cell scratch and is appropriate in the RHS.

The Z4c algebraic projection is placed after prolongation and before ADM conversion
(`src/z4c/z4c_tasks.cpp:73-78`). Its kernel rescales the conformal metric and removes the
trace from `Atilde` (`src/z4c/z4c.cpp:262-303`). PC-GH should keep this placement while
using its independently derived three-part projection, including each `Q_kij` and a
projection-correction norm.

## Numerical-relativity task graph

`NumericalRelativity` owns start, run, and end queues. A queued task contains a symbolic
`TaskName`, dependencies, and a bound `TaskStatus(Driver*, int)` callable. Assembly adds
tasks whose dependencies already have IDs and aborts on a missing or cyclic dependency
(`src/tasklist/numerical_relativity.hpp:98-180`,
`src/tasklist/numerical_relativity.cpp:75-137`).

At this base, task-to-physics dispatch relies on contiguous enum ranges:

```text
task < MHD_NTASKS  -> Phys_MHD
task < Z4c_NTASKS  -> Phys_Z4c
otherwise          -> Phys_None
```

PC-GH needs its own contiguous range, `Phys_PcGh`, availability check, forward
declaration, and queue call. Appending PC-GH tasks without extending `NeedsPhysics`
would incorrectly mark them as always available.

The authoritative Z4c ordering (`src/z4c/z4c_tasks.cpp:33-109`) is:

```text
start: InitRecv, InitRecvWeyl

run: CopyU -> CalcRHS<fd_stencil> -> BoundaryRHS -> ExpRKUpdate
  -> RestrictU -> SendU -> RecvU -> ApplyPhysicalBCs -> Prolongate
  -> EnforceAlgConstr -> ConvertToADM -> optional excision -> NewTimeStep

end: ClearSend -> ClearRecv -> ADMConstraints -> Weyl -> RestrictWeyl
  -> SendWeyl -> RecvWeyl -> ProlongateWeyl -> clear wave communications
  -> Wave -> TrackCompactObjects -> CCE -> DumpHorizons
```

The requested PC-GH graph mirrors this order but uses its own boundary task and has no
reference/cache/controller tasks. Optional MHD dependencies are outside the initial
vacuum-only scope.

`CopyU` deep-copies `u0` to `u1` on stage 1 and performs the RK4 `delta` accumulation on
later stages (`src/z4c/z4c_tasks.cpp:146-178`). `ExpRKUpdate` applies

```text
u0 = gam0*u0 + gam1*u1 + beta*dt*u_rhs
```

over active cells. Both patterns are formulation-neutral. The driver also performs a
manual initialization exchange before time stepping (`src/driver/driver.cpp:564-575`),
so PC-GH needs the analogous sequence and a conversion to ADM where required.

## Finite-difference and RHS utilities

`src/utils/finite_diff.hpp` provides centered `Dx` overloads for scalar, vector, and
rank-2 tensor views at selectors 2, 3, and 4, corresponding here to second-, fourth-,
and sixth-order centered derivatives. It also provides `Dxx`, `Dxy`, and `Diss`.
Production PC-GH may use only `Dx<NGHOST>` and `Diss<NGHOST>` from these families. A
source test must scan production files below `src/pc_gh` and fail on any `Dxx` or `Dxy`
invocation; the utility header itself need not change.

Z4c validates `spatial_order`, derives a stencil selector, and verifies ghost-zone
support (`src/z4c/z4c.cpp:151-179`), then dispatches explicit template specializations
(`src/z4c/z4c_tasks.cpp:45-64`). This validation pattern is reusable.

The Z4c RHS uses compact device-local `AthenaPointTensor` scratch objects, one conformal
3-metric inverse per cell, inverse grid spacing from `mb_size`, and a separate KO kernel.
PC-GH should mirror that kernel organization. Z4c's RHS calls second-derivative helpers
and contains Z4c physics, so none of its equation assembly is reusable.

## Cell-centred communication and boundaries

`MeshBoundaryValuesCC` is the correct generic carrier for a flat 55-component view. The
reusable calls are `InitializeBuffers`, `InitRecv`, `ClearRecv`, `ClearSend`,
`PackAndSendCC`, `RecvAndUnpackCC`, and `ProlongateCC`. The path derives `nvar` from the
view and handles local copies and MPI for all faces, edges, and corners.

The constructor boolean `z4c`, member `is_z4c_`, `isame_z4c` buffers, and related
branches actually encode a high-order cell-centred transfer mode under a
formulation-specific name (`src/bvals/bvals.hpp:109-176`, `src/bvals/bvals.cpp:24-69`,
`src/bvals/bvals_cc.cpp`). Generalize this meaning before routing PC-GH through it.

`MeshBoundaryValues::Z4cBCs` and `src/z4c/z4c_Sbc.cpp` dereference Z4c state and apply
Z4c-specific extrapolation/Sommerfeld behavior. One-sided extrapolation mechanics can be
a software example for a clearly labeled early diagnostic outer condition, but these
functions are not GH characteristic constraint-preserving boundary conditions. Periodic
boundaries use generic exchange and are required for Minkowski and linear-wave gates.

## Restriction, prolongation, AMR, and load balance

Z4c selects the high-order branch with `RestrictCC(u0, coarse_u0, true)` before send and
`ProlongateCC(u0, coarse_u0, true)` after physical BCs. The branch is component-generic,
but the current high-order switch implements only `nghost == 2` and `nghost == 4`.
Because Z4c accepts selector 3, multilevel fourth-order runs have an uncovered transfer
case at this base. PC-GH must implement and test selector 3 or fail closed for that AMR
combination; it must not silently leave fine values unprolongated.

`MeshRefinement::RefineCC`, derefinement, load balancing, restart sizing, and buffer
packing special-case `pz4c` and `nz4c`. Each needs an explicit PC-GH case so all 55
fields move together. Transfer does not preserve PC-GH reduction or curl constraints by
construction. Qualification therefore requires separate pre/post-transfer norms for X,
Y, Q, and B reduction and curl families. Existing infrastructure provides no argument
for automatically resetting derivative fields.

`Z4c_AMR` supplies reusable control-flow patterns for tracker locations, radial shells,
minimum chi, and maximum chi differences. A distinct PC-GH AMR owner is preferable
because the current class dereferences `pz4c` and Z4c indices throughout.

## ADM storage and conversion boundary

`adm::ADM` stores generic `gamma_ij`, `K_ij`, `psi4`, alpha, and shift views. It also
provides device-safe `SpatialDet`, `SpatialInv`, and `Trace` helpers
(`src/coordinates/adm.hpp:20-103`), which are directly reusable.

Currently, `adm::ADM` aliases lapse and shift from `pz4c->u0` whenever Z4c exists and
otherwise allocates them in `u_adm` (`src/coordinates/adm.cpp:29-54`). PC-GH stores
`A=alpha^2`, so it cannot supply a lapse alias. ADM must retain its own alpha slot for
PC-GH, and PC-GH-to-ADM must write `sqrt(A)`. Copying shift too gives a clean ownership
boundary.

The useful software roles in `src/z4c/z4c_adm.cpp` are:

```text
ADM initial data -> conformal evolution state -> derivative initialization
evolution state -> ADM arrays -> generic consumers
```

PC-GH must implement those roles independently. Its forward conversion uses the
specified conformal definitions, sets `pi=-K` and `Lambda=Gamma`, and initializes
X/Y/Q/B with the evolution `Dx`. Its reverse conversion writes physical `gamma_ij`,
`K_ij`, alpha, and shift only for consumers; reconstructed physical derivatives must
never return to the PC-GH RHS.

## Existing initial-data and focused-test plumbing

The current pgens establish reusable software flows, not reusable PC-GH physics:

- `z4c_one_puncture.cpp` fills ADM wormhole data over active and ghost cells, applies a
  pre-collapsed lapse, converts to an evolution formulation, computes initial ADM
  diagnostics, and enrolls AMR refinement.
- `z4c_two_puncture.cpp` configures the external TwoPunctures library, interpolates its
  host-side ADM result onto the mesh, copies to device storage, converts, and finalizes
  the library object.
- `z4c_spectre_bbh.cpp` demonstrates host-side interpolation of external ADM data.
- `tests/z4c_linear_wave.cpp` demonstrates periodic analytic initialization, a final
  error callback, and resolution-oriented error output.
- `z4c_stability.cpp` demonstrates deterministic Kokkos random perturbations and history
  callbacks for robust-stability tests.
- `z4c_gauge_wave.cpp` demonstrates analytic ADM data over ghost zones, but its Z4c
  equations are not PC-GH test data.
- `tests/z4c_boosted_puncture.cpp` supplies an ADM boosted-puncture data path, subject to
  independent validation before it can support PC-GH evidence.

New focused PC-GH pgens should live in `src/pgen/pc_gh/`. The build accepts
`-D PROBLEM=<path-without-.cpp>` and adds `pgen/${PROBLEM}.cpp`, so a path such as
`pc_gh/minkowski` works without flattening the directory. Built-in dispatch in
`src/pgen/pgen.cpp` needs entries only for tests intended to join `built_in_pgens`.

TwoPunctures linking is conditional on the exact name `z4c_two_puncture` in the
top-level `CMakeLists.txt`. Add a PC-GH name or formulation-neutral feature condition
before building that pgen.

## Diagnostics, output, restart, and generic ADM consumers

Existing output and restart paths discover `pz4c`, use `nz4c`, and serialize
`pz4c->u0`. PC-GH needs independent variable names and explicit support in:

```text
src/outputs/outputs.hpp
src/outputs/basetype_output.cpp
src/outputs/history.cpp
src/outputs/restart.cpp
src/pgen/pgen.cpp
src/outputs/pdf.cpp, if retained
src/mesh/load_balance.cpp
src/mesh/mesh_refinement.cpp
```

The diagnostic layout must preserve separate GH, Hamiltonian, scaled momentum,
reduction, curl, algebraic, projection-correction, regularity, state-extrema, and
RHS-family outputs. The Z4c history reduction that combines constraints is not suitable
for PC-GH debugging.

Weyl extraction, waveform interpolation, compact-object trackers, horizon dumps, and CCE
are scheduled as Z4c members but operate substantially through ADM or generic
cell-centred data. Reuse requires separating scheduling and ownership from `pz4c`; no
PC-GH physics belongs in those consumers. This decoupling is needed before puncture
qualification requires horizon and wave evidence, not before the first algebra tests.

## Minimum framework integration surface

1. Add `pc_gh/*.cpp` to `src/CMakeLists.txt` and add a static no-`Dxx`/`Dxy` production
   source test.
2. Add `pc_gh::PcGh *ppcgh` to `MeshBlockPack`, construct it from `<pc_gh>`, construct
   ADM storage in the same run, and destroy it safely.
3. Extend `NumericalRelativity` with distinct PC-GH physics, task names, and queueing.
4. Generalize ADM lapse/shift ownership so PC-GH-to-ADM can populate generic storage
   without pretending `A` is alpha.
5. Generalize the Z4c-named high-order CC mode, then route PC-GH through communication,
   restriction, prolongation, refinement, and load balance.
6. Add the PC-GH initialization exchange and PC-GH-to-ADM calls to `Driver`.
7. Add restart read/write and ordinary output selection for all 55 fields before any
   evolution evidence is considered restart-safe.
8. Add formulation-specific diagnostics and names without collapsing constraint
   families.

## Explicit non-reuse decisions

The following must be newly derived or implemented:

- every PC-GH Einstein RHS term;
- Z4c Ricci, Gamma, Theta, gauge, speed, constraint, and Sommerfeld formulas;
- Z4c floors or guarded conformal divisions;
- every production attempt to use `Dxx` or `Dxy` on configuration fields;
- physical Christoffels or physical Ricci in the production PC-GH RHS;
- any old FO-GH/Ref-GH source, generated equations, reference geometry, reference frame,
  tetrad, controller, cache, or gauge-subtraction code;
- any claim that existing Z4c AMR transfer preserves PC-GH reduction/curl constraints;
- any claim that Z4c timestep, boundary, diagnostic, or qualification criteria prove the
  corresponding PC-GH property.

## Phase-0 conclusion

The `project/bbh` infrastructure can host a Z4c-shaped PC-GH module, but it is not
plug-compatible without the framework edits above. The reusable core is the
flat-array/tensor-view/Kokkos/task/communication/ADM software architecture. The
mathematics, constraint system, gauge source, characteristic speeds, boundary condition,
regularization, and scientific qualification remain unestablished until the independent
derivation and symbolic gates are complete.
