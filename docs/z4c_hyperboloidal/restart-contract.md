# Native hyperboloidal restart contract

The hyperboloidal module uses the original AthenaK `RestartOutput`, native mesh
header, versioned logical locations, atlas identity, load balance, and MPI-IO
block addressing. It adds optional physics metadata and a block payload to that
format. Cartesian v1 reads, v2 reads, and v2 writes retain their existing bytes.

## Stored state and ownership

A checkpoint stores each native leaf GID once. The standard native tree records
retain all leaf locations and refinement levels, including shell angular tiles,
radial children, and evolved overlap ownership. A shared restart file can be
read using a different rank count: the native load balancer chooses the new
local GID range and each rank reads only that range. Per-rank files require the
same rank count and per-rank GID ownership and reject a changed partition.
Before loading per-rank payloads, collective byte comparisons require identical
checkpoint time, timestep, cycle, topology, atlas and physics identity, and
load-balance costs on every rank; a fixed 64 KiB buffer suffices. This rejects
mixed sets whose saved times, topology or parameters differ.

Each fixed-stride hyperboloidal block record contains 39 binary64 scalars per
slot: the 31 `Fields` values followed by the eight explicit `ChiRecord` members
in this order:

1. `minimum`
2. `accepted_minimum`
3. `floor_count`
4. `nonpositive_count`
5. `nonfinite_count`
6. `max_correction`
7. `sum_correction`
8. `first_floor`

The bulk slot index is `((k-base)*ny+j-base)*nx+i-base`, where
`base=origin-overlap` and `na=N[a]+2*overlap`. The following `N[2]*N[1]` slots
hold the owned outer face in `k*N[1]+j` order. Unowned cube slots and surface
slots of blocks without an outer face are deterministic zero padding. All
actual evolved overlap cells are stored; difference ghosts are omitted.

The fixed stride matches the original native block I/O scheme. Packing and
unpacking allocate local rank buffers only. RK stage registers, donor plans,
ghosts, and reference caches are reconstructed rather than checkpointed; a
checkpoint represents an accepted complete step.

## Metadata and validation

The physics header follows the existing optional standard-Z4c/tracker/RNG
header and precedes the native per-block byte count. It consists of eight
uint64 values: magic `0x4859504552525354`, schema version 1, identity string
length, record scalar count, writer rank count, writer first GID, writer local
block count, and per-rank-file flag. The exact UTF-8 identity follows.

The identity includes field/record/formulation versions, RK4 policy, binary64
size, block dimensions, overlap and ghost widths, KO, chi floor, scri penalty,
requested timestep, and every resolved gauge/conformal-profile parameter.
Floating values use exact hexadecimal formatting with the classic locale.
Native mesh-header atlas identity independently binds the chart geometry.
Parameter overrides that change this mathematical identity are rejected.

Byte arithmetic is checked before block transfers; a block transfer cannot
exceed the MPI integer count range. The reader collectively rejects invalid
metadata, parameter identity, record stride, file extent, short reads, and
nonfinite state. Metadata and payload truncation are detected before dependent
collective operations. This is structural validation, not a cryptographic
checksum of the payload.

## Restore lifecycle

The native restart problem-generator constructor reads the module metadata and
local records. It initializes geometry/reference caches before unpacking the
saved evolved bulk, scri values, and cumulative chi records. The hyperboloidal
problem-generator restart callback then returns without resetting them.
Driver initialization builds local communication plans and regenerates ghosts
from the restored state through the ordinary native task path.

## Validation runner

`python3 tst/z4c_hyperboloidal/restart_roundtrip.py --binary build-native/src/athena
--output build-native/restart-roundtrip --target-ranks 8` exercises the actual
native executable, not a substitute evolution driver. It writes with two ranks,
injects distinct nonzero values into all eight record members, reloads with eight
ranks and then two ranks, and compares all field and record payload bits. Both
the 106-leaf N12 topology and the 113-leaf outer-shell SMR topology are included.
It also tests parameter mismatch, metadata/identity/stride/payload truncation,
overflow-sized strides, and nonfinite state. Output includes executable/source
provenance and exact payload hashes.

Add `--trajectory` to compare nine continuous complete RK4 steps on two ranks
with one step, a checkpoint, and eight further complete steps using the selected
target rank count. All state and all chi-record bits must match.

Use `--only-trajectory` to run just the continuation comparison. Use
`--only-per-rank` for coherent per-rank reads, changed-ownership rejection,
mixed-generation rejection, and a payload truncated on one rank only.

## Observed local validation

The 2026-11-08 local CPU MPI checks passed:

- N12 shared-file 2→8→2: 24,704,784 scalar slots bitwise, with 1,842,048
  nonzero injected chi-record values across 230,256 evolved bulk/scri points.
- Outer-shell SMR shared-file 2→8→2: 26,336,232 scalar slots bitwise, with
  1,942,272 nonzero injected record values across 242,784 evolved points.
- Eight shared-file rejection cases: changed KO and timestep, truncated metadata,
  identity, stride and payload, overflow-sized stride, and nonfinite state.
- Per-rank coherent 2→2: both complete rank payloads bitwise; changed ownership,
  mixed checkpoint generations, and one truncated rank rejected collectively.
- Nine continuous RK4 steps on two ranks exactly equal one step on two ranks,
  a native checkpoint, and eight more steps on eight ranks. All 24,704,784
  serialized scalar slots match bitwise, including every chi-record member.

The compact results and provenance references are in
[`evidence/native-restart-validation.json`](evidence/native-restart-validation.json).
The final per-rank consistency check changed only the restart helper and its
object after the independent old-versus-native trajectory validation; the other
387 runtime sources and 193 native objects remained identical. The final dynamic
restart comparison used that final build. These are correctness checks, not GPU
performance measurements.
