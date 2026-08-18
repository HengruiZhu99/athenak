# AMR hierarchy record/replay

AthenaK can optionally record accepted adaptive-mesh hierarchies and replay the
same physical MeshBlock tree at the same coordinate times. The feature is
default-off and uses the ordinary production regridding, load-balancing,
restriction, prolongation, boundary, and reconstruction paths.

Configure it in `<mesh_refinement>`:

```text
amr_history_mode = off       # off, record, or replay
amr_history_file = history.jsonl
```

If `amr_history_mode` is absent or `off`, existing refinement behavior is
unchanged. `record` retains the normal refinement criterion and writes the
initial accepted leaf tree plus every later topology-changing accepted tree.
`replay` makes the history authoritative, clips the preceding CFL timestep to
land on each recorded physical event time, derives production refinement flags
from the requested leaf set, and verifies the accepted tree exactly.

## Canonical JSON-lines format

The first line is a header. It binds schema version, dimension, symmetry and
coordinate map, root level and root MeshBlock counts, physical-domain bounds as
hexadecimal floating-point values, periodicity, maximum AMR level, `Real` size,
and the AthenaK source commit. Cells per MeshBlock are provenance but may differ
between record and replay; this is what permits a 32x32 record to drive a 64x64
same-geometry replay.

Each remaining line is one accepted event with:

- contiguous event index;
- coordinate time in exact hexadecimal and round-trippable decimal forms;
- cycle for provenance only;
- the complete sorted leaf set as `(level,lx1,lx2,lx3)`;
- leaf count, maximum level, requested and accepted topology counts;
- deterministic tree and record checksums.

The embedded checksums are canonical FNV-1a 64-bit integrity checks, not
cryptographic provenance. Evidence bundles should additionally bind the full
history with SHA-256.

Replay rejects malformed or noncanonical records, duplicate/overlapping or
incomplete trees, invalid levels or root geometry, nonmonotonic event times,
incompatible source/geometry contracts, impossible one-transaction changes,
and any accepted hierarchy that differs from the requested tree. A replay
history is immutable. Replay writes a separate local ledger and never modifies
the authority file.

## Restart behavior

Restart output stores a restart-only carrier containing the mode, exact history
digest and byte length, last/next event cursor, post-event state, and current
tree checksum. On restart, AthenaK requires the same history bytes and tree and
rejects any command-line or input-file attempt to mutate the carrier. Record
restart appends only when the file is exactly the checkpoint-bound prefix;
replay restart neither repeats nor skips an event.

## Intended controlled-resolution workflow

Record a reference run, replay once at identical resolution as an identity
control, then replay at a different number of cells per MeshBlock while keeping
the root MeshBlock layout and physical block extents fixed. Compare only through
the last event available in the authority schedule. A history does not license
extrapolating or holding the final tree beyond its recorded endpoint.
