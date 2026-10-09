# Held stopped-failure diagnostic observer

This separate source is prepared for root review, initially motivated by the
failed `wave-map-N16-large-t2` process. It does not execute, edit, impersonate or
relax the c1b948 completed-result analyzer. It uses the exact same binary64
reader/ABI and accepted snapshot probe, with unchanged scientific guard values.
Every result always contains `partial_diagnostic_only=true` and
`accepted_native_run=false`, including a mechanically completed observer run.

Root must release the exact observer and recipe hashes plus each stopped failed
case's original launch-receipt hash and a fresh attempt directory. Only the nine
original fixed t2 specifications are eligible. Failed/nonzero native status is
required; successful cases belong to the separately held completed wrapper.
Original source/provenance before/after equality and a stable complete output
inventory are required. No concurrently running process's files are observed.
The initial failed receipt is preserved verbatim in source-context; all original
inputs, stdout/stderr, arrays and receipts remain untouched.

The first review covers the two separate stopped failures
`wave-map-N16-large-t2` and `c0-N16-large-t2`, with separate releases/attempts.
Native H/M/Z/Theta and metric diagnostics use the same geometric constraint
kernel for both inputs. The unchanged probe also reports a wave-map values-only
gauge-pole evaluation on each supplied state; for C0 fields that is explicitly
a comparison evaluation, not the evolved C0 gauge pole. Full probe stdout is
preserved without relabeling that distinction.

The observer retains all source/compiler/input/output/probe/reader/release pins
before and after the call, with original process return code and target intact.
It enumerates every saved RST, recording parse/structure failures and all active
finite/positive/SPD guards. The unchanged probe is called only after those
preconditions hold. Normal thresholds remain 1e-10 for pulse or1e-11 reference,
initializer2e-13, initial/reference constraints1e-9, reference drift1e-10 and
history/native RMS2e-11. Breaches are explicit diagnostics; none is converted
into a passing subset or repaired state. A malformed array is retained as a
failed observation and never sent to a native geometry kernel.

Full15-column native history is retained only if its original shape, finite
values, positive dt and strictly increasing times pass the unchanged guards;
otherwise raw history hash and parser/guard failure are retained, and array
comparisons explicitly remain unavailable. Missing t0 or exact-time pairing is
reported rather than synthesized. Console dt remains a six-digit observation,
distinct from binary64 RST/history timing. Histories, extrema, shell/radial
norms and available-array times are observations only. No finite-control or
completed-run acceptance, valid future state, continuation, PDE theorem, scri
closure or BH inner transition inference follows.

Held invocation shape:

```
PYTHONPATH=/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/python-deps \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Library/Developer/CommandLineTools/usr/bin/python3 -B observe_partial.py \
    root-observer-release.json wave-map-N16-large-t2
```

Root must capture complete observer stdout/stderr/exit for early malformed
release/schema guards before a fresh attempt directory can be established.
Subsequent protocol stops retain an explicit observer receipt. Observer exit0
means the diagnostic protocol finished, never that native evolution passed.
No compiled/native/array/history query has been made from this source-only tree.
