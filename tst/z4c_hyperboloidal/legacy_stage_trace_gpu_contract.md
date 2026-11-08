# Pinned GPU Evolution accessor and comparison contract

This is a test-only entry point around the unchanged historical `Evolution`
implementation included from the pinned `coupled_mpi.cpp`. The builder replaces
only the original `run.cpp` translation unit and its object/output, retaining the
archived compiler, definitions, floating-point flags, device target, linked
objects and spherical library. All 772 archived source hashes and the immutable
original executable are checked. No runtime source is changed. The new entry
point is independently validated against that unchanged executable; matching
source alone is not a numerical equivalence result.

Run `legacy_stage_trace_gpu INPUT` in a fresh directory. The input must specify
the fixed N12 configuration, KO=.1, chi floor=1e-10, the exact nominal timestep
0.00046944203735617445, and one or eight complete steps. As in the original
wrapper, one initial diagnostic RHS occurs before stepping and one final
diagnostic RHS occurs after stepping. Both phases are recorded separately.
No new RHS is inserted inside RK. `first-rhs` reads the persistent first-stage
rate arrays immediately after the first complete step; the spherical policy
retains the raw rates without logarithmic transformations. All eight actual
per-cell ChiRecord members are copied from the old Evolution ledger, rather
than reconstructed from aggregate diagnostics or compensation restart arrays.

## Format and ordering

Each rank writes `LABEL-bulk-rankR.bin` and `LABEL-scri-rankR.bin`, including
empty surface files. Required labels are geometry, accepted0, accepted1,
first-rhs and final-snapshot; an eight-step run also requires accepted8.

The 40-byte header contains a 16-byte magic (`LEGACY_TRACE_V1`, 15 characters,
followed by one NUL), uint64 row count, and uint32[4]
`{6,nvalues,region,1}`. Region 0 is bulk and region 1 is scri.
Each row is int32[6] followed immediately by float64[nvalues], without padding.
Keys are legacy ordinal, old block index, k, j, i, face. Scri i is 12. Values
are three actual cached coordinates for geometry; 31 stored-component rates
for first-rhs; otherwise 31 stored fields followed by
`minimum,accepted_minimum,floor_count,nonpositive_count,nonfinite_count,max_correction,sum_correction,first_floor`.
The tested platforms store these values little endian.

The comparator requires every one of 229392 evolved bulk points, including
overlap, and all 864 scri points exactly once in every required label. It checks
finite values, exact payload length and consistent point keys. The independently
validated mapping supplies legacy-ordinal to native-GID/canonical-ordinal and
integer cell indices. It must form a complete 106-leaf bijection. The mapping's
CPU coordinates are excluded from exact GPU comparisons. Actual old and native
geometry dumps provide both bulk and scri coordinate equality checks. CPU map
weights are used only to report weighted differences; all acceptance tests are
pointwise.

## Admission and numerical gates

The comparator first admits the immutable original control's binary/configuration,
input, complete step history, clocks, SP1 header/topology/physics identity and
full snapshot versus checkpoint bytes. Control and accessor final snapshots
must have the same step count and rank count. The accessor build provenance
must bind its C++ source and all pinned sources to the archived command.

Before seeing the GPU trace results, admission was fixed to the already adopted
short gates: `1e-12 + 5e-10*abs(reference)` for fields and rates, with
`1e-14 + 5e-10*abs(reference)` for chi. These also apply to rebuilt-accessor
initial/final fields versus the immutable executable. Byte equality is reported
separately and prominently; a different translation unit can change code
generation even when the mathematical source and compiler flags are unchanged.
Actual geometry and clocks/timestep identities must agree exactly. Numerical
clock residuals use the short scalar gate. No tolerance is fitted to results.

Native accepted0 is after Driver initialization and boundary fill; legacy
accepted0 is after the original initial diagnostic RHS. Accepted1/8 are after
the completed RK step with no extra diagnostic RHS. Final-snapshot is after
the original final diagnostic RHS. These phase distinctions are retained in
the report, including an optional accepted1 comparison with the immutable
one-step executable's final diagnostic snapshot. Both the legacy accessor and
native accepted1 fields are compared directly with that unchanged one-step
control at the short/chi gates; the native result is not admitted solely through
agreement with the accessor. The final diagnostic RHS must
leave the actual old ChiRecord ledger byte-identical to its accepted state.

All eight old/native per-cell records are compared. The two raw minima use
the existing chi gate; the six event/correction/first-event records must agree
exactly for the admitted zero-floor short run. Sentinel values are finite and
are retained. Native per-step clock maxima are computed from the four existing
stage records and compared with the immutable control's `stage_clock` column.
The frozen native observer does not record CFL; maximum RHS is not substituted
for CFL. This limitation remains explicit.

All native labels, finite values, ownership keys and stage-clock ordering are
validated, including the exact input/output/accepted transitions. Separate
native clean/poison and 24/96 audits require byte equality of every field, RHS,
record, geometry value and all 32 stage records. A short-gate match against the
legacy operator is not described as byte equality.

## Comparator invocation

```
python3 legacy_stage_trace_compare.py \
  --mapping MAP_DIRECTORY \
  --legacy-trace LEGACY_ACCESSOR_DIRECTORY \
  --legacy-build-provenance ACCESSOR_BUILD/provenance.json \
  --legacy-control IMMUTABLE_CONTROL/output \
  --legacy-staging CONTROL_STAGING.json \
  --native-trace NATIVE_CLEAN_DIRECTORY \
  --native-trace NATIVE_POISON_DIRECTORY \
  --legacy-one-control IMMUTABLE_ONE_STEP/output \
  --legacy-one-staging ONE_STEP_STAGING.json \
  --output NEW_ANALYSIS_DIRECTORY
```

`--native-trace` is repeatable, including other admitted decompositions.
The one-step control arguments are optional. A one-step staging record binds
its actual `input_sha256` and the eight-step `source_input_sha256`; comparison
proves that only `nlim=8` changed to `nlim=1`. If that record is omitted, the
original eight-step staging hash is explicitly treated as the derivation
source, not attributed to the one-step input. Results include hashes of all
consumed files and comparison sources. Input artifacts are read-only and the
output directory must be new. Any missing, malformed or failed evidence makes
the command fail; no scheduler submission or evolution is performed.
