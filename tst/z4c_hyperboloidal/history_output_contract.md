# Native hyperboloidal history output gates

These gates precede the first run. `history_output.cpp` links the actual native
objects, substituting only a test main. The synthetic case tests output reduction
and format, not physical constraint formulas or horizon finding. The real case
calls the actual diagnostic coordinator through `HistoryOutput::LoadOutputData`;
independent pointwise/quadrature/horizon oracles establish its numerical meaning.

The appended physics identifier must leave existing identifiers unchanged and
write a distinct `.z4c_hyperboloidal.hst` file. The ten columns are full volume,
H/M/Theta/Z RMS, exterior volume and exterior H/M/Theta/Z RMS. Both volumes are
compact-coordinate Jacobian quadrature volumes. The loader supplies ten local
raw sums; the existing MPI sum runs before either RMS normalization. Native
coordinate time and dt are retained, with no implicit factor of three.

The synthetic case partitions the same 24 weighted samples over two and eight
ranks. Local volumes and local RMS values differ; odd-numbered ranks have no
exterior contribution. Compare the resulting global weighted RMS to the
independent arithmetic over all 24 samples. The two decompositions must write
identical files. A second row replaces raw data, checking repeated output does
not accidentally reuse already normalized values. Zero global exterior volume,
a negative global square integral and a nonfinite value on one rank must fail
collectively before a history row is written. Existing hydro, MHD, ordinary Z4c
and user-defined history records retain raw sum behavior and existing filenames.
`user_hist_only` must suppress the new expensive hyperboloidal diagnostics.

The real case initializes the preserved N12 reference with the native Driver,
then loads/writes its history twice at the same accepted state. Every loader
value must equal the module's rank-local raw sum, and every label must retain
the documented meaning. Independently reduce those raw sums for comparison to
the writer's global RMS. Repeated calls must return exactly the same local sums
and rows. Compare every owned bulk/scri field and all eight ChiRecord members
byte for byte before/after both calls; require all RK bases/accumulators and
RHS buffers to retain distinct sentinels. Thus history output must neither
advance state nor evaluate a complete RHS as a hidden dependency.
The coordinator's counters must increase by exactly one refresh and one
horizon search across all four repeated consumers at that accepted state.
Then perturb an independent Theta value on rank zero only, invalidate only
that rank, and require every rank to refresh/search exactly once. Restore that
value exactly and repeat the same one-rank invalidation check. Finally change
only cycle/time and require another refresh/search/write, with no owned state
change. Every phase rechecks all owned fields/records, RK/RHS sentinels and
stage observations. Each phase calls the consumer twice; the second call must
reuse its cache. The FastFlow summary must have exactly four entries, with
cycles `0,0,0,1`. Warm-start numerical results are not required to equal the
original search bitwise.

Cross-rank synthetic output is exact. Real output uses the adopted short
arithmetic gate `1e-12+5e-10*abs(reference)` and reports exact agreement and the
maximum absolute difference separately; local/device/MPI summation orders can
differ. This initial-state comparison does not replace the saved N12 history
comparison protocol. Preserve source, native-object and executable provenance
and verify no source/object changed during each test.
