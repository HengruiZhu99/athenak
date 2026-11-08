# Same-state FastFlow monopole warm-start comparison

The additional native restart compact-radius comparison failed at unchanged
state and time: the mean coordinate radius changed from
`0.016863854480305415` to `0.016863917882895093`. After converting lengths to M units by multiplying by 3, the
difference `1.9020776903522219e-7` exceeds that test's original compact-radius
gate. This failure is preserved. No solver parameter, runtime source, or
tolerance was changed to obtain the evidence below.

An isolated replay loaded the actual cycle-one native checkpoint and supplied
the same sparse native physical-geometry callback to two solvers: the current
FastFlow object and a separate copy of the unchanged predecessor FastFlow.
The latter was mechanically renamed and put in a separate namespace to avoid
symbol and enum collisions; its mathematical code and stopping rules were
unchanged. Both used the native FastFlow optimization and floating-point flags. All predecessor,
transformed, harness and linked-object sources were hashed before execution.

The exact two seeds were read from the native checkpoint parameter headers:

| Monopole seed | Iterations | Mean coordinate radius | Expansion RMS |
|---|---:|---:|---:|
| Previous cycle: 0.05977616725771618 | 11 | 0.016863854480305415 | 4.9494508055859628e-5 |
| Saved current cycle: 0.059780807629455139 | 2 | 0.016863917882895093 | 4.3855884612460574e-5 |

For each seed, current and predecessor results are exact for every reported
quantity. Complete summary, harmonic-shape, and per-iteration verbose files
also compare byte for byte. Thus the shift is preserved predecessor behavior,
not a changed restart field, changed MPI ownership, or changed native solver.
The separate two/eight-rank restart control agrees for the compact radii too.

`InitialGuess()` itself is byte-for-byte source-identical to the predecessor:
it resets all non-monopole coefficients and starts with the retained `last_a0`.
`FastFlowLoop()` initializes the previous mass to zero, so a nonzero-mass warm
seed cannot converge at its first mass comparison. It performs one flow update
before the saved-seed search satisfies the existing mass-change criterion.
The second search lowers expansion RMS. Its mass and areal radius change much
less than its compact radius; this does not make the separate compact-radius
comparison pass.

This audit establishes preservation of the existing warm-start algorithm. It
does not establish invariance of the returned compact shape under repeated
searches, nor does it authorize changing a numerical gate. The N12 areal-radius
comparison and complete restart state/history comparisons retain their own
admission criteria.

Evidence: `evidence/native-fastflow-warmstart-equivalence.json`. Isolated source,
compile commands, executable and raw logs are under
`build-native/fastflow-warmstart-oracle`; the original failing restart artifacts
remain under `build-native/history-restart-v1`.
