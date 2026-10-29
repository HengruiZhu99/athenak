# V4 duration analysis

V4 failed the bounded duration test. See the [outcome assessment](../README.md)
and [verified comparison](final-v1/comparison.json). This analysis implements
the frozen [protocol](../protocol.md); it does not modify or evolve any state.

From the repository root, after the complete campaign and terminal records
have been downloaded and verified, run:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/stability-dissipative-minkowski-v1/analysis/analyze.py \
  --v4-dir output/hyperbolic-core-shell/aurora/stability-dissipative-minkowski-v1 \
  --require-v4 --verify-payloads \
  --out /tmp/v4-duration-reanalysis
```

The output directory must be new. NumPy and Matplotlib are required. The three
previous controls are read from their immutable portable packages. For a
portable reproduction without the large local payloads, use:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/stability-dissipative-minkowski-v1/analysis/analyze.py \
  --v4-dir docs/hyperbolic-shift-3d/reproduction/aurora/stability-dissipative-minkowski-v1/analysis/raw-v4 \
  --require-v4 --out /tmp/v4-duration-portable-replay
```

Checkpoints and failed stages are not stored in Git. Available scientific input
bytes must still match the worker's hashes. The optional payload check streams
raw bytes only; it neither parses a failed checkpoint as a state nor admits it
for restart. `package_evidence.py` copied the nine small original records and
whitelisted terminal metadata to `raw-v4/`, rechecked all five payload hashes,
and separately inspected the 88-byte V4 checkpoint header. The native header's
`surface` count is a scalar count (`31*1536`), not a point count. It explicitly
records `admitted=0`; no state arrays were loaded.

`comparison.json` records source, executable, plan, input and admission bindings,
every replayed worker check, checks of the recorded trajectory, missing fields,
actual fit samples and outcome distinctions. `summary.csv` gives the accepted
endpoint, last fresh diagnostic, fit interval/count/rate/R² and last diagnostic
values. `recorded-diagnostics.csv` preserves the selected recorded values with
blank missing/nonfinite entries. PNG and PDF figures compare state deviation,
Hamiltonian and invariant conformal momentum RMS, algebraic residuals, null
closure, and Z/Theta RMS. No curve is continued to a later failure time.

The growth fit uses unweighted least squares of log state deviation versus
coordinate time. It selects the longest contiguous recorded interval within
`1e-10 <= state_deviation <= 1e-3`, choosing the earlier interval on a tie, and
requires at least four samples. A shorter or absent interval produces a null
rate, not a zero growth claim. No H fit is made near its cancellation/roundoff
floor. Initial seed amplitudes and different sampled intervals remain visible;
an endpoint delay alone is not evidence of a reduced growth rate.

The limits in the V4 protocol are also applied to every recorded sample, not
only to the last report. The isolated angular-Q CFL bound is taken from its
reported running maximum because there is no corresponding history column.
Passing it is not a coupled stability proof. Original algebraic histories were
not recorded and are explicitly unavailable. Completing this bounded
Minkowski run would not establish puncture stability, a wormhole-to-trumpet
transition, convergence, AMR, or long-duration stability.

The lightweight analysis controls can be run independently:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/stability-dissipative-minkowski-v1/analysis/analyze.py --self-test
```

These check disconnected fit-window selection, the four-sample minimum,
missing-fit semantics, and rejection of an absent threshold metric. The three
retained controls reproduce state growth rates 74.4915294803, 75.8060271199 and
72.4355186600 per coordinate time for the original, V2 and V3 respectively.
V4 gives 72.1950013224 from the same five recorded times, 0.15 through 0.35.
