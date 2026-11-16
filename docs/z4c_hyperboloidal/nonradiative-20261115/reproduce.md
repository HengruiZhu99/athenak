# Reproduction and retained products

Run commands from the repository root. The analysis needs Python, NumPy and
Matplotlib; recorded Aurora versions and binary hashes are in the compact
provenance. Runtime metadata was inspected on the service node after the runs;
the shared NumPy module hashes and actual analysis launcher paths are retained,
but per-rank interpreter binary hashes were not captured on compute nodes.
The primary run roots are:

```
local:  /Users/hz0693/research/results/nonradiative-20261115
Aurora: /lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-20261115
```

## Small checks

The existing MPI/Kokkos CPU build is `build-binary`. It has the focused tests
under `tst/z4c_hyperboloidal`; the new test checks the signed lapse identity and
an independent finite-difference tangent of the implemented endpoint projection.

```sh
cmake --build build-binary --target hyperboloidal_scri_rate_tests
ctest --test-dir build-binary --output-on-failure
python3 tst/z4c_hyperboloidal/scri_frame.py
python3 tst/z4c_hyperboloidal/scri_news.py
python3 tst/z4c_hyperboloidal/nonradiative_passivity.py build-binary/src/athena /tmp/nonradiative-passivity
```

The passivity command requires a fresh output directory and performs two steps
with two CPU MPI ranks. The inventory executables
`hyperboloidal_nonradiative_stationarity`,
`hyperboloidal_nonradiative_characteristics`, and
`hyperboloidal_nonradiative_gauge_admission` print measurements; they are not
science pass/fail tests. Gauge admission uses the independent metric-derived
K/lapse/shift oracle:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_gauge_admission.py \
  build-binary/tst/z4c_hyperboloidal/hyperboloidal_nonradiative_gauge_admission \
  /tmp/nonradiative-admission
python3 tst/z4c_hyperboloidal/nonradiative_pullbacks.py
```

## Inputs, executable, and native evolution

The committed input copies and run receipts specify the actual cases. The
same inputs can be generated in a fresh directory with:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_cases.py /tmp/nonradiative-inputs \
  --base-cycles 510 \
  --duration-reason 'Outgoing travel .05419-.08240 M; measure through .718246 M.'
```

N=12/16/20 use 510/680/850 RK4 steps, with dt proportional to 1/N. N12 halfdt
uses 1020 steps and records every other accepted step, giving the same physical
recorder cadence. All other principal cases record every step.

The Aurora build intentionally reuses the exact existing numerical objects from
`/lus/flare/projects/CompactBinaryMerger/hzhu/n12-full-scri-20261115` and its
referenced build roots. `aurora/build-nonradiative.py ROOT BASELINE` checks every
baseline source hash, requires that only `src/outputs/scri.cpp` changed, compiles
the recorder and test seed main, and hashes all reused objects before/after.
`aurora/build-nonradiative-seed.py ROOT` creates the separately receipted
`seed-v2/nonradiative_seed` used by every admitted run. The initial seed executable
in `build/` is retained for history and is not the admitted seed writer.
These overlay helpers require the documented baseline objects; they are not a
standalone portable build recipe. Exact compiler/link commands are in the
receipts. The source tree is sufficient for a fresh full build using this
repository's normal MPI/Kokkos configuration.

The inspected launchers specify account CompactBinaryMerger, queue debug,
`select=2`, 24 GPU ranks (12 per node), fixed CPU binding, full overlap, and
oneapi/release/2025.3.1. At most one task allocation ran at once. Original jobs:

```sh
qsub -l walltime=00:10:00 -v CASE=pilot source/tst/z4c_hyperboloidal/aurora/nonradiative-two.pbs
qsub -l walltime=00:20:00 -v CASE=pair-N12 source/tst/z4c_hyperboloidal/aurora/nonradiative-two.pbs
qsub -l walltime=00:30:00 -v TASK=native-N12 source/tst/z4c_hyperboloidal/aurora/nonradiative-analysis.pbs
qsub -l walltime=00:35:00 -v CASE=pair-N16 source/tst/z4c_hyperboloidal/aurora/nonradiative-two.pbs
qsub -l walltime=00:50:00 -v CASE=pair-N20 source/tst/z4c_hyperboloidal/aurora/nonradiative-two.pbs
```

Use a new run root or retain only one `pair-N*-JOB` candidate per resolution
when reproducing: the analysis driver rejects ambiguous duplicate candidates.
Do not rerun in occupied output directories. The historical pilot used the
saved `inputs-pilot`; the present generator describes the principal extension.
The run receipt checks seed/executable hashes and planned final cycle, so a
walltime stop with program exit zero cannot be mistaken for completion.
N16/N20 run extraction in the same allocation after the evolutions. N20 also
performs four zero-step restarts to evaluate absolute initial constraint norms;
these read immutable seeds and disable checkpoint/scri outputs. N12 was
analyzed separately with halfdt and angular/cadence controls. Submissions above
are a recipe, not automatic successor jobs.

## Analytic and extraction layers

`nonradiative_analytic_suite.py` defines the full exact-jet and actual-donor
matrix. The two-rank driver requires a valid global MPI rank environment and
uses one CPU rank per allocated node, with BLAS threads fixed at one. It snapshots
its Python sources before running. Original analytic job 9093456 retains its
successful cases and failed H=1 slice admissions; the H=16 follow-up and two
interrupted N20 row cases are included with corrected N12 analysis 9093556.
A separate follow-up can be launched with `TASK=analytic-followup`.

Saved `layouts/N12`, `layouts/N16`, `layouts/N20` contain native initial records
and topology epochs, with hashes in `layout-provenance.json`. An individual
row replay is, for example:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_sampling.py LAYOUT OUTPUT \
  --family rotation --parameter .8 --at .2 --dt .002
```

The Python callable `run(..., height=16)` selects the explicitly declared
strong-gauge analytic chart; the complete suite calls it directly. Exact jets
are supplied independently by `nonradiative_analytic.compact` and passed through
the production geometry/frame modules. `nonradiative_pullbacks.py` checks the
finite-radius coordinate conversion. None of these prescribed metrics changes
A or is passed off as an A evolution.

For any completed native case:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_native_analysis.py RUN_CASE ANALYSIS_CASE
python3 tst/z4c_hyperboloidal/nonradiative_analysis_controls.py ANALYSIS_CASE
python3 tst/z4c_hyperboloidal/nonradiative_pointwise_summary.py ANALYSIS_CASE
```

Substantial reanalysis belongs on allocated nodes. The controls script is used
only on N12 gauge in this campaign. `pointwise.npz` retains both full tensors,
signed lapse summands, penalties and chi/P/Theta/lapse pole/h values at the same
points/times. Original `.scri` records retain all sampled jets and normal lift
jets; `.hst` files retain the existing remaining pole/regularity groups. The
cache is derived data, not a substitute for those native archives.

The final plots and common-cut comparisons can be regenerated from the compact
collected native news/series and analytic case products using:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_report.py \
  --native RAW/analysis/native-N12-9093556 \
  --native RAW/analysis/native-N16-9093536 \
  --native RAW/analysis/native-N20-9093626 \
  --analytic RAW/analysis/analytic-9093456 \
  --analytic RAW/analysis/native-N12-9093556/analytic-followup \
  --output FIGURES
```

`nonradiative_compact.py` packages selected analysis/runs without copying large
native archives or caches. The report's comparisons intersect physical Bondi
intervals, retain common complex modes and use the existing polarization
convention. They include all original cuts in the shared interval so a narrow
sampled maximum is not discarded by coarse plotting. The halfdt tensor
comparison uses identical native coordinates and matched accepted times.
