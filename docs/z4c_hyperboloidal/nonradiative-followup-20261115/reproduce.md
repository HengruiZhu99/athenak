# Reproducing the wider-pulse follow-up

The parent commit is `54ecbcf5865a1b29ed41eba9245db838148b779c`. Its baseline
and analytic tests remain in `../nonradiative-20261115/`. This follow-up adds
a wider native gauge test and a separately controlled reference-jet correction.
Do not mix their executable identities.

## Locations and immutable snapshots

- Worktree: `/Users/hz0693/.codex/worktrees/hyperboloidal-binary-integration/hyperboloidal`.
- Local raw/collected results: `/Users/hz0693/research/results/nonradiative-followup-20261115`.
- Aurora raw results: `/lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-followup-20261115`.
- Parent Aurora results: `/lus/flare/projects/CompactBinaryMerger/hzhu/nonradiative-20261115`.
- Committed compact products: `../evidence/nonradiative-followup-20261115/`.

`source.tgz` / remote `source/` freezes the N12 launcher and analyses before
tensor attribution was added. `source-v2.tgz` / `source-v2/` freezes the N16/N20
Python analyses and launch script. The latter contains no production C++ source.
`reference-fix-source.tgz` / `reference-fix/source/` freezes the candidate C++
source. `source-halfdt.tgz` / `source-halfdt/` freezes the separate timestep
control and reference pointwise comparison. Snapshot hashes and manifests are
retained with the compact provenance. `source-halfdt-preflight` is an unused
preparation snapshot, retained for audit; no job was submitted from it.
`source-analysis-v2` is the immutable analysis-recovery snapshot. The preceding
`source-analysis-fix` snapshot was superseded after import preflight; its job
9094380 was cancelled while queued and never received nodes. Never overwrite
a running job's snapshot.

The principal wider runs use the original executable, SHA-256
`a577cabc82d15c2aa47cb38a2ce5bdc37cbcd64ac204eefc2690c45de6764885`.
Only the initial-data test executable is rebuilt to accept center, half-width
and shape arguments. The candidate reference executable has SHA-256
`5961115430a56a30efa42f770862119e1c55df082219db4cc50e545328d74721`.
Build receipts contain source manifests, compiler/link commands, reused object
hashes, and replaced-object mappings. The corrected build replaces the three
objects consuming `SphericalReference3D`; all original objects remain unmodified.
The saved symbol-owner inventory and complete source-consumer search identify
those consumers; the source search also covers fully inlined calls.
The final header adds an inner-gauge applicability guard after the immutable
Aurora candidate was built. For the campaign's validated flat-Omega profiles,
`r>=outer_radius` implies `r>flat_radius`, so its active derivative branch is
identical at every radius. The original candidate source remains archived.
`reference-guard-regression.json` tests the non-flat counterexample; the
separate CPU guard comparison checks initial and two-step payload equality.

## Local checks

From the worktree, use the configured CPU build and Python environment with
NumPy/Matplotlib. The persisted `ctest.log` records all 15 focused tests.

```sh
cmake --build build-binary --target hyperboloidal_reference_accuracy_tests
ctest --test-dir build-binary -R hyperboloidal --output-on-failure
python3 tst/z4c_hyperboloidal/nonradiative_pulse_resolution.py /tmp/pulse-resolution.json
```

The before/after stationary-reference witness is in `reference-regression.json`;
the old header overlay and deliberately failing executable remain under the
local raw root. The failure disables core dumps. The reference-check helper
runs fresh initializations and two-step CPU/MPI comparisons with the original
and corrected executables. Its receipt includes commands and checkpoint
payload comparisons. The zero-step data are bitwise equal; later differences
are expected because the reference derivatives have changed.

## Aurora execution

Use the existing account `CompactBinaryMerger`, `debug`, two nodes, at most
one task allocation at a time. The inspected queue allows 1–2 nodes and
5 minutes–1 hour, with one running and one queued debug job per user.
These launchers use 24 GPU ranks for evolution and two CPU ranks for analysis.
The fixed environment is `oneapi/release/2025.3.1`, with the existing
`n12-full-scri-20261115/analysis-env/bin/python` (Python 3.11).

Run from the remote result root, after preparing the corresponding immutable
snapshots, test seed, candidate reference executable and parent inputs:

```sh
qsub -v RESOLUTION=12 source/tst/z4c_hyperboloidal/aurora/nonradiative-wide.pbs
qsub -W depend=afterok:9093867 -v RESOLUTION=16,SOURCE_SNAPSHOT=source-v2,REFERENCE_PROBE=1 source-v2/tst/z4c_hyperboloidal/aurora/nonradiative-wide.pbs
```

Replace dependency IDs when reproducing. N20 uses `RESOLUTION=20` and
`SOURCE_SNAPSHOT=source-v2`, without `REFERENCE_PROBE`. The separate
`source-halfdt/tst/z4c_hyperboloidal/aurora/nonradiative-wide-halfdt.pbs` follows
the principal runs. It generates a fresh half-step seed, verifies all fields
bitwise against the wider full-step seed, and checks the requested-dt identity
and checkpoint timestep differ by exactly two. It also compares the original
and corrected reference controls pointwise on matching records. Its second
analysis rank repeats extraction at lmax=4 and 8, alongside the principal
lmax=6 at stride three, and compares common modes through ell=4.

The half-step native-rate T comparison samples every third dense baseline
record and compares identical physical times; it isolates the evolved-state
difference. Its accepted-record T_fd comparison also changes the differentiation
stencil width, so it is not a pure timestep measurement. The news timestep
comparison uses the baseline stride-three archive and the half-step stride-six
archive, giving identical physical recorder cadence.

The drivers assert target cycles, retain command durations and hashes, and
write `exit-code.txt`. Check scheduler completion and exit status as well as
this file; a missing file is not success. Do not rerun completed evolutions if
only analysis needs recovery. Histories, `.scri` records, epochs, restart
checkpoints, full pointwise arrays and metric caches remain in the raw roots.

## Analysis recovery

Half-step job 9094227 completed its evolution, then analysis rank 1 imported
the test CLI `scri_news.py` instead of the extraction module. MPI terminated
the companion analysis rank. The original `exit-code.txt` remains 143, with
`evolution-exit.txt` equal to zero; the failed analysis log and partial cache
are retained. Service-node preflight also identified an unwanted Matplotlib
dependency in the common-cut comparison import. `nonradiative_alignment.py`
now contains the unchanged comparison routine without plotting imports; local
before/after comparisons are bitwise identical for both families, all modes
and ell≤4. The reporting helper reuses that routine.

The recovery launcher uses two CPU ranks on two debug nodes for at most
20 minutes, with no evolution command:

```sh
python source-analysis-v2/tst/z4c_hyperboloidal/nonradiative_wide_recover.py --check-imports
qsub source-analysis-v2/tst/z4c_hyperboloidal/aurora/nonradiative-wide-recover.pbs
```

Use the documented analysis-environment Python in the first command. The
recovery receipt binds the new source manifest and scheduler job to the
original evolution checkpoint, checking its hash before and after analysis.
Completion checks require both a successful evolution and successful recovery;
they do not relabel the original scheduler failure as success. Recovery costs
are included separately. Original partial analysis moves to `analysis-attempt1`;
the original failure log is unchanged.

## Collection and plotting

On Aurora, run `nonradiative_wide_compact.py ROOT RUN OUTPUT.tgz` from the
`source-analysis-v2/tst/z4c_hyperboloidal/` snapshot to package completed or
explicitly recovered run receipts, histories and compact analyses. It emits a per-file
SHA-256 manifest. After transfer, verify that manifest before plotting:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_wide_report.py \
  /Users/hz0693/research/results/nonradiative-followup-20261115 \
  /Users/hz0693/research/results/nonradiative-20261115 \
  /Users/hz0693/research/results/nonradiative-followup-20261115/plots
```

The report helper requires all six principal cases unless `--partial` is
explicitly used. It preserves all 45 complex modes through ell=6 on common
Bondi cuts, separate absolute control and gauge errors, native-time tensor
comparisons, constraint histories, and the isolated reference correction.
Its plotting interpolation is for comparison, not a background subtraction.
The original dense and stride-three N12 extractions are retained separately.


After every allocation finishes, the bounded-memory checkpoint audit checks all
31 state fields and eight persistent chi-record scalars, cycle/time endpoints,
and the final-checkpoint SHA-256 values against the evolution receipts. Run
this small read-only check on Aurora, using the final audit script and the
archived restart reader:

```sh
/lus/flare/projects/CompactBinaryMerger/hzhu/n12-full-scri-20261115/analysis-env/bin/python \
  source-analysis-v2/tst/z4c_hyperboloidal/nonradiative_checkpoint_audit.py \
  . checkpoint-audit.json
```

Copy that receipt to the local raw `receipts/checkpoint-audit.json`. The final
compact-evidence assembler verifies completed scheduler receipts, input and
checkpoint hashes, seed preservation, executable identities, collection
manifests, finite products and the presence of every control before creating
source/artifact manifests:

```sh
python3 tst/z4c_hyperboloidal/nonradiative_wide_evidence.py \
  /Users/hz0693/research/results/nonradiative-followup-20261115 \
  docs/z4c_hyperboloidal/evidence/nonradiative-followup-20261115
```

The compact CSV inventory normalizes line endings; original bytes and hashes
remain in the raw root. The runtime receipt inspects the shared analysis Python
and NumPy installation from a service node and matches the parent campaign's
hashes; it is not an attestation collected on every compute rank. The separate
`local-report-runtime.json` records Python/NumPy/Matplotlib and executable/core
hashes for local compact-product interpolation, plotting and assembly. Immutable
run-source manifests and the final source manifest intentionally differ where
reporting and guard checks were added after the runs were launched.
