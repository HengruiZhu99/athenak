# Saved evidence at the user-requested pause

[Comprehensive review](../../../docs/pcgh-scheme-review-20260907/README.md)

- `stop-record.json`: owned jobs/processes terminated and verification.
- `remote-stop-inventory.json`: final printed progress and checkpoint inventory, not inferred final checkpoint times.
- `first-bad-summary.json`: intrinsic uniform32 terminal cell and variables.
- `intrinsic-first-bad-rank0.txt.gz`: complete preserved first-bad dump, losslessly compressed.
- `uniform32-prefailure-analysis.json`: independent primary-field physical/reduction diagnostics from the already-analyzed last checkpoint at 11.0067479023M.
- `uniform32-temporal.json`: previously completed early temporal comparison.
- CPU logs: previously completed and interrupted runs, including the recovered uniform64 half-step T=2 completion.
- `remote/`: copied existing CUDA/MPI smoke summaries, progress records, logs, histories and health output. Controller files can still say running because their processes were terminated; the stop record supersedes those status strings.
- `legacy-raw/`: 44 compact artifacts recovered read-only from the originating legacy production directory, with original paths/hashes in `manifest.json`. Includes waveforms, histories, logs, input and executable/source provenance. Existing slice figures are in the document; full slice binaries and restarts were not duplicated here.
- `figure-manifest.json`: hashes of the nine unchanged existing figures.
- `compression-manifest.json`: original filenames and SHA256 values mapped to `.gz` storage for larger text artifacts. The legacy manifest names the original uncompressed files; use this mapping to locate compressed counterparts.

Cancellation is not a numerical failure or a completed qualification. No intrinsic binary evolution occurred. No scientific test or solver build was run during this review. An unfinished performance-only patch is preserved outside Git at `/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/review-pause-20260907/unfinished-health-optimization.patch`; it was not used by production runs.
