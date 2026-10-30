# Field-batched radial derivatives — unchanged retry

This is a fresh plan/output directory after job 8987469 failed its startup
storage guard before running any tests. It uses exactly the same frozen V24
source and four executables, V23 sparse control, initial data, eleven cases,
419 checks and one-node/twelve-tile CompactBinaryMerger debug resources.
The campaign directory changes, and the planned output-space allowance doubles
from 520 MB to 1,040 MB to cover the observed allocated-byte accounting. No physical or acceptance thresholds
are relaxed. The [verified build](../preflight-radial-field-batch-v1/build-v24/)
and its provenance remain in the preceding campaign directory.

The independent reader's default build directory intentionally points to the
previous campaign's immutable V24 build. Complete pulse restart and puncture
control/candidate checkpoints must be byte-identical. No GPU speedup or fidelity
result is yet available.

Nineteen old terminal-run checkpoint and failure-capture copies were verified
against local copies by size and SHA256, then removed from Aurora. The logical
file total was 690,845,888 bytes; the conservative allocated-byte footprint fell
from 4,883,963,904 to 3,502,080,000 bytes. The 100 MB atomic-checkpoint reserve
remains separate. Cleanup scripts, manifest and receipt are retained here.

Job **8987760** was submitted with 357,749,288 bytes of headroom after the
1,040 MB output allowance and separate 100 MB checkpoint reserve.
