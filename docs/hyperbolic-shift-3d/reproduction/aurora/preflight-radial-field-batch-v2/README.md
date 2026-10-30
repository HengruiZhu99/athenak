# Field-batched radial derivatives — verified retry

This is a fresh plan/output directory after job 8987469 failed its startup
storage guard before running any tests. It uses exactly the same frozen V24
source and four executables, V23 sparse control, initial data, eleven cases,
419 checks and one-node/twelve-tile CompactBinaryMerger debug resources.
The campaign directory changes, and the planned output-space allowance doubles
from 520 MB to 1,040 MB as a conservative additional allowance for allocation
overhead and concurrent campaign growth. No physical or acceptance thresholds
are relaxed. The [verified build](../preflight-radial-field-batch-v1/build-v24/)
and its provenance remain in the preceding campaign directory.

The independent reader's default build directory intentionally points to the
previous campaign's immutable V24 build. Complete pulse restart and puncture
control/candidate checkpoints must be byte-identical. The verified result is a 1.921x per-step speedup over the V23 sparse control,
with complete puncture checkpoints byte-identical; see the [comparison](../../radial-field-batch-v1/README.md).

Nineteen old terminal-run checkpoint and failure-capture copies were verified
against local copies by size and SHA256, then removed from Aurora. The logical
file total was 690,845,888 bytes; the measured root footprint changed
from 4,883,963,904 to 3,502,080,000 bytes. Other work can change the shared root
concurrently, so this difference is not proof of twice-logical allocation for
those removed files. The 100 MB atomic-checkpoint reserve
remains separate. Cleanup scripts, manifest and receipt are retained here.

Job **8987760** was submitted with 357,749,288 bytes of headroom after the
1,040 MB output allowance and separate 100 MB checkpoint reserve.

## Verified outcome and evidence collection

Job 8987760 finished in PBS state F, exit 0. All eleven cases, 419 independently
recomputed checks and 63 worker-bound artifacts passed. The independent reader
replayed every component rank log and checked full checkpoint content/configuration,
source/build/binary bindings, rank placement and timer accounting.

The first full-archive SSH transfer disconnected after 113,050,876 bytes and was
not used as verified evidence. After reconnecting, each of the six remote
checkpoints was freshly rehashed against the new worker manifest. All six hashes
and lengths matched complete local files from the independently verified V23
boundary-profile campaign. `reuse-verified-checkpoints.py` verified these matches
and copied full files into this campaign's local evidence directory. It did not
reconstruct or synthesize any fields. The full V24 control/candidate equality
was also checked natively on Aurora by `execution-audit`.

A 71,448-byte archive contains the new reports/logs/metadata and freshly rechecked
remote checkpoint hashes. Its SHA256 is
`55f08e1d67dd4d48584ed48e7b2c15f280e868bd94c0ed1cee5d4d39a4f5835a`.
`stream-small-evidence.py`, `unpack-evidence.py`, `archive.json`,
`checkpoint-hashes-rechecked.json` and `checkpoint-cache-verification.json`
retain the collection mechanism and proof. The independent verifier then read
and compared all six complete local checkpoints. No large remote archive was
created.

After independent verification, only this run's six redundant remote checkpoints
(478,351,248 logical bytes) were removed. The cleanup checked that the other
allocation's declared plan did not reference these files and left that allocation
and its files untouched. The resulting campaign footprint was 3,787,747,328 bytes.
`cleanup.py`, `cleanup-manifest.json` and `cleanup.json` retain the exact scope,
hashes and storage receipt. Full local checkpoints remain available.
