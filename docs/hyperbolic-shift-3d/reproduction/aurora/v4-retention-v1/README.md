# Verified V4 payload retention

Nine large files from the completed V4 preflight and failed V4 duration run
were archived locally and independently checked against their original worker
hashes and byte counts before remote removal. All nine were rechecked on Aurora
before any unlink. Both allocations had terminal-release records, and no
campaign allocation was held during cleanup.

The [manifest](manifest.json) retains every exact path, logical byte count,
SHA256 and local archive path. The [receipt](receipt.json) records each removal
and the measured footprint. This reclaimed 334,582,448 logical bytes; the final
campaign footprint was 1,688,928,256 bytes under the conservative storage
counter, below the 5,000,000,000-byte limit. Logical file size and filesystem
allocation accounting are distinct.

All source exports, executables, small reports, histories, bindings, admissions
and failure analyses remain. The three new early-window V4 checkpoints were
preserved. The preflight checkpoints supplied implementation checks only; the
failed duration checkpoint was explicitly unadmitted. None was needed for a
continuing scientific evolution. Reproduction that needs these payloads must
use the preserved local archives rather than the purged remote paths.

The one-time [purge script](purge_verified.py) rejects an existing receipt and
unexpected paths. It is retained as an action record, not a command to rerun.
