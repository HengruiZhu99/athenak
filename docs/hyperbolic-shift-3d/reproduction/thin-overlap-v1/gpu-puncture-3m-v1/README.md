# Bounded thin-overlap puncture targeting 3M (externally interrupted)

Job **8988310**, CompactBinaryMerger/debug-scaling, one node and twelve ranks/tiles.
Plan SHA256 `ad599d4e6718fb0561fb1d1f68c83a5842c14b66cc947dc5478d1ea14916aa23`.
The unchanged frozen c530 sparse executable evolves the original n16 L1 initial
data to coordinate time 1 = 3M, using dt=1/2048 and diagnostic interval 1/32.
The existing scoped preflight admission binds source, binary and exact interface
parameters. The queue-isolated launcher comes from a551fc12c.

The lapse floor, log variables, scri closure and original horizon/exterior norm
criteria are unchanged. Online profiles are disabled; the final checkpoint can
be sampled independently against the preserved same-time 1D profile. Failed
horizon samples remain unavailable. This is a fresh stability experiment;
submission is not evidence of completion, accuracy or convergence.

Prospective storage guards reserve 180MB plus 100MB for atomic replacement.
Verified removal of locally retained observer checkpoints/profiles reduced the
shared campaign footprint to 3,590,409,728 bytes before submission. Other task
files and the debug queue lane were not modified by this cleanup.

## Terminal outcome

PBS reports terminal state F, exit 143, and termination by the user account.
The native log records rank termination by signal 15. This task did not issue
the termination. The worker was interrupted; no final report or checkpoint
was produced. This is neither a completed stability test nor evidence of
a numerical failure. No automatic replacement job was submitted.

The retained history has 11 samples through coordinate time 0.3125 = 0.9375M
(640 steps), with accepted horizons at every recorded sample. The last
sample has exterior H norm 0.0303603 and exterior M norm 0.0535711.
The maximum recorded null error is 1.5544e-15; sampled lapse stays positive.
A final floor-count audit is unavailable without the native final report.
These partial diagnostics do not establish stability through 3M or convergence.
The raw small evidence and a SHA256 manifest are retained alongside this file.

The current implementation remains cell centered, following the user's updated
instruction. Vertex-centered tests are deferred.
