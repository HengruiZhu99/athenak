# Phase 4 per-kernel device-image discriminator

Aurora job `8791775` rebuilt exact first-bad source `ab30fa96` with the sole
override `-fsycl-device-code-split=per_kernel`. It passed one full cycle on
twelve PVC tiles at history time `0.003404497M`. The executable SHA-256 was
`e4f7d1aefd16b135dbf96912f1e72760144c59f5cf1c0fa438fd3158c1d6974e`.

The default-layout `ab30fa96` executable failed. Both binaries expose 7119
unique `_ZTS` strings, but the generated SYCL offload bundle and total text
size differ. Per-kernel splitting did not reduce the reported Ref-GH private
spill pressure; one physical-geometry/gauge kernel instead rose from roughly
332 to 893 Reals. The pass therefore tracks device-image organization, not a
demonstrated private-memory threshold.

This classifies the defect as a SYCL/IGC/Level Zero device-image/code-layout
portability issue. It does not by itself establish an upstream compiler bug or
scientific evolution stability.
