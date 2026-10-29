# V11 GPU preflight v1 — stopped by stale metadata checks

Job 8986398 terminated with failed preflight status. CPU/GPU arithmetic,
Minkowski, pulse, and bitwise restart checks passed. Two copied plan expectations
incorrectly required V10 magic and a 136-byte header; output correctly contained
V11 magic and a 152-byte header. The worker therefore stopped before the puncture
case. This run is not an admission. Its original plan and records are preserved.

Fresh preflight-continuous-positive-v2 changes only those two expected metadata
values and the campaign paths. Scientific source and binaries remain exactly
source-v19 (faee66ea3). No numerical tolerance is relaxed.
