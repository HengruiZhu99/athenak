# Execution status: complete

Completed 2026-11-16. Scientific interpretation is in [report.md](report.md).
Scheme A remains unvalidated for accurate native zero-news evolution despite
smaller full-interval errors with refinement.

- N12 pair 9093867, N16 pair/reference probe 9093938 and N20 pair 9094022:
  scheduler F, exit 0; collected products verified against package manifests.
- Half-step evolution 9094227: 3408 cycles, evolution exit 0; initial analysis
  failed on an import collision, leaving original scheduler exit 143 intact.
- Analysis-only recovery 9094386: scheduler F, exit 0. Timestep, frame and
  point-matched reference comparisons completed. The original evolution
  checkpoint is bitwise unchanged by recovery (matching SHA-256 before/after).
- Recovery submission 9094380: cancelled while queued after preflight exposed
  an unnecessary plotting dependency; no allocation started.
- Eight final checkpoints verified: hashes, endpoints and 337,598,976 finite
  stored scalar values. All 15 focused CTests pass after the final C++ rebuild.
- Total follow-up allocation: 4.48389 node-hours, at most two nodes concurrently.
  Original failure and recovery costs are both included.

No current-task jobs remain running or queued. Large raw results and immutable
source snapshots remain at the locations in [reproduce.md](reproduce.md).
Unrelated jobs and untracked directories are preserved.
