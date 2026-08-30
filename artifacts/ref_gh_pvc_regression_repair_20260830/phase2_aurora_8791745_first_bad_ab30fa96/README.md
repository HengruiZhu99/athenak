# Phase 2 first-bad ab30fa96

Aurora job `8791745` tested commit
`ab30fa963f5d1d7ce54748ffb287c91c87705153`, the production
fully-subtracted gauge dispatch, on twelve PVC tiles using the frozen Phase-1
configuration. It reproduced the Level Zero PDE-level `NotPresent` write
fault after all ranks reported the evolved-stage physical metric and gauge
boundary fences. PBS exit status was 143 and no positive-time history was
written.

Its executable SHA-256 was
`0c64e3cbb784590581af847a55ad993943f93c81bfaf6f6289bfc7dec7e1f475`.
The prior production-changing commit `b290c2bd` passed, so `ab30fa96` is the
first-bad source commit. The visible post-boundary fault remains a fence
localization boundary, not proof that the boundary kernel corrupted memory.

`first_bad_source.diff` preserves the complete source diff against the exact
parent `8e028caf`. Full build/run logs are retained in deterministic gzip form.
