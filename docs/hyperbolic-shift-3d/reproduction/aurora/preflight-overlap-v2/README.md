# V9 overlap GPU preflight passed

Job8985223 finished F/exit0, passed all7 cases and was released. Independent
verification checks194 scientific gates,39 artifacts,2642 frozen source blobs,
12 ranks on12 distinct GPU tiles, all4 checkpoint arrays and exact direct/restart
byte equality. This admits only implementation/restart correctness, not duration
stability. Source a6dfbd4474e653b7a15a8df39d3c79f5e66e0a13, frozen source-v17.

The plan differs from the prior failed preflight only by fresh output paths and
the corrected restart header-length check (120 bytes). Physical gates, binaries
and initial data are unchanged. The initial n16 Minkowski maximum CFL at
dt=.00025 is about.27153; native evolution rejects CFL above.4, so subsequent
duration tests use dt=.0003. No physical acceptance tolerance has been relaxed.
