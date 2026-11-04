# Matched constant-driver failure

Job9038285 failed at the same accepted step1237/time0.463875M as9038253.
Frozen controller verified source/binary/artifact bindings, rejected numerical
admission and released only the matching terminal lock. All151 artifacts are
locally hash/length verified. No restart or failed endpoint claim.

Only the inner driver policy changed. The four recorded exterior histories and
cumulative pole maxima nearly overlap. Nominal shell fields beyond r=.875 differ
by at most0.00090318 scaled, while substantial D differences occur farther in.
This is evidence against the inner policy being the controlling cause of this
particular early failure; it does not establish a universal root cause.

`compare_driver_failures.py` reproduces preservation verification, histories,
regional active-field differences and per-rank exclusive timing statistics.
The plot has been visually inspected. Failed surfaces were not captured by this
frozen binary, so the exact failed surface interpolation remains unavailable.

Measured four-node median exclusive stepping:238.38s constant versus238.48s
separated, for1238 attempted steps. Setup82.49s versus82.91s; diagnostics5.30s;
failure capture2.29s. These are nearly identical costs, not a controlled speedup
or strong-scaling result. Prioritize the measured stepping/setup costs over
output optimization; nested timers overlap and cannot be added.
