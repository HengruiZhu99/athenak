# Overlap preflight v1: plan check failure

Job8984990 ended F/exit1 and its allocation lock was released. CPU/SYCL parity
on12 tiles, short Minkowski, pulse half/direct/resumed evolutions all passed.
The restart comparison itself is bitwise exact over all7,763,888 doubles; both
files have SHA256 dab50e90e79d870eebc16e3dc6eb313c1f520620bc4d16c6aa65f449d0a19e9e.

One inherited plan gate incorrectly expected header.header_bytes=104, although
the new V9 schema correctly has120 bytes. The worker stopped before puncture,
so this campaign is not admitted. Preflight v2 changes that gate to120 and
reruns the complete campaign with unchanged source, binary, input and physical
tolerances. The independent verifier records this failed plan without changing
its historical outcome:32 artifacts checked and153 scientific gates replayed.
