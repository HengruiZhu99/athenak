# N12 performance validation

The selected cache/sum-reuse implementation changes only the agreement module.
The full-overlap mode takes 1.184239 times the frozen legacy wall time in the
matched two-node, 24-rank 12–13.5 M0 windows of job 9091365. Opposite-order
pairs give 1.174450 and 1.194147. Timesteps and output cycles match; startup
and output are included. These short windows do not establish the total
cost of a fresh 0–30 M0 run or reliable margin across allocations.

The 26–30 M0 optimized continuation in job 9091104 reproduces the entire
original final checkpoint byte-for-byte and all 16 constraint-history rows.
The source checksum is c2b5030b8d60d0bf4329b637a159f5df19594d37aac4835af258c1256688c4a2;
the selected Aurora executable checksum is
 de633cda33dd17fc312214a0d7e2253512bf4ea0f2ba19911b71fe2fb94a36e3.

The optimized executable's second legacy-mode repeat in job 9091219
failed strict reproduction (maximum normalized field error 4.7892e-7).
Its first legacy and both full-overlap repeats passed. That job's raw
16.33% overhead is excluded. The accepted timing comparison uses the
original frozen legacy executable; it does not resolve this regression.
Do not treat this push as general production qualification of legacy mode.
The N12/N16 science campaign used the original frozen executable throughout.

`manifest.json` hashes the included scripts and receipts. `control_pair.py`
and `control_pair.pbs` record the exact Aurora benchmark procedure and use
the archived paths under `/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111`.
They require those external checkpoints and qualified binaries; large binary
artifacts are not in Git. `final-archive.json` identifies the full receipt archive.
`receipts-final/control-pair-9091365/results.json` contains timings and checks;
per-window logs preserve the measurements. The failed repeat is included
under `receipts-final/group-pair-9091219` for traceability.

The original feature baseline was captured on 2026-11-11 at HEAD
8d9a34ec37bc05461e7a31f88722fbcbc3d16560. The required pre-existing binary/scri
source and tests are committed separately from the overlap implementation
and this optimization. Historical unrelated experiment artifacts remain outside
the commits.
