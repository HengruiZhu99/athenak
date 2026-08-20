# Symmetric O4 Brill N256 long-run campaign

This directory binds the production input and Aurora head-build/capacity-segment
wrappers for the symmetric O4 Cartoon AMR path.  The source implementation and
focused tests are commit `3453b65a6b13c8f72cc1da6f05c565d245ce0f45`.

The evolution is the established direct-IrisK Brill case with `A=-0.047`, root
`128x256x1`, `32x32x1` MeshBlocks, O4 bulk/AMR transfer, RK4, CFL `0.15`, KO
`0.02`, `dchi_max=0.02`, and derefinement at `0.25*dchi_max`.  Investigation-only
recorders are disabled.  The production run uses one Aurora PVC tile in two-hour
capacity allocations; Athena's internal `-t 01:45:00` limit leaves time for a
final authenticated restart before PBS wall time.

The coefficient payload is intentionally reused byte-for-byte from
`../brill_collapse_contract_fixes_20260819/aurora_scratch_n256/` and must have
SHA-256 `ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b`.
