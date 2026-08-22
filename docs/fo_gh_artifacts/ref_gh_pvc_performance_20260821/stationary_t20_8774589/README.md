# Stationary Ref-GH trumpet ladder through t=20 (Aurora job 8774589)

All three equation-identical stationary trumpet cases completed from t=0 to
t=20 on eight distinct PVC tiles.  The 64/96/128 cases took 3908/5875/7842
cycles, reported field Linf errors `6.465162e-12`, `9.628409e-12`, and
`1.326683e-11`, and reported native-constraint Linf values `9.321782e-11`,
`2.032577e-10`, and `3.583889e-10`.  Every history row has bad-state count
zero.  The stationary initial-RHS Linf values remain between `1.060398e-16`
and `1.344374e-16`.

The field error grows almost exactly with the number of timesteps: the fitted
exponent is 1.028 versus cycles, and the final error per cycle stays within
`1.639e-15` to `1.692e-15`.  The native-constraint Linf exponent is 1.932;
this is consistent with one additional finite-difference factor acting on a
state-level error proportional to cycle count.  These are roundoff/secular
scalings, not truncation convergence of the analytically stationary solution.

The supplementary binary64 common-ADM constraint data use fixed physical masks
`r >= 0`, `0.125`, `0.25`, and `0.5`.  At `r >= 0.5`, the fitted t=20 orders
for H L1/L2/Linf are 3.969/3.934/3.731 and for |M| L1/L2/Linf are
3.985/3.962/3.748.  The values change by at most `5.89e-5` relatively between
t=0 and t=20.  The unmasked Linf norms do not converge because the puncture is
included; the masked result must remain clearly supplementary to the official
histories.

The numerical workload completed, but the PBS wrapper returned exit status 1
afterward because Aurora's default `python3` is version 3.6 and rejected
`from __future__ import annotations`.  The exact same analysis scripts passed
against the transferred outputs under Python 3.12.  The launcher now selects
Aurora's installed `python3.10`, and one Python-3.12-only multiline f-string
was rewritten without changing the analysis.  All three scripts now parse with
Aurora's actual Python 3.10, and regenerated tables are byte-identical; no
evolution rerun was needed.

The binary64 cbin files and restart chains are intentionally excluded from Git.
Their hashes are in `binary64_cbin_sha256.tsv`; the full data remain at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs/stationary_t20_8774589.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The executable SHA-256 is
`65bdc913f1d3494002a57ea619a132197344b93d1cb3e586bc2ead478b6f4e82`.
Source authority is commit `d7b9646f7b55a2d7b949dfd7ab413b0631b2531d`.
