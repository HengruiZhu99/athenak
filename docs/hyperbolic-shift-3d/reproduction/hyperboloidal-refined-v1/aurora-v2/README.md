# Aurora both-layer placement follow-up

Job 9037836 used one `debug` node, CompactBinaryMerger and 12 distinct GPU
tiles/ranks. All five commands completed; scheduler exit status is zero.
Mapping and profile/reference-Z4c gates precede dynamic AMR, then a repeated
legacy N16 control and the final both-layer N16 experiment. Both evolutions
reach t=0.05 (t/M=0.15), 200 steps at dt=0.00025. No trajectory was resumed.

`gauge-focused-v2.cfg` is the exact immutable runtime configuration, verified
against `plan2.json`. It places shell boundaries at .235,.32,.44,.56,.875,1.
This matches the resolved parameters of current `redistributed.cfg`; comment
text can differ. The plan also binds the successful local qualification and
completed first job. The baseline checkpoint had to match the first job's
SHA256 before the final experiment could start. That gate passed. Independent
collection also found every baseline scientific history column identical;
only wall-clock times differ. See `../aurora-comparison/baseline-repeat.json`.

The scientific binaries are the same frozen ddc5b4800 binaries as job 9037790.
`raw-evidence.tar.gz` preserves logs, reports, histories, unsmoothed profiles,
build/source provenance and the versioned launcher. `archive-receipt.json`
hashes every member; `verification.json` records independent local checks.
The separately included configuration is verified against the plan binding.
`job-state.json` hashes retained large checkpoints; `checkpoint-framing.json`
checks exact RF1 serialization and admission. No original files were deleted.

Final measured task footprint was 23,869,398,770 bytes including prior
performance outputs and this scratch campaign. With the additional 2 GB
atomic-checkpoint reserve it remains below the user's 50 GB allowance.
BN1 sources, launchers, binaries, checkpoint files, jobs and monitor were not
modified. Each allocation was submitted once after a live queue/account check.
The collected pre-submission records preserve those checks.

The final placement improves H relative to flat alone but worsens momentum
and some scri residuals. It does not demonstrate improved robustness or
convergence. Full comparisons and timing scope are in `../aurora-comparison`.
