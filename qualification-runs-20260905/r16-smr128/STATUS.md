# Completed R16 SMR study

2026-09-05. See REPORT.md for the measured convergence, caveats and decision.

- Existing M/256 jobs are terminal: R16 failed at 5.187818M; P1 failed at 5.462666M.
- Requested R16 h=M/8,M/10,M/12 SMR runs all reached 20M cleanly.
- Domain +/-128M, seven SMR levels, 400 blocks; actual spacings verified at startup.
- Native chi>=0.0625 diagnostic constraint convergence measured through 20M.
- All 40 half-M field comparisons are present; asynchronous snapshots have an explicit interpolation-sensitivity audit. The 20M endpoints require no time interpolation.
- R16 remains a partial improvement: fine-core failure and unresolved asymptotic field convergence block the binary gate. No binary was launched.
- All associated evolution jobs have terminated. Do not restart failed screens or submit duplicate runs.

Remote root: /scratch/gpfs/FPRETORI/hz0693/pcgh-r16-smr128-20260905-1640

Array 13484719: elements 0 and 1 completed in Slurm. Element 2 was canceled while pending, then ran to completion on della-vis1 using head driver PID 1704699 (now exited). Completion times UTC: h8 17:00:42, h10 17:06:37, h12 17:17:10.

Final analysis: analysis/convergence.json and analysis/completion-audit.json.
Input/build hashes, commands, histories, boundedness data and final logs are preserved under runs/. Cartesian sample files remain locally and remotely; their hashes are in analysis/artifact-sha256.json. Full checkpoints/native outputs/stage CSVs remain remote.

Reproduce the analysis from the worktree root:

```sh
.venv-bbh-plots/bin/python analysis/pc_gh_regular_extension/analyze_r16_smr128.py \
  qualification-runs-20260905/r16-smr128/runs/h8/R16-smr128-h8-t20 \
  qualification-runs-20260905/r16-smr128/runs/h10/R16-smr128-h10-t20 \
  qualification-runs-20260905/r16-smr128/runs/h12/R16-smr128-h12-t20 \
  --output qualification-runs-20260905/r16-smr128/analysis
```

The initial mesh-only cleanup fault is documented in mesh-evidence/; actual evolution initialization succeeded. Earlier agent modifications elsewhere in the worktree were preserved.
