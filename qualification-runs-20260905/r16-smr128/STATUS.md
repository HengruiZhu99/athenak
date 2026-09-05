# R16 large-domain SMR convergence follow-up

2026-09-05, 17:01 UTC. Active user goal: monitor both fine-core jobs to termination;
measure R16 SMR convergence through 20M at h=M/8,M/10,M/12 with boundaries +/-128M
and chi-excised constraints; attempt binary evolution only if the gates pass.

## Fine-core outcomes

Both existing M/256 jobs have terminated with numerical failure, confirmed in
raw segment logs and stage-results.json, not inferred from scheduler disappearance.

- R16: strict conformal-metric positivity failure at t=5.187818M, post-RK update,
  det=-2.584038, position (-.005859375,-.04101562,-.06054688)M. Exit 1 at
  2026-09-05T16:35:19Z. Final segment 0005.
- P1: strict rho failure at t=5.462666M, rho=-.09892196, post-RK update,
  block/cell (m,k,j,i)=(50,4,4,11). Exit 1 at 2026-09-05T16:42:19Z.
  Final segment 0007.

Compact raw evidence is in refinement-evidence/. All checkpoints and larger
diagnostic files remain in the original remote campaign. Neither run was restarted
after failure. No binary is authorized by a passed science gate at this point.
The requested coarser study proceeds as an exterior-convergence measurement;
it cannot erase the failed fine-core stability gate.

## New runs

Remote root on della-vis1:
`/scratch/gpfs/FPRETORI/hz0693/pcgh-r16-smr128-20260905-1640`.

Submitted array **13484719**, elements 0/1/2 correspond to h=M/8,M/10,M/12.
At 16:56 UTC elements 0/1 are running on della-l02g11/della-l01g15 at
12.70M/7.4125M. Element 2 was verified PENDING, canceled before allocation, and
then launched on the verified-idle della-vis1 A100 at 16:53 UTC. Its physical
time is 2.375M. Head driver PID is 1704699; exact command was
`nohup env SLURM_ARRAY_TASK_ID=2 bash run.slurm > head-h12-driver.log 2>&1 < /dev/null &`
from the remote campaign root. Driver PID is also saved in head-h12-driver.pid.
The input and binary are identical to the queued experiment, which never ran.

At 17:00:42 UTC, h=M/8 completed 20M cleanly in segment 0001, cycle 1601,
confirmed by stage-results.json, completed.json and the terminal log. A tiny final
step follows cycle 1600 and is preserved. The final full-state callback reports
min conformal-metric eigenvalue .8252974. Elements h10 (Slurm 13484719_1) and
h12 (head PID 1704699) were verified live at 17:01 UTC. No restarts or replacements
were triggered by observation timeouts.
Slurm assigned partition gpu and QoS gpu-short despite the script requesting
gpu-test. Each element has one A100, 2 CPUs, 32GB RAM, and a 2-hour wall limit.
An attempted pending-only relocation of element 0 was prevented by its running
state; it was neither canceled nor duplicated. Elements 0/1 remain Slurm jobs.

Runs live at `runs/hN/R16-smr128-hN-t20`, N=8,10,12. The existing driver preserves
15-minute wall-segment checkpoints and refuses implicit reruns. Inspect each
run's actual fatal/completed marker; the stage wrapper can return zero even when
an individual evolution exits nonzero. Resume only verified allocation interruption,
with the same binary/input hashes, never a strict numerical failure.

The executable is the original hybrid monitor CUDA binary:
`/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/monitor-source/build-hybrid-monitor-cuda/src/athena`
SHA256 `96bc34157896c164904a6ff426e84f4c97d95a7b8c8c1a43971940fc744e60b1`.
The batch script verifies this hash. No evolution code was changed or rebuilt.

## Geometry and measurement

Inputs and source hashes are in inputs/manifest.json; generator is
analysis/pc_gh_regular_extension/make_r16_smr128.py. FD6/RK3, CFL .2, KO .3,
dt ceiling .0125M, R16 core/taper .125/.5M, reduction/gauge projection off.

Seven nested static refinement levels give the same physical block tree at all
resolutions: 400 blocks, 64 finest blocks covering [-2,2]^3, with 8^3,10^3,12^3
cells per block. Finest spacings are .125,.1,1/12 respectively. Actual exported
trees were parsed to verify these numbers; see mesh-audit.json. The old binary's
mesh-only (-m) path writes the complete tree then faults in MeshBlockPack's
destructor (exit 135). This setup-only fault is retained in mesh-evidence/;
evolution startup independently confirms all three requested spacings and 400
blocks on CUDA; all three report zero guarded initial-data cells. The mesh-only
cleanup fault did not recur on their evolution startup paths.

Primary constraint histories select chi=w^2 >= .0625, diagnostic excision only.
They retain full-domain and fixed-radius norms separately. All strict state and
stage/transfer monitors remain enabled and unexcised. Native history constraints
use coordinate-volume weighting; shared ADM columns use physical-volume weighting.
An outer boundary at 128M reduces boundary influence on the central 20M study;
the domain size alone is not a proof that every gauge characteristic is excluded.

Analysis prepared in analysis/pc_gh_regular_extension/analyze_r16_smr128.py:
native 3D chi-excised GH/H/alpha-weighted-M/reduction/curl/algebraic norms,
L2 and RMS, both unequal-spacing pair orders through time, plus all-55-field
self-convergence on common Cartesian slices using the intersection of the three
chi masks. Missing common output times are reported rather than substituted.
Negative/unresolved orders remain visible. This does not replace native puncture
power or interface-position qualification. The unequal-spacing Richardson solver
was checked against known orders -2,2,4,6.

Local analysis Python: `.venv-bbh-plots/bin/python` (system python lacks matplotlib).
The analysis ran successfully on an early common interval through 3M. Its partial
plot was visually inspected. Early constraint norms decline with refinement, but
some field difference alignments are negative; early norm ratios are not a pass.
Partial outputs in analysis-partial/ are explicitly incomplete and must be
replaced by the final common-interval measurement after all runs terminate.
After copying completed run inputs, logs, histories and cart output locally, use:

```
.venv-bbh-plots/bin/python analysis/pc_gh_regular_extension/analyze_r16_smr128.py \
  qualification-runs-20260905/r16-smr128/runs/h8/R16-smr128-h8-t20 \
  qualification-runs-20260905/r16-smr128/runs/h10/R16-smr128-h10-t20 \
  qualification-runs-20260905/r16-smr128/runs/h12/R16-smr128-h12-t20 \
  --output qualification-runs-20260905/r16-smr128/analysis
```

Goal remains active. Await all three terminal results, verify actual mesh/spacing,
inspect strict failures and unmasked health, measure convergence over the common
completed interval, and explicitly report missing coverage if any run stops before
20M. Do not declare the binary gate passed from coarse survival or excised norms.
