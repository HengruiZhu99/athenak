# N256 lapse bisection recovery — supervision complete

The corrected boundary configuration and device synchronization are deployed. Both requested fresh midpoint evolutions completed cleanly, their checkpoint/history/classification checks passed, and the controller automatically submitted the next midpoint. The existing hourly monitor is now ACTIVE.

## Classification and settings

Every completed timestep checks the global minimum lapse. The first event wins: lapse <1e-5 gives early collapse; a previous dip <0.1 followed by recovery >0.8 gives early dispersal. Otherwise the run reaches coordinate t=200 and final lapse <0.01 gives collapse, otherwise dispersal. Previous minimum lapse persists in restart parameters. Failures and incomplete runs do not update the bracket. Relative amplitude tolerance is 1e-8.

N256, 32x32 meshblocks, live AMR (not replay), CFL 0.15, dissipation 0.50, telegraph lapse, prescribed zero shift and zero constraint damping are retained. Boundary configuration is full_constraint_bjorhus with extrap_order=2. Device shared-node synchronization is enabled; runs use shared_interactive.

## Verified run

| Case | A | Job | Result | Final t | Global minimum lapse |
|---|---:|---:|---|---:|---:|
| cycle_01 | -0.04925 | 58140294 | Early collapse; COMPLETED 0:0 | 79.3727275712953 | 9.999072185270558e-6 |
| cycle_02 | -0.048875 | 58141886 | Early dispersal; COMPLETED 0:0 | 76.93279283067247 | 0.8000006888462381 |

Initial imported endpoints are A=-0.0485 (authenticated recovery certificate) and A=-0.05 (authenticated clean early-collapse history). Neither counts as a fresh midpoint. Current verified bracket: supercritical -0.04925, subcritical -0.048875.

The first midpoint final checkpoint is `/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910/cycle_01/rst/lapse200.00080.rst`, SHA256 `0f5f6a5fab976f60e1a79e5f1f938c26138d916982129232516e24058349fb17`. Native stopping marker, final history and checkpoint time/cycle agree. The controller automatically submitted cycle_02.

## Qualification and limitations

CPBC plus linear ghost extrapolation passes the original corner-instability window and evolution to t=200; either modification alone failed the longer nonlinear check. A fresh A=-0.047 run reached t=200 with finite fields, minimum lapse 0.992179 and maximum absolute Kretschmann 2.23472e-4. This is empirical qualification, not a general stability proof.

The device synchronization comparison reproduced all 4422 history samples across 72 columns, AMR records and checkpoint payload exactly through t=50, with 2.268x speedup. Native stop-rule GPU validation passed, including exact above-threshold evolution parity and recovery-state persistence across restart; 33 workflow regression tests passed.

Large interior constraint errors remain in the near-critical evolutions. In cycle_01 the final curvature maximum is 273200.3125 on the axis at z=1.1875, while corner curvature is 2.08e-9. These results define an operational lapse bracket, not a spatially converged physical threshold or validated MOTS.

## Provenance and outputs

Branch: `codex/vc-cartoon-outer-corner-fix-20260910`; executable source SHA `400f784f95e055ca3457953a463b00f03aa71a8b`. Source and workflow changes are committed and pushed; later commits record evidence. Executable SHA256: `b875502b6150cdf3f3a657bfea63b1caf2da2fb0702c94d9a9821c20ade60054`.

Remote campaign: `/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910`. Local plots/data: [PNG](/Users/hz0693/research/collapse/lapse-bisection-recovery-20260910/lapse_all_runs.png), [PDF](/Users/hz0693/research/collapse/lapse-bisection-recovery-20260910/lapse_all_runs.pdf), [CSV](/Users/hz0693/research/collapse/lapse-bisection-recovery-20260910/lapse_all_runs.csv). The A=-0.05 imported curve extends beyond its first qualifying crossing because it predates automatic stopping.

## Second completion and autonomous continuation

Cycle_02 previously reached a stored minimum lapse of 0.02926637, then stopped natively on recovery above 0.8. Its final checkpoint is `/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910/cycle_02/rst/lapse200.00077.rst`, SHA256 `b62f367a1562dba630ace699fb890956c8e1081b98a4a01ec817d06cb025cf9f`. Final fields are finite: maximum absolute Kretschmann 28.23665 at rho=0.0625,z=1; corner maximum 1.72283e-9. The observed boundary runaway has not recurred in these two midpoint runs.

The strict audit is saved in [SUPERVISION_COMPLETE.json](evidence/recovery-supervision-complete.json). Successor A=-0.0490625, job 58143650, was automatically submitted to shared_interactive and is queued for resources. The campaign continues toward relative bracket width <=1e-8; this precision has not yet been reached.

Automation `monitor-n256-brill-lapse-bisection` is ACTIVE every 60 minutes in this task. It verifies new classifications/checkpoints and successors, refreshes plots, reports meaningful changes, and stops on evolution failures. The supervision milestone is complete; the autonomous bisection remains in progress.
