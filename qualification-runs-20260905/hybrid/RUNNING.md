# Active campaign continuation

As of 2026-09-05 15:16 UTC. Goal is active; do not report completion or a qualified
hybrid. Worktree is `/Users/hz0693/research/athenak-pcgh-localization-20260902`,
branch `codex/pc-gh-gamma2-20260904`. Remote root:
`/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317` (R below).

SSH uses `-o BatchMode=yes -o ControlPath=$HOME/.ssh/codex-control/%C`
to `hz0693@della-vis1.princeton.edu`. Load `/home/hz0693/athenak_env` before
commands; only enable shell nounset afterward. Evolution is CUDA-only.

## Running/queued

- 13481105 array, core-controls C1/C4/C16, throttle 1. C1 failed 4.649832;
  C4 running (~3.77 M); C16 pending. Uses R/source/build-hybrid-cuda (legacy
  operation-8 diagnostic labels, unchanged correct dynamics).
- 13481378 uniform hybrid array: R4 failed 9.5875; R16 running (~7.85 M).
  P1 element 2 was cancelled while pending and run on head A100 instead.
  P1 completed 12 M cleanly, 961 complete steps including a tiny final step.
- 13481379 refined hybrid array: R4 running (~3.30 M), R16 pending. Throttle
  raised to 2, within the global gpu-test cap of three simultaneous jobs.
  P1 element 2 cancelled while pending, now running on the head A100.
- R4 was submitted for 1 hour; Slurm denied increasing its running time limit.
  Pending R16 was successfully increased to 2 hours. Conditional resume job
  **13482142**, dependency afterany:13481379_0, uses resume_hybrid_screen.py:
  resume allocation interruption from checkpoint, never resume a strict failure.
- Head A100: P1 refined-core screen (~0.34 M), output R/core-hybrids/P1/core256-P1.
  exec_command session **89553** wraps its SSH driver. GPU near fully utilized;
  avoid another concurrent GPU evolution on that device.

Hybrid runs use R/monitor-source/build-hybrid-monitor-cuda (MPI-capable CUDA,
source commit 1e6b0612; later changes there are analysis scripts only). Authoritative
inputs are R/inputs-final, committed locally. All raw evidence stays remote.

## Next actions

1. Continue checking completed.json, stage-results.json and all segment logs.
   A zero stage-runner exit is not a science pass; inspect each run exit/fatal.
2. Let all six prescribed core screens finish or fail. Full monitor overhead
   makes these much longer than baseline screens. Wall-segment restarts are normal.
3. Assess intersection of uniform/refined survivors. Only surviving candidates
   enter the prepared resolution/mask/FD2/interface/half-step qualification;
   use make_hybrid_qualification.py for explicit candidate inputs. No binary yet.
4. For completed runs, summarize_hybrid_history.py preserves stage envelopes
   and cumulative map corrections. analyze_transfer_brackets.py records changes
   of maximum norms (not vector corrections); RK brackets include stale ghosts.
5. Update REPORT.md, plots and exact evidence; commit focused changes. Do not
   add unrelated untracked files. The implementation is committed through a694e4cc.

## Completed evidence beyond REPORT.md

Both original failures reproduced exactly: uniform C1 8.379261 M, fine C16
4.891496 M. All 26 serial and 3 MPI oracles, six flat controls and all six
shifted-wave convergence ladders pass. Fixed pulses all four families complete
for all six arms. P1 has much slower curl convergence (0.410,1.114) than finite
relaxation; its actual jump schedule was independently recovered.

P1/R16 moving pulse restart differences <=3.495e-23 in all121 components.
New MPI one/two-rank P1 pulse results are identical. New health diagnostics leave
fields identical to old monitor, trackers differ <=1.23e-30. Empty region handling
and exact Minkowski minima verified. Shifted moving-wave trackers converge.

Frozen source samples at R/frozen-sources: C1-uniform-t8 has sampled min energy
margin -.797 and max spectral abscissa +.477; C16-core-t4 has +15.575/-15.138,
despite later failure. These samples are not global/late-time bounds.

Transfer summaries for original fine C16, controlled fine C1 and uniform R4 are
complete. Late fine Q/curl-Q maxima grow especially across algebraic enforcement;
this is amplification near breakdown, not proof of the initial cause. Need
regional onset analysis from stage envelopes. P1 uniform history/transfer summary
also complete. On the uniform grid its immediate map corrections are zero
outside the taper; correction convergence remains untested.

Known diagnostic schema correction: old op8 collides with physical boundaries;
actual correction rows recover only inside op3 before/after bracket. New op100
is unique. Parser tests prevent duplicate and intermediate-stage events. New
health min_* values are minima in the legacy max column, block=-1 means empty.

The root CONTEXT.md records the completed skill interview. Requested pinned
skills were installed and used; no more questions or permissions are needed.
