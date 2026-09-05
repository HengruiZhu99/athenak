# Direct lapse-gradient campaign continuation

Goal remains active. Worktree /Users/hz0693/research/athenak-pcgh-localization-20260902,
branch codex/pc-gh-gamma2-20260904. Do not touch preexisting dirty analysis/docs.
Remote root R=/scratch/gpfs/FPRETORI/hz0693/pcgh-direct-lapse-20260905.
SSH: hz0693@della-vis1.princeton.edu, BatchMode=yes,
ControlPath=/Users/hz0693/.ssh/codex-control/%C. Source /home/hz0693/athenak_env
before nounset. No unrelated jobs touched. No Slurm allocation needed: head A100
idle verified immediately before launch; saved partition/QoS inspection retained.

Gate 1 PASS in gate1.json: all 12 CUDA FD2/4/6 uniform curl cases, 39 independent
production projection oracles and 3 diagnostic checks; full MPI-capable CUDA build.
CPU curl matrix also passes. One pre-execution Git-status/Kokkos-symlink setup
failure was preserved, then fixed by copying identical Kokkos files. No dynamics
changed after build. Binary SHA256 0bfd4c5ce9c9e55285614fb988ebd90624a12640f21a41eb60d674148e4e1788.

Stress is running via nohup head driver PID 2361206, R/run-stress.sh.
R/stress/core256-R16 holds raw evidence. R/stress-driver.log and eventual
R/stress-exit.json record driver status. Runner uses 15-minute clean checkpoint
segments and same-binary automatic resumes, never resumes a strict failure.
Input unchanged from baseline, SHA256 ff69d9d88a75e4156735454b19b7cd91522d133ce4ff0f8fd9ced1e2d7105f80.
Target 6M; no downstream runs have started.

Baseline compact evidence exists locally and at R/evidence/baseline. Raw baseline
is /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317/core-hybrids/R16/core256-R16.
Its strict metric failure is t=5.187818M. Last native hst=5.17507M. Last healthy
completed-step minimum eigenvalue fell to .0616958; peak curl_Q=3.45415e6.
Raw 4.155GB hybrid monitor SHA256 69fd76396a1550b65f3f78f23c6f44e0e3325cc259a723e8b7e6a29c2b428486.

Next: monitor stress through terminal status. Once terminal, run
source/analysis/pc_gh_regular_extension/reduce_direct_lapse_monitor.py on its
*.hybrid.csv into R/evidence/corrected, copy histories/logs/input/provenance and
completed.json if present, then rsync compact evidence locally. Run
.venv-bbh-plots/bin/python analysis/pc_gh_regular_extension/analyze_direct_lapse_stress.py
 qualification-runs-20260905/direct-lapse-gradient/evidence/baseline
 qualification-runs-20260905/direct-lapse-gradient/evidence/corrected
 --output qualification-runs-20260905/direct-lapse-gradient/analysis
The plotting venv now has pandas 2.3.3 as well as numpy/matplotlib.

The new reducer preserves rollback-corrected stage-3 operation -2 samples and
half-M operation/stage extrema. It requires a finished immutable monitor.
Independent rollback test is test_direct_lapse_monitor.py. Separate old Ralpha
and new RL_direct diagnostics coexist; 55 evolved fields unchanged, one diagnostic
field appended. New native RL is in pcgh_con and regional monitor; old hst columns
are deliberately preserved.

Inspect all/chi constraints, separate curls, completed-step and regional growth,
locations and minimum eigenvalues. PLAN.md freezes criteria. A strict failure is
FAIL even if lifetime improves; unresolved growth is INCONCLUSIVE. Stop downstream
on either. Only pass proceeds to original M/8,M/10,M/12 SMR +/-128 through20M and
then exact documented binary criteria (see PLAN). No extra tuning/sweeps.

Finish with inspected plots, source/binary/input provenance, explicit gate decisions,
focused commits and report. Mark goal complete only after gated report finished;
a documented scientific failure is a completed campaign. Avoid bulk staging of
preexisting outputs or unrelated user edits.
