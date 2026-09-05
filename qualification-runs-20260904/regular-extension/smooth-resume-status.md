# Smooth finite relaxation: 2026-09-05 work checkpoint

Classification: partial dynamical improvement; smooth fallback implemented but
not numerically qualified. No new binary has reached merger or 100M.

The latest constant-rate candidate is FD2/RK4, lambda=2, kappa=1, eta=2, KO=.3,
mass=.5. All three large-domain SMR resolutions completed 10M. The analogous
lambda=1 coarse control failed at 9.838348M. The lambda=2 inner exterior K ladder
still has a negative fitted Richardson order (-.551), although successive raw
differences decline; it is not a clean all-field convergence pass. The outer
aggregate order is 1.928. Twelve cached remote uniform pulse results have maximum
fitted rate error 4.71e-11 and finest adjacent spatial order 1.797. Three gauge-wave
resolutions passed the existing checker.

Last live Della observations before loss of connection:

- Long coarse lambda=2 single: segment 2, t=13.42331M, finite at the last printed
  cycle. Final state and whether the process survived disconnection are unknown.
- Uniform r20 completed t=3M; r24 had started on the head-node sequence.
- Slurm 13468851: core m/256, tlim=3M, running at about 40 minutes elapsed; latest
  sampled progress t=2.686576M shortly before that. Completion is unverified.
- Slurm 13468852: AMR lambda=2 controls running, with B32/48/64 and L32/48 complete
  among the locally copied metrics. These jobs are independent of the SSH session.
- Slurm 13468414 no longer appeared in squeue. Inspect its accounting and
  collection log before declaring all KO=.3 interface controls complete.

The SSH master disappeared during work; the control directory became empty and
the hostname della-vis1.princeton.edu stopped resolving. No remote cancellation
was requested. The half-mass rsync exited 255 mid-transfer, so existing local
copies may be stale or incomplete. In particular the pulse summary script
correctly refuses missing raw dumps; the separately named cached summary does
not claim a fresh independent raw-data verification.

The new source lives locally on the current branch. The remote copy is
`/scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b/source-smooth`.
It was created separately from the old running source and binaries. Source and
the C++ pulse/curl dump header were copied successfully before disconnection.
The CUDA oracle build reached at least 36%; its completion and the subsequent
binary build are unverified. Later Python verifier and input uploads failed DNS
resolution and must be copied again. The new local serial build completed as a
compilation check only; no evolution was run locally.

Resume in this order:

1. Reestablish SSH and inspect Slurm accounting, running processes, all collection
   logs, checkpoints, and completion markers. Fetch missing evidence without
   deleting previous runs. Check head-node processes before resuming a checkpoint.
2. Complete constant-rate lambda=2 core/uniform/AMR/long analyses. Survival alone
   does not release the binary gate. If a new constant-rate binary is justified,
   use a new lambda=2 run directory and the unchanged old CUDA binary.
3. Copy the current analysis scripts and smooth input collection to source-smooth
   and ../inputs/smooth-controls. Inspect build processes before continuing an
   interrupted build. Preserve its original build log; use a new continuation log
   and record new binary/configuration hashes when compilation finishes. Do not
   rebuild either old executable while its runs are active.
4. Run the prepared smooth control Slurm scripts only after the CUDA build is
   complete. Check the twelve affine matrices independently in core, taper, and
   exterior. Compare the constant-profile regression's original u/E/norm columns
   and initial/final times against the old executable (the new C columns are extra).
   Apply the unchanged gauge-wave checker, analyze fixed/moving pulse and curl
   convergence, and compare moving-wave trackers to the exact implicit trajectory.
5. Exercise a moving-mask restart at a completed intermediate checkpoint using
   a distinct output directory and --restart-from. Compare all active fields and
   endpoint centers with uninterrupted evolution. Test invalid profile settings.
6. Only then dispatch the prepared smooth single-puncture inputs, add three
   uniform resolutions and m/256 core controls, and vary the physical mask widths.
   Follow with the moving-mask binary only if those gates improve. Preserve all
   failures and assess full-volume/core fields, exterior convergence, constraints,
   tracks, symmetry, and waveforms through merger and 100M if stable.

No clipping, floors, stronger KO, hard projection, or reduced diagnostics are
part of the new option. All four reduction families use the same bounded rate.
The derivation includes the homogeneous taper curl source and the lack of a
uniform puncture-point theorem. The exact off-surface covariant residual and
smooth finite-relaxation symbolic checks passed; CUDA evolution checks remain
pending.
