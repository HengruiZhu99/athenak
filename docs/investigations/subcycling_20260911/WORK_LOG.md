# Execution log

User approved implementation and Perlmutter shared_interactive tests on a separate branch/worktree. The review PLAN.md is the original proposal; its pre-approval status paragraph is historical.

- Existing local worktree vc-corner-fix was clean at c930074d, matching the latest relevant origin branch after fetch. SSH push reported everything up to date, so no empty commit was manufactured.
- Local new branch: codex/vc-cartoon-subcycling-20260911; worktree: /Users/hz0693/research/collapse/vc-subcycling.
- Remote separate bare repository and worktree: /pscratch/sd/h/hzhu/vc-subcycling-20260911/{repository.git,source}. Uses the existing qualified Kokkos dependency via symlink.
- Added default-off time/execution_profile. Timings fence Kokkos operations; hierarchy distinguishes inclusive from exclusive work and initialization from evolution. Fences can perturb execution overlap; paired uninstrumented measurements are required.
- Paired production/profile restarts prepared at t62.03460401194272 (3434 blocks, cycle118310) and t62.20556032298022 (8924 blocks, cycle133489). Twelve live-AMR cycles each, original physics and output cadence, separate output/AMR files. Full checkpoint payload and history comparisons are required.
- Source inspection found a quadratic contributor scan in shared-node topology construction. It is a candidate performance bottleneck; await measured evidence before changing direction.
- Initial remote configure failed because git's empty submodule directory received a nested symlink. Replaced only that newly created empty directory with the intended dependency symlink; restarted configure/build.

No asynchronous level evolution or production campaign changes have been made at this point. Gate results will be recorded here.
