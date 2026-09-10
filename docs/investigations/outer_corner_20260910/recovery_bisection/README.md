# N256 lapse-recovery bisection

Prepared; not launched yet. Supersedes the stopped lapse-bisection-t200-corner-fixed-20260910 campaign after the user's policy revision.

Rules, evaluated every completed timestep using GLOBAL minimum lapse:
- Below1e-5: early collapse.
- After dipping below0.1, recovery above0.8: early dispersal.
- The first early event terminates. Otherwise continue to coordinate200; final lapse below0.01 means collapse, otherwise disperse.
- Strict inequalities. Failed/nonfinite/incomplete evolution is not classification evidence.
- Relative amplitude width |A_sub-A_super|/|A_super| <=1e-8.

Native monitor persists minimum lapse seen in checkpoint parameters. Imported older checkpoints require authenticated historical evidence for any seeded prior minimum. The current A=-0.0485 evolution was intentionally cancelled after verifying a qualifying dip/recovery. Its cancellation remains recorded; a separate short clean restart certificate is required before boundary reuse.

Planned initial bracket: -0.0485 dispersal and -0.05 collapse, after validated endpoint imports. Two MORE actual midpoint iterations (-0.04925 first) and their automatic successors must be verified under this workflow. Then activate hourly monitoring and complete the supervision goal. Do not count reused endpoints or short verification restarts as midpoint iterations.

N256 physics, live AMR record policy, full_constraint_bjorhus + extrap_order2, and device shared-node synchronization remain unchanged. Only lapse stopping/classification and requested amplitude precision change. Runs use shared_interactive.

33 Python tests currently pass, including strict recovery thresholds, event ordering, malformed marker rejection, and simulated successor sequencing to1e-8. CUDA build and actual recovery-stop/restart validation remain pending; these tests do not establish completed simulations.
