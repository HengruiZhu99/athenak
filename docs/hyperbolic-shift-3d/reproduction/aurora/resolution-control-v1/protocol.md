# Resolution discriminator for the original scri failure

The user requested this control before further boundary development. It tests
whether the original Minkowski instability is a consequence of very low spatial
resolution and measures throughput with larger blocks. It is not a convergence
admission and does not qualify a candidate formulation.

Use the **exact original Aurora v4 executable**, SHA256
`6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef`,
from numerical commit `113aaff783008204e1ff8cbf2fb8bb5f2af2204c` and source
manifest `3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f`.
The equations, unprojected tensor evolution, face-local scri derivatives, KO,
fixed 19-block geometry, double precision and 12-rank tile placement stay fixed.
Every case starts from the analytic Minkowski equilibrium. No failed checkpoint
is restarted. The updated launcher merely permits a reviewed sequence of
independent resolution controls after a recognized numerical rejection.

| Case | Cells per block | Active volume cells | dt | Requested coordinate endpoint |
| --- | ---: | ---: | ---: | ---: |
| Existing control | 16^3 | 77,824 | .0005 | failed at .4895 |
| New spatial control | 24^3 | 262,656 | 1/3000 | .6 |
| New spatial control | 32^3 | 622,592 | .00025 | .6 |
| New timestep control | 16^3 | 77,824 | .00025 | .6 |

The n24/n32 timesteps keep n*dt fixed relative to the original. The additional
n16 run halves dt at fixed mesh. Run order is n32, n24, then n16 with half dt;
each has its own fresh output and bounded time allowance in one one-hour debug
allocation. Numerical failures remain failures, and their checkpoints remain
unadmitted even if the launcher proceeds to the next independent case. Timeouts,
storage failures and infrastructure errors stop the sequence. Healthy walltime
checkpoints are bounded outcomes, not completed endpoint results.

Assess state, H, M, Z4 and null histories at common coordinate times. Compare
exponential growth rates in the resolved linear-growth interval as well as
failure times: different grids can seed different floating-point amplitudes,
so failure time alone is insufficient. Reduced error/growth with refinement
would support a coarse-grid limitation; comparable or increasing growth despite
smaller dx and CFL-scaled dt would argue against that explanation. The timestep
control distinguishes time-discretization sensitivity. These controls alone
cannot identify the unique unstable operator.

Measure median elapsed time per accepted step from history intervals after
startup, and report active cell-steps per second. Keep the runner's bulk,
closure and exchange timings separate because the closure timer includes some
communication. Larger blocks supply more work per GPU, but speedup must be
measured rather than assumed.

Retain small reports, histories and failure hashes. Budget 1.6 GB of new charged
output plus a 700 MB atomic-checkpoint reserve. Aurora home files have two
Lustre mirrors; n32's 311,853,128-byte logical checkpoint is charged approximately
624 MB. Archive and purge redundant campaign snapshots before admission when
needed to remain below the 5,000,000,000-byte cap.
