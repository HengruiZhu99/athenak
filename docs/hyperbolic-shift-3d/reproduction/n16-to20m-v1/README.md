# n16 continuation from 3M toward 20M

Continue the exactly reproduced 3M checkpoint at dt=.0005 (dt/dx=.0444444),
with unchanged frozen SYCL/MPI executable and numerical coefficients.
Target coordinate time is 20/3 for M=1/3. Checkpoint SHA256:
628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31.

Use one CompactBinaryMerger debug node, 12 MPI ranks/GPU tiles, one-hour walltime.
Native worker deadline leaves time for an admitted checkpoint. The user requests
30-minute monitoring and automatic resubmission after a healthy walltime stop.
Do not restart a numerical failure or an unadmitted checkpoint. All segments use
fresh output/plan directories and the same source/executable hashes. Keep home
campaign footprint below 5 GB; copy and hash-verify large evidence before purge.

Remote root: /home/hzhu/hyperbolic-core-shell-20261028.
First plan: campaigns/n16-to20m-v1-segment01/plan.json.
Controller: launcher-timestep-v3/control.py; source manifest:
source-timestep-v1-manifest.json. No evolution rebuild is required.

Each monitor run should inspect saved submission/worker records and qstat -fx.
Do not duplicate an existing queued/running job. A terminal checkpointed run can
resume only after validating header HYP3D_MPI_DT1, n=16, nf=31, kind=2, admitted=1,
dt=.0005, finite time in [1,20/3), intact length 152+16*(31*cells+surface), and
artifact SHA256. Match native report/header time, steps, and admission. Keep
source, binary, dt, horizon settings and diagnostic interval unchanged. Carry
an existing bound horizon-seed sidecar only if it matches the checkpoint.
Release the terminal plan's root lock using collect --release-terminal-lock;
prepare a new sequential segment plan, bind, and submit. Numerical, storage, or
ambiguous scheduler failures require inspection; never treat them as walltime.
Check all user debug jobs before submission and avoid concurrent allocations.
On 20M completion collect and verify evidence, report results, pause the monitor.
If SSH expires, request renewal without repeated submissions.
