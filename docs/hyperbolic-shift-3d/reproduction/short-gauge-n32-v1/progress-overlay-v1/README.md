# SG1 N16/N24/N32 live constraint overlay

N32 job9027014 was running at capture, 2026-11-01T17:22:55Z. This immutable complete-row snapshot reaches3.75M; it is not terminal/checkpoint verification. Plan and scientific source/binary bindings match the frozen N32 campaign. N16/N24 histories are the preserved completed10M runs; their history hashes were checked against worker artifact records.

`constraints.png` shows the shared interval above and all available history below. Each run uses its own complete candidate horizon as exterior cutoff, including candidates without finder acceptance. These are production normalized compactified-coordinate volume RMS norms, not common-cutoff endpoint reconstructions. Missing values are not filled and N32 is not extrapolated. At3.75M, N32 H/M/Mr are .00874472/.02762259/.02969666, versus N24 .01086697/.02990461/.03293842. The reduction alone does not establish an asymptotic convergence order.

`capture.py` preserves complete rows and their hash, scheduler state and binding in `snapshot.json`; it refuses to overwrite a snapshot. `plot.py` verifies input history hashes and reproduces the figure and numerical `comparison.json`.
