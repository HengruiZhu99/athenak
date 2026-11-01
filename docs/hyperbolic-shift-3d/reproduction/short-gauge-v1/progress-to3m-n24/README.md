# Provisional SG1 N16/N24 comparison through 3M

N24 job9019406 was confirmed running at capture,2026-11-01T06:59:17Z.
This is a measured live-history snapshot, not a terminal report or admitted
checkpoint. N24 continues toward10M under unchanged science. N16's reference
history comes from its previously verified completed10M campaign.

`capture.py` reads only complete newline-terminated rows, requires the3M row,
binds the known plan identity, and preserves scheduler/worker/binding evidence
in `snapshot.json`. The transferred21-row snapshot is SHA256-verified locally:
`904121500fc0bdfeef3ccac20d79050723b037e5d3e1316670443ac6c157a3f3`.
It does not overwrite an existing snapshot. `plot.py` also verifies the N16
history against its terminal worker artifact hash. No missing values or time
samples are interpolated. `histories.png` was visually inspected.

At3M, using each run's own complete candidate horizon:

| Diagnostic | N16 | N24 | N24 / N16 |
|---|---:|---:|---:|
| Exterior Hamiltonian RMS |0.01870572|0.01284911|0.6869|
| Exterior momentum magnitude RMS |0.02953466|0.02845861|0.9636|
| Exterior radial momentum RMS |0.04341354|0.03294796|0.7589|
| Full-domain Hamiltonian RMS |0.06654343|0.17188033|2.5830|

The cutoffs are0.07947950 and0.07964907. These are close but not identical;
a common-cutoff endpoint comparison remains deferred until verified matched
checkpoints are available. The momentum magnitude and radial diagnostics are
different quantities and must not be interchanged when quoting reductions.
The full-domain Hamiltonian history has large, irregular interior transients;
it does not show clean convergence through3M. The favorable exterior ratios do
not resolve that issue or establish an asymptotic order.

N16 accepted9/21horizon rows and N24 accepted17/21. At3M the N16 horizon is
unaccepted, so no N16 mass value is supplied. N24's accepted irreducible mass
is0.3333397678 versus reference1/3 (fractional error+1.93033e-5, or+0.00193033%).
All missing accepted masses remain gaps. Surface/ghost null residuals stay at
roughly1.6e-15. Floor/sign aggregates for the still-running N24 segment are not
yet available; they must be collected from its terminal native report.
