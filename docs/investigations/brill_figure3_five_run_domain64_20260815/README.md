# Brill Figure-3 five-case progress report

This artifact assembles the three authenticated original-domain controls and
the two enlarged-domain controls requested for the IrisK `A=-0.047` Brill
datum.  The physical case inventory is exactly five:

1. `R=Z=16`, fixed Gamma-driver shift, KO `0.02`;
2. `R=Z=16`, fixed Gamma-driver shift, KO `0.5`;
3. `R=Z=16`, zero shift, KO `0.5`;
4. `R=Z=64`, fixed Gamma-driver shift, KO `0.5`;
5. `R=Z=64`, zero shift, KO `0.5`.

All use direct in-AthenaK interpolation of the authenticated two-dimensional
IrisK coefficients, pre-collapsed lapse `alpha=psi^-2`, scale-invariant
telegrapher lapse `(tau,kappa)=(1,1)`, zero Z4c constraint damping,
`dchi_max=0.02`, and strict positive-chi gates.  The enlarged cases preserve
the base spacing `0.25`; only the outer domain and the capacity needed for the
larger root grid differ.

The enlarged cases were executed serially on one `shared_interactive`
A100-SXM4-80GB.  The zero-shift case spans an initial segment and a one-case
restart continuation because the first step reached its wall-time limit.
Those segments are deduplicated at the restart time and treated as one
physical curve, not a sixth case.

Across the tested matrix, neither changing from the fixed Gamma-driver shift
to zero shift, increasing KO dissipation from `0.02` to `0.5`, nor moving the
outer boundary from 16 to 64 produces a stable evolution.  The boundary-pair
trajectories agree extremely closely before their terminal growth, so the
outer boundary is specifically disfavored as the cause.  Shift and KO still
alter details and terminal times, but the evidence does not support either as
the sole failure mechanism.

The report records, but does not execute, a clean next diagnostic: increase
the number of cells per MeshBlock while preserving the logical block layout,
physical domain, AMR criterion, maximum level, gauge, and dissipation.  A
separate one-factor-at-a-time study should then vary the initial resolution
and `dchi_max`.  This is intended to distinguish insufficient resolution,
AMR-selection sensitivity, and a resolution-independent formulation or gauge
failure.

Run `bash build_report.sh` to regenerate the four comparison figures, compact
machine-readable summary, generated LaTeX table, and `report.pdf`.  Run
`bash verify_report.sh` after the final self-excluding SHA-256 manifests have
been frozen.  The report is diagnostic: it does not claim a horizon search,
convergence qualification, or complete reproduction of the published Figure
3.  Rendered paper curves are comparison graphics rather than raw published
numerical data.
