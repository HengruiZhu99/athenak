# Read-only remote review prompt

Repository: https://github.com/HengruiZhu99/athenak

Branch: `codex/cartoon-allbulk-brill-scaleinv-20260813`

Investigation bundle:
https://github.com/HengruiZhu99/athenak/tree/codex/cartoon-allbulk-brill-scaleinv-20260813/docs/investigations/brill_figure3_five_run_domain64_20260815

Please perform a strictly read-only mathematical and source-code review of the
AthenaK axisymmetric Cartoon Z4c Brill-wave Figure-3 investigation.  Do not
propose claims that require data not present in the repository.  Begin with
`report.pdf`, `report.tex`, `analysis/five_case_summary.json`, and the four
plots, then inspect the relevant gauge, Z4c, Cartoon, AMR, restriction,
prolongation, dissipation, and history-output code on this branch.

The authenticated case is the direct in-AthenaK interpolation of the IrisK
two-dimensional `A=-0.047` Brill coefficients with pre-collapsed lapse,
scale-invariant telegrapher lapse `(tau,kappa)=(1,1)`, zero Z4c constraint
damping, `dchi_max=0.02`, and strict positive-chi gates.  The five controls
vary fixed Gamma-driver versus zero shift, KO `0.02` versus `0.5`, and outer
boundary 16 versus 64.  None reaches the target.  The matched boundary pairs
track each other extremely closely before late catastrophic constraint and
curvature growth; the enlarged zero-shift case reaches AMR level 20 and 13,580
blocks before the strict chi gate rejects 2,346 parent stencils.

Our current inference is deliberately limited: the tested shift condition,
KO increase, and outer-boundary location are not individually curative, and
the boundary evidence strongly disfavors a direct boundary artifact.  Please
check whether the source and artifacts actually support that statement and
identify any hidden confounder.

Then recommend the smallest decisive next step.  In particular, analyze a
prospective higher-resolution test that increases cells per MeshBlock (for
example, 32 to 64 in each evolved 2D direction) while keeping the logical
MeshBlock layout, physical domain, AMR criterion, maximum level, gauge, KO,
and refinement policy unchanged.  Explain whether that really preserves the
refinement structure in this implementation, what additional capacity and
memory consequences follow, and what telemetry would distinguish:

1. insufficient resolution or ordinary truncation-driven instability;
2. instability caused by dynamic AMR selection or coarse-fine operations;
3. a resolution-independent formulation or gauge instability.

Also propose a separate one-factor-at-a-time study of initial/base resolution
and `dchi_max`; do not conflate those changes with the first in-block
resolution comparison.  Specify prospective pass/fail criteria based on
pre-terminal constraint convergence, terminal time and location, AMR level and
physical coverage, MeshBlock inventory, timestep collapse, and strict-chi
rejections.  Check whether a prescribed or replayed refinement mask is needed
for a genuinely identical refinement structure.

Finally, rank the plausible failure mechanisms from the available equations
and evidence, identify the exact source locations supporting your reasoning,
and state clearly which conclusions are observations, inferences, or open
hypotheses.  Do not recommend weakening the chi gate, adding a floor, or
changing several controls simultaneously unless you can justify why that is a
cleaner discriminator.
