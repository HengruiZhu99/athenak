# Signed state budget fixture

Build AthenaK with `PROBLEM=../../analysis/pc_gh_clean_reduction/state_budget_oracle`
and run the committed `state-budget-cpu-{2,3}d-001/used_input.athinput` inputs from
the dated qualification directory, each in a new output directory. Run
`read_state_budget.py OUTPUT.pcgh-reduction.csv.state-budget.rank0.bin --oracle`.
The test adapter requires one block and nlim=0. It writes two known operations
following five actual uniform startup operations; every active and ghost cell
is checked with independently indexed signs. The first reader version expected
only the two adapter operations and failed its record-count assertion. Inspection
identified the five initialization records; all are now explicitly checked zero.
The numerical tolerance did not change.

Production switch `state_budget=true` enables existing reduction-monitor brackets
and a raw post-RK bracket, sampled by positive `state_budget_dcycle`. Default is off.
The binary format is described directly by the independent reader's FIELDS and
the writer. Each record contains explicit mesh geometry, active bounds, field
count and endianness. Payload is n,k,j,i,{before,after,after-before}, float64.
NaNs are retained, not filtered. Both ghost-valid flags are zero (unasserted).

This first writer snapshots only u0, including allocated ghosts. It does not
snapshot coarse_u0, so restriction can legitimately have zero recorded increment.
Operation 2 still combines algebraic and GH projections as in the historical
bracket. These are stated remaining diagnostic gaps, not complete causal budgets.
File appending requires a fresh output directory for an unambiguous run epoch.
Targeted fixtures only: complete state snapshots can become large in evolution.
