# Common-hierarchy symmetric-O4 Brill convergence campaign

This campaign records the accepted `dchi_max=0.01` N256 AMR hierarchy and
replays that exact physical-time/LogicalLocation schedule at N128 and N512.
All three cases use the same 4x8 root MeshBlock lattice, O4 method, RK4,
`CFL=0.15`, gauge, damping, KO dissipation, and physical domain. Only active
cells per MeshBlock change (16, 32, and 64 per active direction).

The campaign is fail closed. `aurora_build_qualify.pbs` must pass before any
long authority or replay segment is submitted. The final evidence tree and
scientific verdicts are populated only from authenticated Aurora runs.
