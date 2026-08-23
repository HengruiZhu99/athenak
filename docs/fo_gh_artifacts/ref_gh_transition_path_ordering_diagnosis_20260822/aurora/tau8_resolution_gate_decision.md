# Fixed-core tau-8 three-resolution decision

Candidate: fixed core `r_core=0.30M`, `tau_transition=8M`, compatible Phi
ordering, controller off.

Result: PASS for the bounded open-loop `t=4M` gate.

- All M/16, M/24, and M/32 runs reached `t=4M` through an actual checkpoint
  restart with finite state, positive physical lapse, admissible relative
  metric, and zero controller state.
- Final GH L2 orders are 3.763 and 3.892; curl orders are 3.051 and 3.789;
  reduction orders are 3.262 and 2.711. Psi-reference L2 decreases at orders
  0.713 and 0.989.
- The common ADM momentum L2 improves at orders 3.009 and 3.358.
- The common ADM Hamiltonian L2 in the 2--4M and 4--8M shells grows with
  resolution after about t=2M. This does not cause a resolution-dependent
  failure and lies outside the 0.30--0.60M transition shell, but it blocks any
  broader physical or production qualification.
- The path is only half transitioned at `t=4M`. No stability beyond `t=4M`,
  completed transition, or dynamic regularization is established.
- A feedback smoke cannot legally activate: `r_full=0.60M` is already outside
  `r_fit_min=0.15M`, even before the required grid-cell buffer. The goal forbids
  moving the fitting shell, so no feedback job was submitted.

Machine-readable values are in `tau8_resolution_gate.json` and
`tau8_resolution_gate.tsv`.
