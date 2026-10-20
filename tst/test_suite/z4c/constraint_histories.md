# Z4c constraint histories

Set `z4c/history_interior_radius = R` with `R > 0` to enable interior columns.
Their integration region contains cell centers satisfying both
`x*x + y*y + z*z < R*R` and `chi >= z4c/excise_chi` (default `0.0625`).
The ball is centered at the coordinate origin and does not follow the puncture.
There is no inner-radius cut or horizon-surface excision.
The obsolete `history_inner_radius` input is ignored with a warning.

`interior-H2`, `interior-M2`, and `interior-C2` are squared, proper-volume
integrals. For example, `interior-H2 = sum(H^2 sqrt(abs(det(gamma))) dx dy dz)`.
Take its square root for the integrated L2 norm, or divide by `interior-volume`
before taking the square root for the RMS. The original nine full-grid columns
keep their positions and definitions, including the same chi cutoff.

The coordinate ball stays fixed over time. Its chi-selected cells and proper
volume can still change as chi and the metric evolve; a fixed ball does not
imply a constant proper volume. `interior-coordinate-volume` measures the
coordinate volume of the cells passing both cuts, and
`interior-chi-excluded-volume` measures the proper volume rejected by chi
inside the whole ball.

`history_boundary_buffer + history_boundary_speed * time` defines a separate
boundary-travel audit. `interior-boundary-risk-volume` counts the proper volume
inside the ball whose distance to the nearest outer face is at most this
value, including chi-excluded cells. It never removes cells from the norms.
This column replaces the former `boundary-excluded-volume` in the same slot.
`speed-bound-violation-volume` measures sampled speed-bound violations over
the full grid; `unexcised-speed-violation-volume` applies the chi cutoff to
that diagnostic. These estimates do not establish boundary independence.

Do not append new outputs to old history files: the old interior histories
used an inner-radius cut and a shrinking boundary mask. Existing results
cannot be corrected from their scalar histories alone.

The regression `test_z4c_history_cpu.py` independently sums 3D binary fields
on a small puncture grid. It checks the radius and chi cuts, ignores the old
inner cut, and repeats at a late start time with an entirely unsafe boundary
envelope to ensure the integration domain does not erode.
