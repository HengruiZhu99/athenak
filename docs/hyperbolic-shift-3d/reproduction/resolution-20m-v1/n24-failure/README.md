# Location of the N24 failure

The last accepted time was 5.869M. The next attempted final RK volume state
triggered `invalid accepted volume algebraic projection`. The phrase "accepted"
names the final-step projection routine; this failed candidate was rejected.
It does not mean that a bad step was admitted as a restart.

Analysis of the hash-verified failed-stage capture finds one indefinite conformal
metric, at Cartesian (0.00375, 0.00375, -0.01125), radius 0.0124373429638.
Its eigenvalues are approximately (-0.0780890, 1.4235464, 3.5345877), with
determinant -0.3929166. All volume fields in the capture are finite. The bad
cell's chi is positive, 0.0006574554. This is a metric-positivity failure near
the puncture, not a negative-chi stopping condition.

The algebraic projector enforces det(g)=1 and trace_g(A)=0. It first checks
positive definiteness; normalization cannot turn this indefinite metric into
a valid spatial metric. Lapse and chi floors do not impose positive definiteness
on g. The separate scri null closure is not the source of this error message.
All saved surface metrics are positive, minimum eigenvalue 0.9816453. The saved
surface snapshot precedes final surface acceptance, so that observation does not
prove that all boundary dynamics are harmless or establish the instability's cause.

Reproduce with `inspect_failed_volume.py` from the n16-timestep-to3m-v1 evidence
directory, passing the local n24-ac1-segment01 directory and n24/coordinates.csv.
The capture and coordinate SHA256s are in failed-metric-analysis.json.
