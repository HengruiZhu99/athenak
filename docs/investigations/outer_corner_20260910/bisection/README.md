# N256 lapse bisection with corrected outer boundary

PREPARED ONLY: controller not started and BOUNDARY_QUALIFIED.json absent.

Classification requested by the user:
- At any completed timestep, global minimum lapse strictly below 1e-5 means collapse and clean early termination.
- Otherwise continue to coordinate t=200 in existing code units. Final global minimum lapse strictly below 0.01 means collapse; otherwise disperse.
- Relative amplitude bracket tolerance is 1e-5. Failed/nonfinite or incomplete evolutions cannot update the bracket.

Native runtime parameter problem/collapse_lapse_threshold=1e-5 checks canonical active vertices with a device reduction and MPI global minimum. Zero disables it. A successful early stop writes a schema3 collapse_lapse termination record and final outputs through the driver's normal stopping hook. No horizon finding is enabled.

Settings remain N256/block32, CFL0.15, diss0.50, original gauge and initial-data family, live AMR record policy, full_constraint_bjorhus with extrap_order2, device shared-node synchronization. Full-domain constraints and curvature output every5 code units are retained.

Before launch, require an executable-qualified gate and actual short restart validation of the new stop hook. Existing historical endpoint reuse requires explicit source/input/output provenance; a different historical executable requires explicitly authenticated file hashes in the qualification gate. Only clean completed segments are imported; cancelled continuations are excluded. Do not change a cancelled job's exit status to manufacture successful evidence.

28 workflow tests pass locally, including strict thresholds, required t200 completion without an early stop, rejected malformed stop markers, and mock automatic bisection successors. These are not actual midpoint cycles. Supervise two actual midpoint cycles and automatic successors before activating the existing two-hour monitor.
