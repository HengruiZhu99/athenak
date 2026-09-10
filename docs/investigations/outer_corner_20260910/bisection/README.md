# N256 t200 bisection with boundary correction under qualification

PREPARED ONLY. No controller started. BOUNDARY_QUALIFIED.json is deliberately absent.

Candidate configuration: full_constraint_bjorhus + extrap_order2, vc_single_rank_device_sync=true. Bulk gauge, CFL0.15, diss0.50, AMR record/refinement policies, N256/block32, initial-data parameters remain the archived baseline. Diagnostics cover the full domain (history_constraint_radius1000) and save curvature every5 code units. The unchanged qualified production executable contains device synchronization already.

Before launch, inspect fresh endpoint qualification to t200, complete finite histories, global curvature/constraint locations, final restart evidence, and actual scheduler termination. Record the evidence and executable hash in BOUNDARY_QUALIFIED.json with approved_configuration. Do not manufacture this gate from t110 success alone.

Both endpoints are rerun fresh in the controller; failed/incomplete cases do not classify. Final global minLapse<0.01 at t200 defines collapse. Relative amplitude tolerance1e-5. Endpoint successes do not count as the two supervised midpoint cycles. The old campaign and monitor remain paused.

Validation: 9 unit/integration tests pass locally, including a fake-allocation complete bisection and crash/nonfinite/incomplete midpoint rejection. These do not substitute for two supervised real midpoint cycles.
