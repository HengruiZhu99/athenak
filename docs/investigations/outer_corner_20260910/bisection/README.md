# N256 t200 bisection with boundary correction under qualification

PREPARED ONLY. No controller started. BOUNDARY_QUALIFIED.json is deliberately absent.

Candidate configuration: full_constraint_bjorhus + extrap_order2, vc_single_rank_device_sync=true. Bulk gauge, CFL0.15, diss0.50, AMR record/refinement policies, N256/block32, initial-data parameters remain the archived baseline. Diagnostics cover the full domain (history_constraint_radius1000) and save curvature every5 code units. The unchanged qualified production executable contains device synchronization already.

Before launch, inspect fresh endpoint qualification to t200, complete finite histories, global curvature/constraint locations, final restart evidence, and actual scheduler termination. Record the evidence and executable hash in BOUNDARY_QUALIFIED.json with approved_configuration. Do not manufacture this gate from t110 success alone.

Both endpoints are rerun fresh in the controller; failed/incomplete cases do not classify. Final global minLapse<0.01 at t200 defines collapse. Relative amplitude tolerance1e-5. Endpoint successes do not count as the two supervised midpoint cycles. The old campaign and monitor remain paused.

Optional --adopt-super /absolute/completed/run reuses a fresh A=-0.05 endpoint only after run-status0, matching executable/amplitude/settings/input hashes, finite t0->200 history and final restart checks. A reused endpoint does not allocate another GPU job. It does not count as a midpoint cycle. Sixteen controller tests pass, including an entire fake-allocation bisection with endpoint reuse. The qualification gate remains absent.

--adopt-sub is also supported under the same checks. Both fresh qualification endpoints may be reused;17 tests pass including both-endpoint reuse followed by automatic midpoint submission. This option avoids redoing the endpoint runs; it does not waive qualification.
