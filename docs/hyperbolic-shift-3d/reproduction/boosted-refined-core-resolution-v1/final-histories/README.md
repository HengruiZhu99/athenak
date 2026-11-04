# Final boosted BC2 refined-core resolution comparison

The authorized N20/N24 campaign has terminal numerical outcomes. N16 is the
preserved completed baseline. These results do not establish a stable or
converged boosted scheme.

| Resolution | Nodes/ranks | Last accepted t/M | Native minutes | Disposition |
|---|---|---:|---:|---|
| N16 | 2/24 | 10 | 39.302 | Completed, admitted endpoint |
| N20 | 4/48 | 6.2082 | 32.219 | Invalid metric, unadmitted |
| N24 | 4/48 | 6.1985 | 43.590 | Invalid metric, unadmitted |

N20 and N24 both fail near6.2M despite finer space/time grids. They use the same
48-rank/four-node layout, which removes the N16-to-N20 node-count change as a
difference between those two failures, but does not independently validate MPI
or identify a unique cause. No failed state is restarted or treated as a10M
endpoint. The user-discussed coordinate-normalized inner Gamma-driver has NOT
been implemented, built or submitted; these remain the original frozen BC2.

`../plot_final_histories.py` verifies plan/job/source/binary/report/history
identities separately for all three cases. It plots recorded points without
smoothing, interpolation or extrapolation to the failed stages. N20/N24 last
diagnostics are6.15M. There are41 three-way exact-time matches through6M (within
1e-9M matching tolerance). `comparison.json` preserves those samples and cuts.

At6M, the own-candidate H norms are.0456424/.0430494/.0483939 forN16/N20/N24;
momentum norms are.0588174/.0627386/.0786623. The enclosing cuts are
.140044/.143280/.138338. Early momentum errors generally decrease with
resolution, but late recorded norms do not improve uniformly. These differing
moving domains are not a fixed-domain convergence study, and no Richardson or
fourth-order claim follows. Large unexcised interior norms alone are not used
as evidence of an exterior accuracy failure.

Both higher resolutions have a core metric failure with positive local chi and
lapse, while scri null closure remains at roundoff. The failure reports/captures
are localized independently in `../n20-failure` and `../n24-failure`. N16's
existing admitted endpoint radial/profile reconstruction remains preserved in
`../../boosted-refined-core-v1/completed`; it is not rerun. There is no admitted
N20/N24 endpoint for a same-time10M radial comparison. Failed captures are
forensic evidence only, not substitutes for successful endpoints.

## Horizons, trajectory and boundary diagnostics

Online accepted candidates/mass records:25/68,26/42,26/42 forN16/N20/N24.
Last recorded masses are at3.75M in all three. Post-only complete-candidate
expansion RMS<=1e-2 counts are65/68,40/42,35/42. The N24 final recorded RMS
is.00834222 and passes that selection. Missing masses remain gaps: post RMS
selection does not establish the unrecorded mass-stationarity condition or
recover absent area/mass values. Original online1e-3 settings are unchanged.

The trajectories remain in the inner gauge plateau. The late return of the
puncture toward the origin coincides with the failures; this correlation does
not establish causality. Surface/ghost null maxima remain<=1.67e-15. Cumulative
pole maxima are6.0336/5.2767/4.4949, clock maxima14.013/12.041/10.037. They
are historical maxima, not instantaneous late-time regularity measurements.

## Segment totals and cost qualifications

Each case has one segment. Chi-floor stage/final counts are22188/7484,
80987/27153,196468/65563 forN16/N20/N24. Pre-floor negative-chi counts are
19392/7000,53571/20571,136030/53144. N16 has2272/760 negative-lapse events
and3032 lapse-floor events; N20/N24 have none. Boundary sign/floor counts are
zero and are recorded separately in comparison.json. Counts span different
numbers of cells, timesteps and durations; they are not normalized errors.

Native timings include initialization and diagnostics. Raw per-rank phase
timings remain in the bound reports. Changed work, resolution, duration and
node count prevent a controlled strong-scaling conclusion. No additional
scaling or formulation run was launched.

Both final figures were visually inspected; hashes are in figure-inspection.json.
The campaign follow-up preserves the failures rather than claiming success from
N16 survival. A new gauge/numerics control requires separate authorization.
