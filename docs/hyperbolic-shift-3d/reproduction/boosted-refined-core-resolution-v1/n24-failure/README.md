# N24 BC2 terminal numerical failure

Job9037378 failed at last accepted6.1985M/12397steps, last diagnostic6.15M.
Native runtime2615.387s (43.590min); PBS walltime43:59. Native exit2,
workerfailed/PBS1, with no timeout or storage error. Frozen source, actual
binary, input, launcher and scientific configuration bindings verified.
All six worker artifacts are copied locally and hash/length verified; complete
checkpoint payload and failed capture are finite. BC2 header resolution/dt/time/
steps/tracker agree with the report. Admission flag is0; no restart is allowed.
Reproducer: `../preserve_n24_failure.py`.

The capture contains exactly one nonpositive volume metric: cell566804 at
(-.013125,.001875,.001875), r=.01339018, about1.98 cells from the last accepted
tracker. Minimum eigenvalue-.0009618394, determinant-.17961592. Chi.000314339
and conformal lapse.04217984 remain positive at this cell. The other two metric
eigenvalues are13.2381 and14.1065. The surface metric remains positive, minimum
eigenvalue.838420. This is immediate invalid core geometry, not a diagnosed
root cause or a lapse/chi sign stop. The rejected RK-stage time is unrecorded.

The tracker at(-.0200471,-.00002591,-.00001281) remains far inside the inner
gauge plateau; cube margin.159953. Like N20, the failing point is the first
positive-side cell next to y=0,z=0 Cartesian block faces. That structural
similarity is evidence to investigate, not proof that block-seam KO caused it.

Surface/ghost null maxima<=1.56e-15; cumulative pole/clock4.49492/10.03675.
Chi floors196468stage/65563final, negative chi136030stage/53144final.
There are no lapse-floor or surface sign/floor interventions. Event counts
include the rejected attempted step and are not normalized convergence norms.

Thirty-five of42 complete candidate records pass the post-only1e-2 expansion
selection, including last RMS.00834222. Twenty-six have original online1e-3
acceptance and recorded mass; the last recorded mass is3.75M. Missing masses
remain missing. At last diagnostic6.15M own-candidate exterior H/M are
.04848037/.08048696 with cut.13826872, not an admitted failed-time endpoint.

The unadmitted checkpoint SHA256 is
09002ae08bd9a62c3bb989f3b9ab4663df1b84b31308e9ae064b7c6687649082,
806091472bytes. Capture SHA256
1b5f26abbe08f59a078da17fb62abf2efc59c4a01663edc6afc19650605e43c8,
403045632bytes. Local preservation and matching terminal lock release are
recorded separately. No remote outputs were purged and no new run submitted.

See `../final-histories` for the completed three-resolution comparison. N16
survival10M and N20/N24 failures near6.2M do not establish robustness or
fourth-order convergence. Both requested higher-resolution runs are terminal;
the campaign monitor is retired after preservation and analysis.
