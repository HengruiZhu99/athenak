# Fixed-snapshot puncture horizon measurement audit

No evolution, restart, admission, tolerance change, remote-file modification or
production-code change was performed. All diagnostics have finished. Source
records, command lines, hashes, exact results and logs are retained alongside
this note. The original checkpoint still has admitted=0 and its hash is unchanged.

The frozen puncture run, job 8982136, first fails its horizon acceptance at
coordinate t=0.20. Sampling is complete and valid, but all 300 iterations are
used and trial expansion RMS is 0.00148397, above the fixed 0.001 tolerance.
At t=0.25/0.30/0.35 it again exhausts 300 iterations with complete sampling and
RMS 0.00258087/0.00393585/0.00411887. This does not establish that the horizon
disappeared. The evolution later fails at t=0.369 (1.107M), step738, because of
nonpositive interpolated scri geometry.

The first unsuccessful search uses the preceding found radius, approximately
0.06564658. After a failure the existing finder falls back to its input radius
0.07. Therefore this fallback is not the cause of the first failed search.

Only the final unadmitted rollback state at **t=0.369** was available for the
fixed-snapshot measurements below; the earlier t=0.20 state was not reproduced.
The diagnostic directly reads the native V1 snapshot and retains admitted=0.
It uses the same FastFlow, physical ADM reconstruction and six-point SampleJets.
Every sampled stencil is guarded to lie entirely in active Cartesian-core
cells. Ghost values are NaN, no scri reclosure is attempted, and every run reports
zero rejected stencils. Thus the known remote outer-boundary failure cannot
prevent this strictly local horizon measurement.

All runs use seed radius 0.066, flow coefficient 0.1, expansion RMS tolerance
0.001 and mass-change tolerance 1e-10:

| lmax | ntheta | Iteration cap | Actual iterations | Expansion RMS | Found |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 12 | 300 | 300 | 0.005737686347859011 | no |
| 6 | 12 | 1000 | 1000 | 0.005737686347859036 | no |
| 12 | 24 | 1000 | 111 | 0.000220899942846602 | yes |

The first two trial radii are identical, 0.06428436117755529. Their RMS values
agree to roundoff after 700 additional iterations: this is a demonstrated
iteration floor for the original angular settings, not merely insufficient
iteration count. The higher-angular result has mean coordinate radius
0.06428375331588937, areal radius 0.6669942559314379, and mass
0.33349712796571895. No tolerance relaxation was needed.

This demonstrates an angular measurement-resolution limitation on this saved
state. Because lmax and quadrature resolution changed together, the tests do
not separately isolate basis truncation and quadrature. They also do not
establish continuum accuracy: the candidate radius spans only about 4.1 core
cell widths at n=16 (dx=0.015625). More angular modes do not add spatial data.
The active-cell annulus 0.04<r<0.09 remains positive, with minimum metric
eigenvalue 0.99752, maximum determinant error 7.50e-6, and maximum A trace
1.95e-4. FastFlow reconstructs the full physical curvature, including this
off-constraint trace contribution.

The recovered local horizon does not admit the globally failed trajectory or
justify continuing its checkpoint. The result supports improving measurement
resolution before considering a looser acceptance tolerance. Further horizon
and candidate experiments were stopped after these three runs in accordance
with the user's request to prioritize unchanged-formulation spatial-resolution
stability controls. Independent angular quadrature and spatial refinement remain
future accuracy checks, after a stable evolution has been established.
