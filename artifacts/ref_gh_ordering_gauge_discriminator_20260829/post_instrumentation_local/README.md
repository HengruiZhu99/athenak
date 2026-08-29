# Post-instrumentation local smoke evidence

Date: 2026-08-29 (America/New_York)

These are compact results from local Release/Serial smoke tests of the new
diagnostic-only code.  They do not qualify the MPI implementation, Aurora PVC
execution, the A-D evolution matrix, or puncture stability.

The unchanged full `ref_gh_q_controlled_reference.athinput` source-unit gate
passed after instrumentation.  Selected unchanged results were:

- analytic coefficient oracle: 216 samples, `8.88178e-15`;
- expanded radial oracle: 2160 samples, `1.48837e-13`;
- generated geometry oracle: 2376 samples, `2.33147e-15`;
- moving gauge/dtTheta oracle: 2160 samples, `1.24829e-14` motion error;
- compact boundary oracle: 2160 samples, `4.56474e-14` metric error;
- compatible/standard all-61 RHS oracle: 4320 samples, `2.84217e-14`;
- production cache oracle: `1.63758e-14`;
- exact Minkowski evolution: zero maximum error.

A 16^3 cycle-zero stationary-trumpet sector smoke reported:

- production RHS Linf: `5.80750631801021725e-14`;
- maximum component/radius: component 10 at `r=1.5562374497`;
- sector-sum conditioned error: `3.94430452610505903e-31`;
- exact production rerun difference: zero.

At that maximum, the recorded Pi-sector contributions were approximately:

| sector | signed contribution |
|---|---:|
| principal | `1.38415015525282459e-16` |
| covariant vacuum | `5.67804240284055353e-14` |
| ordinary gauge | `1.28470824131400861e-15` |
| gamma0 | `1.89174605287627989e-18` |
| gamma2 | `-1.27124748030785766e-16` |
| KO | `-3.25110316474429363e-18` |

Cycle-zero 16^3 smokes for the frozen A-D input configurations also completed,
with initial RHS Linf values `5.69207e-14`, `5.68505e-14`, `5.82022e-14`, and
`5.80751e-14`, respectively.  These low-resolution smoke values are execution
checks only and must not be used to classify the formulation.
