# N16 smoother-profile continuation: numerical failure

Job9009718 failed after advancing from3M to last accepted time9.157500000000724M
(coordinate3.0525000000002414,6106 total steps). The native command returned2;
worker and scheduler record failure, with no external timeout or storage error.
The error is "device RHS failure at active cell3071 reason7". Reason7 maps to
nonfinite_bulk_RHS in hyperbolic_composite_device.hpp, not the scri closure.
The failed checkpoint is unadmitted and must never be resumed.

The last recorded diagnostics are from9.15M, not the failed stage. They show
surface/ghost null residuals1.22e-15/1.55e-15, no lapse-floor interventions,
H exterior RMS.0077909 and radial momentum exterior RMS.0136567. The exterior
cut uses the complete candidate radius.1066377; its expansion RMS.0028252
exceeds the.001 acceptance tolerance. Preserve that provisional qualification.

The matching fixed-grid coordinate table places first reported bad cell3071
at r=.05124619, inside that candidate radius. Captured volume values are finite;
at this cell chi=.02430564, physical lapse=.02714535 and all metric eigenvalues
are positive. Field18 stores the lapse perturbation, so its negative stored
value is not a negative physical lapse. This local check does not yet identify
which RHS term became nonfinite or why. Do not infer a scri failure or claim
that the complete geometry is healthy from one cell.

All worker artifacts were SHA256/length-verified locally before remote cleanup;
full checkpoint and rejected-stage capture remain in
output/hyperbolic-core-shell/smooth-gauge-20m-v1/campaigns/n16-smooth-gauge-to20m-v1-segment01/output/.
Following the authorized protocol, N16 is terminal and N24 proceeds from its
own verified3M GW1 state with unchanged equations and parameters.
