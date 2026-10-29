# Fresh V7 matched puncture control submitted

Job **8984472** uses the qualified source-v14 V7 binary, one
CompactBinaryMerger/debug node,12 MPI ranks and12 GPU tiles. It starts from
the same n16 isotropic initial data as V6, with dt=.0005 and target coordinate
time1=3M. The horizon settings remain lmax12/ntheta24/nphi48, expansion RMS
limit.001,300 iterations; all prior physical limits are retained.

Only the already-qualified inner-transition interpolation differs from V6.
The extra report check binds that policy explicitly (46 checks total). No
failed checkpoint is resumed and no lapse floor is added. The V6 comparator
failed at2.8905M with a negative core-stage lapse and lost accepted horizons
from1.95M onward.

This control follows independent verification and release of Minkowski
job8984359. That run reached t2 but failed the state-error limit and retained
inner-interface growth. The puncture control therefore tests the persistence
of the separate central lapse/horizon failure; it does not promote V7 as
stable. The declared interval is short enough to compare the original core
failure before the much smaller Minkowski disturbance becomes nonlinear.

No puncture duration result is available yet. The retained plan, binding and
submission identify the exact binary, initial data, source and preflight
admission. The binding reports a3.43GB campaign footprint, below the5GB cap.
