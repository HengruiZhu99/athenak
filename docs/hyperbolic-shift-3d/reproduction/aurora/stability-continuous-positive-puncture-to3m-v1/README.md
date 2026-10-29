# V11 puncture continuation to 3M

Job 8986656 reached coordinate t=1 = 3M in 2001 total steps, 1701 in this
segment, at fixed dt=.0005. Native status completed with no RHS error, positive
chi/lapse, zero floor interventions, and null residual <=1.56e-15. Segment time
1152.77 s; together with the parent pilot, measured evolution time is 1375.47 s.

**Overall scientific validation failed.** The final fast-flow expansion RMS
.00139189 exceeds the unchanged .001 tolerance after 300 iterations. No accepted
final horizon mass or exterior-constraint norm is available. Eight final gates
fail for the absent horizon and its dependent diagnostics. These are numerical
validation failures, not the metadata mistakes in the earlier pilot. The worker
and scheduler failure are preserved; the independent verifier confirms evidence
integrity and reports scientific_checks_passed=false.

Final min chi .0001208043, min conformal lapse .00937907, full-domain H/M norms
.0518183/.0371362. Max pole/clock residuals .0187866/.0155758 remain finite but
are not roundoff, unlike the algebraic null condition. Native checkpoint is
restartable, but this is not a long-time, convergence, or settled-trumpet
admission. Horizon convergence and exterior diagnostics need repair before
claiming accurate 3D puncture reproduction.

Source and binary remain frozen V19/faee66ea3, the same as the parent pilot.
The new boundary performance candidate is not used in this duration run.
