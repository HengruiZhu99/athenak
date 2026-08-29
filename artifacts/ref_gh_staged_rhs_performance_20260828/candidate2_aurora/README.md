# Candidate 2 Aurora result

Candidate 2 (`7e8ab79b41213546be882b22d728b7f95eec3051`) passed the
bounded full-output PVC discriminator in job 8790555.  Both one-rank and
eight-rank dynamic-q RK cycles completed on eight distinct PVC tiles and their
conditioned history Linf difference was `1.56518125145110397e-14`, below the
unchanged `5e-12` gate.  The executable SHA-256 was
`8fcfef501b8b02080854222cf2acf1ce478038a66df82804c3b6991cb3942adb`.

The matched warmed `64^3` run and synchronized kernel profile completed in job
8790573.  Warmup-subtracted complete-stage time was `0.10803409375 s`, or
`11.4388x` the matched Z4c control.  Ref-GH main-RHS time without dissipation
was `0.0909013228 s`, or `12.7851x` Z4c.  Although the componentized scalar
source reduced the FD4 compiler spill warning to 7 Reals, it costs
`0.0584220886 s` per stage (54.08% of the complete stage).  Physical
geometry/gauge costs another `0.0277257232 s` (25.66%).  The source preparation
kernel is only 0.82% at runtime despite a 903-Real spill warning.

The three overhead gates pass: q control is 0.0836% of a stage, compact
reference preparation is 7.26%, and dynamic/static complete-stage time is
1.08865.  The central performance gate fails, and the complete stage is slower
than both the frozen production baseline (`9.244886x` Z4c) and rejected
candidate 1 (`14.956614x` Z4c only in the sense that candidate 2 is less slow
than candidate 1).  Candidate 2 is therefore correctness-qualified but
rejected as the production implementation.

Full remote artifacts remain at:

- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260829_candidate2_475e0963/runs/analytic_q_pvc_8790555.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260829_candidate2_475e0963/runs/analytic_q_performance_8790573.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The retained compact copies intentionally omit restart dumps.  The profile
justifies the next bounded equation-preserving discriminator: replace the
low-spill but arithmetic-heavy componentized covariant source with a staged
standard-coordinate GH source followed by the analytic frame transform, while
remaining below the hard 128-Real transient-scratch cap.
