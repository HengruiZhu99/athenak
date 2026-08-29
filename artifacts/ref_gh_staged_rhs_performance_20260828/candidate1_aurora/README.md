# Candidate 1 Aurora result

The exact candidate commit `e02e8ced53a66ae45de5615ae8943081c217f8ac`
passed the bounded one/eight-tile full-output dynamic-q gate in Aurora job
8790518.  The one/eight conditioned Linf difference was
`1.36191095905485950e-14` against the unchanged `5e-12` tolerance.

Matched warmed benchmark job 8790530 completed successfully on one PVC tile.
The controlling warmup-subtracted complete-stage ratio was 14.956614 versus
9.244886 for the frozen baseline.  The main RHS ratio was 17.269546 versus
10.902828.  Candidate 1 is therefore rejected by the controlling retention
rule even though q-control and reference-update overhead targets pass.

The full compact remote runs remain at:

- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_qdevice_7335058c/runs/analytic_q_pvc_8790518.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`
- `/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_qdevice_7335058c/runs/analytic_q_performance_8790530.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`

The compact atomic q accumulator and component-staged analytic metric boundary
are equation-preserving PVC portability corrections.  The rejected element is
the production loop-form scalar source coupled to the 141-Real hot cache; it
reduced reported spill from 1,279 to 895 Reals but increased source time.
