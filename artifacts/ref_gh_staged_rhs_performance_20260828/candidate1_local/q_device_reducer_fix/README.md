# Device-result q reduction correction

The candidate-1 PVC failure exposed a SYCL portability issue in the compact
q-controller reducer destination.  The mathematical reduction and MPI
collective are unchanged; the 32-Real result now remains in `DevMemSpace` until
one explicit host mirror copy.

Fresh local validation at commit
`f29f6b0eb3c6045ee9d1ec506ba03ec792032462` passed the complete source-unit
suite (including the unchanged 4,320-sample all-61 oracle) and an evolved
closed-loop one-cycle test with every q-reduction fence reached.  Aurora PVC
qualification is intentionally still pending.

