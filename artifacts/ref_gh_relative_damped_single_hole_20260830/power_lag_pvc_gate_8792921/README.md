# Aurora paired-power PVC gate 8792921

Aurora job `8792921` built exact source commit
`9eff5b524c88cf3c2adf8fc4c219fb9c2e72ed29` on one compute node and ran one
fresh, frozen, compatible-Phi RK4 cycle on the production 328-MeshBlock SMR
tree.  Twelve MPI ranks mapped to twelve distinct PVC tiles.  The job exited
zero after `00:20:40`; its evolved endpoint was `t=0.0041383110461M`.

Every power shell was valid.  At `t=0`, all paired `q_phys-q_ref` means were
at or below `1.22e-16` in magnitude.  At the evolved endpoint they were finite
and of order `1e-8`; `xi=xi_dot=xi_ddot=0` remained exact.  The full-output
history and debug-fence cycle contained no fatal, SYCL, page-fault, or
nonfinite signature.  This qualifies only the new diagnostic's PVC execution,
not stability or causation.

The 1.3 MB build log is intentionally not committed.  It remains at the
remote directory below with SHA-256
`40698dcf0c27f0c0adefda0f2daab897097b553bb06d101f04b53d0489f78bd0`:

```text
/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_relative_damped_power_lag_20260831_9eff5b52/power_gate_8792921.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
```

`compact_sha256.txt` verifies every committed numerical/provenance file copied
from Aurora.
