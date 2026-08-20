# Fresh N256 Brill scratch execution

This is the user-authorized from-scratch replacement for the unavailable
authenticated cycle-1721 restart. It starts from the exact archived
direct-global-coefficient Brill data and preserves all numerical/evolution
parameters through the physical target-event window:

- root grid `128 x 256 x 1`, half-plane Cartoon; `32 x 32 x 1` MeshBlocks;
- O6, RK4, CFL `0.15`, KO `0.02`, dynamic dchi AMR with `dchi_max=0.01`;
- telegraph lapse (`tau=kappa=1`, max-domain-abs-K scaling), Gamma-driver shift,
  and zero constraint damping;
- `A=-0.047`, ADM mass `2.660301967997158`, direct coefficients SHA-256
  `ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b`;
- pre-collapsed `psi^-2` lapse.

Only the following non-physical controls differ from the archived long input:

1. The first segment uses `nlim=1800` and `tlim=10.0` as a safety bound.  The
   exact cycle-1800 restart is continued with `nlim=-1`, `tlim=10.0`; this
   changes only the stopping/output controls.
2. Default-off chi-parent provenance starts at `t=9.49` and writes a separate
   diagnostic output.
3. The AMR-jump output basename is unique to this run.

The production transfer, gauge, damping, KO, AMR criterion, resolution,
MeshBlock layout, and direct initial data are otherwise unchanged.  The fresh
trajectory does not reproduce the old event at cycle 1722 because the corrected
timestep contract changes cycle numbering.  It does reproduce the exact
physical parent pair at cycle 2833, `t=9.476710063617325 M`, and terminates
normally at `t=10 M`.

`aurora_head_build.sh` builds the exact Iris 2a069fd dependency and current
AthenaK source on the login node with `--parallel 64`. `aurora_run.pbs` launches
one rank on PVC tile 0.0 from a fresh output root, and
`aurora_continue_physical_window.pbs` continues the authenticated segment to
the physical bound.  All scripts fail on reused roots, wrong payload checksums,
or incompatible source/build state.  The results are under
[`fresh_n256_physical_run_20260820/`](fresh_n256_physical_run_20260820/).
