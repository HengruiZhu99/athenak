# Fresh N256 Brill scratch continuation

This is the user-authorized replacement for the unavailable authenticated
cycle-1721 restart. It starts from the exact archived direct-global-coefficient
Brill data and preserves all numerical/evolution parameters through the
cycle-1722 target event:

- root grid `128 x 256 x 1`, half-plane Cartoon; `32 x 32 x 1` MeshBlocks;
- O6, RK4, CFL `0.15`, KO `0.02`, dynamic dchi AMR with `dchi_max=0.01`;
- telegraph lapse (`tau=kappa=1`, max-domain-abs-K scaling), Gamma-driver shift,
  and zero constraint damping;
- `A=-0.047`, ADM mass `2.660301967997158`, direct coefficients SHA-256
  `ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b`;
- pre-collapsed `psi^-2` lapse.

Only the following non-physical controls differ from the archived long input:

1. `nlim=1800` and `tlim=10.0` bound a scratch run around the historical
   cycle-1722/time-9.50625 target.
2. Default-off chi-parent provenance starts at `t=9.49` and writes a separate
   diagnostic output.
3. The AMR-jump output basename is unique to this run.

The production transfer, gauge, damping, KO, AMR criterion, resolution,
MeshBlock layout, and direct initial data are otherwise unchanged. The target
diagnostic stops normally at target cycle plus eight accepted cycles. If the
fresh run does not encounter the same transaction before the explicit bound,
that is a result to report rather than a reason to extend or alter it.

`aurora_build.pbs` first builds the exact Iris 2a069fd dependency and the
current AthenaK source for one PVC tile. `aurora_run.pbs` then launches one
rank on tile 0.0 from a fresh output root. Both scripts fail on reused roots,
wrong payload checksums, or incompatible source/build state.
