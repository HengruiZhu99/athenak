# Brill collapse shift-transport controls

## Scope and provenance

This is a bounded control campaign on branch
`codex/brill-zero-shift-advection-controls-20260818`, source commit
`1c95db8a2adc743672b49a525c21c4f762f35223` (tree
`5f343d0e19bc47fa5cfcf199c342885fde14154b`). The runs used separate
one-GPU `gpu_shared_interactive` allocations; no ordinary interactive
allocation was used for these experiments. Kokkos is pinned to
`6739bc623081648af9e752b616d9671527922cbf`.

The history norm was audited before interpreting the jumps. Cartoon history
reductions use the axisymmetric ring measure, not a fictitious collapsed-y
Cartesian slab factor. There is therefore no evidence that these results are
an artifact of an omitted or extra `dy` normalization.

## Controls

* **Z1 (`arm_zero_shift`)**: prescribed zero shift, exact beta invariant,
  O6 transport where applicable, replay of the authenticated N128 hierarchy.
* **U2-short (`arm_gamma_o2_short`)**: Gamma-driver shift with only the
  genuine `beta^j d_j` transport dispatch changed to O2 upwind. Geometric
  beta derivatives and gauge sources remain O6/unchanged.
* **Z2-native (`arm_zero_shift_native`)**: prescribed zero shift with native
  dynamic AMR (`amr_history_mode=off`), bounded to one 35-minute
  `gpu_shared_interactive` allocation.

## Observations

| run | result | last/terminal time (M) | C history norm | dt (M) | max level | MeshBlocks |
|---|---|---:|---:|---:|---:|---:|
| Z1 replay | fail-closed at pre-update chi diagnostic, cycle 1770 | 3.97265625 | 1.3758e6 | 2.34375e-3 | 4 recorded (level 3 near failure) | 74 |
| U2-short | strict chi parent gate, cycle 5037 | 10.23750 | 2.4876e11 | 5.859375e-4 | 5 | 98 |
| Z2 native | bounded wall-time stop, no strict gate | 2.45273323 | 1.1238e9 | 5.722046e-7 | 15 | 962 |

### Z1

The zero-shift invariant passed. At cycle 1770 the pre-update RK-stage
diagnostic found chi = `-0.18994101508179184` at GID 44, approximately
`(rho,z)=(5.2578125,1.9765625)`, while beta transport was exactly zero. The
constraint norm had already reached approximately `1.3758e6`. The replay
tree was under-resolved relative to the native shadow AMR requests near the
failure (raw dchi requests of order `1820`--`2105`, while the replay authority
had no event until approximately `t=8.247M`). Thus Z1 is a prescribed-shift
control, not a fair native-AMR survival or convergence comparison.

### U2-short

U2 reached `t=10.2375M`, beyond the Z1 endpoint, but before the authenticated
Gamma/O6 reference crossing near `t=10.5357421875M`. It stopped at the strict
chi parent-stencil gate with 240 invalid parent stencils (first reported near
GID 33, level 5). Its final history values were approximately
`C=2.49e11`, `H=5.13e9`, `M=2.44e11`, and `max|K|=1.71e3`. O2 transport did
not avoid the failure. The conditional U2-full run was therefore not
authorized and was not executed.

The separately requested dissipation-0.5 attempt is not part of this
campaign: job `57102293` was revoked before allocation after its interactive
connection timed out. It used no GPU and produced no science data.

### Z2-native

The native zero-shift run repeatedly refined/derefined, reached level 15 and
962 MeshBlocks, and collapsed to `dt=5.722046e-7M`. At the bounded stop it
had `C=1.12e9`, `H=8.69e8`, `M=2.55e8`, and `max|K|=3.55e2`. It did not
reach `t=10.60M` or the strict chi gate. This is an expensive native-AMR
stress result, not a completed comparison run.

## Interpretation

**Observation:** Z1 fails early on a replay tree that is demonstrably too
coarse for the native requests; U2-short fails later but still before the
reference crossing; Z2-native is dominated by repeated native AMR and
timestep collapse.

**Inference:** changing the explicit shift transport from Gamma/O6 to
Gamma/O2 is not, by itself, a rescue. Native dynamic AMR is strongly coupled
to the runaway and is not represented by the Z1 replay tree.

**Hypothesis (not established):** the remaining failure is an AMR/coarse-
fine or high-frequency mechanism interacting with the post-RK chi state. The
existing diagnostic evidence is insufficient to distinguish active RK
instability from transfer amplification, and these three controls do not
provide convergence evidence.

## Artifacts

* `arm_zero_shift_history.csv`, `arm_gamma_o2_short_history.csv`,
  `arm_zero_shift_native_history.csv`: normalized history tables.
* `comparison_common_time.csv`, `comparison_proper_time.csv`: bounded
  comparisons by coordinate time.
* `constraints_history_comparison.png`, `topology_timestep_comparison.png`:
  compact history plots.
* `anomaly_causal_trace.csv`, `first_high_k_anomaly.json`,
  `advection_operator_audit.csv`: diagnostic/inference boundaries.
* `evidence_manifest.json`, `SHA256SUMS`, `SHA256SUMS.sha256`: checksums for
  the report package and selected text evidence.

The large restart and binary field dumps are intentionally not copied into
the commit; their authenticated remote paths and terminal hashes remain in
the evidence manifest. No Figure-3 reproduction, convergence, or physical
critical-behavior claim is made.
