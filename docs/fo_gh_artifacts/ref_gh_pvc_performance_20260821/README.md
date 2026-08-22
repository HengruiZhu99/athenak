# Compact Aurora PVC performance evidence

This directory contains the compact evidence for the final
`docs/ref_gh_pvc_performance_convergence_20260821.md` report and its earlier
performance checkpoint. The directory names encode the PBS job that produced
each record. Large field outputs, restart files, executables, and the compiled
profiler library remain only on Aurora.

- `baseline_8773102`: checked and bounds-off Ref-GH build/configuration,
  provenance, device mapping, and throughput.
- `z4c_8773273`: matched mature-Z4c run provenance, device mapping, and the
  completed measured-run log. The copied `throughput.tsv` is the inherited
  Ref-GH baseline table; the Z4c rate is in `z4c_performance.log` because the
  PBS postprocessor failed after the numerical run.
- `profile_8773335`: synchronized baseline kernel profile and hashes.
- `traversal_8773363`: checked gate, throughput, and the measured neutral/slight
  regression from the traversal reorder.
- `static_cache_8773414`: checked stationary and time-dependent gates,
  throughput/speedup, synchronized profiles, hashes, and provenance.
- `diagnostic_split_8773745_8774035`: corrected static, checked, and
  production-only reference-Ricci call-count evidence.
- `symmetry_rejected_8774093_8774143`: the two failed checked-oracle results
  that rejected workspace symmetry compression without weakening tolerances.
- `lean_source_8774182`: the passing full-source oracle, checked evolved
  gate, histories, time-dependent gate, throughput, profile, executable hashes,
  and provenance for the lean production scalar source.
- `fused_source_pi_8774435`: source/Pi fusion oracle, checked/full-output gate,
  profile, compiler pressure, and the best adjacent one-tile performance run.
- `robust_final_8774480`: final hardened-source checked and production gates,
  lower-dimensional robustness evidence, profile, and conservative performance
  authority.
- `perturbed_misaligned_8774541`: rejected cross-resolution comparisons whose
  nominal output times did not match exactly.
- `perturbed_exact_binary64_8774567`: accepted exact-time binary64 64/96/128
  convergence evidence at t=0.2, 0.4, and 0.6.
- `stationary_t20_8774589`: completed 64/96/128 t=20 histories, final summaries,
  secular/roundoff fits, fixed-mask common-ADM tables, GPU mapping, provenance,
  and hashes for the remote binary64 dumps.

The complete remote evidence is retained under
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_performance_convergence_20260821/runs`
on Aurora. `SHA256SUMS` covers every committed file in this directory except
itself.
