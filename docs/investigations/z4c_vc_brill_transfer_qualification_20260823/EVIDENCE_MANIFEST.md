# Evidence manifest

This inventory separates source authority, immutable remote evidence, copied compact evidence, and local post-processing. The original Perlmutter `SHA256SUMS` files retain their absolute paths. `verify_rebased_artifacts.py` changes only the path prefix during local verification; every copied payload hash matches, as recorded in `artifacts/rebased_manifest_verification.log`.

## Source authority

| Item | Authority |
|---|---|
| Repository | `https://github.com/HengruiZhu99/athenak` |
| Branch | `codex/z4c-vc-brill-transfer-qualification-20260823` |
| Exact base | `2d59f85c11cb0da4614c84a695d64f032fb9eec7` |
| Source/test HEAD used for final runs | `278b63a740a947de55ad8bdd1c333095c68fedcd` |
| Kokkos commit | `6739bc623081648af9e752b616d9671527922cbf` |
| Phase-6 input SHA256 | `e574cb2731c581193ddfc905d51a0cdb83c996c4dc6f03e0556056f1109a22db` |

## IrisK authority

| Item | SHA256/value |
|---|---|
| AthenaK handoff header | `23bc2187c29ccb2695a54fc5c59e08a2e7b9d3389a63c1081cf953a507fb0cdb` |
| Static interpolator library | `d4afad6d3a20a8dd8197eb7d70d5a23903a7e2401a5d8b034d32005bf07f3f39` |
| Global coefficient file | `ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b` |
| Brill amplitude | `-0.047` |
| ADM mass | `2.660301967997158` |
| Stored spectral Hamiltonian residual | `6.9517373601955137e-13` |

## Perlmutter remote root

```text
/pscratch/sd/h/hzhu/z4c-vc-brill-transfer-qualification-20260823
```

Raw run products and restarts remain under the remote root. Compact logs, summaries, CSVs, provenance, and original manifests are copied into this repository.

## Copied immutable evidence

| Phase/event | Job | Local directory | Remote event directory | SHA256 of original `SHA256SUMS` | Disposition |
|---|---:|---|---|---|---|
| Phase 0 host authority retry | 57441616 | `artifacts/perlmutter_phase0` | `evidence/phase0-host-retry` | `573a807c64f57a51417ae181aa23eae351c1cc4612b59fc59cd0918f34700481` | 123/123 portable tests; compiler-bound literal fingerprint isolated |
| Phase 3 CUDA controls | 57443278 | `artifacts/perlmutter_phase3_cuda` | `evidence/phase3-current-cuda` | `ace9213b351c43c681fa356d811946add94c8da79dc93f2438f59417efc2ca1e` | passed |
| Phase 4 CUDA/MPI clean retry | 57446241 | `artifacts/perlmutter_phase4_current_cuda_mpi` | `evidence/phase4-current-cuda-mpi-retry2` | `90982dcee8eb69432aea841a5a787a504b09d40d95bf3be7e6417f913409d15a` | passed |
| Phase 5 initial run / failed first analysis assertion | 57447013 | `artifacts/perlmutter_phase5_brill_initial_failed` | `evidence/phase5-brill-initial` | `e815e3c2c848a618dde6f3586e8e995a3b3693fb8df0e5b0535e04043c8c8b1a` | science arrays valid; assertion incorrectly expected discrete Gamma to be resolution-identical |
| Phase 5 corrected existing-data analysis | none | `artifacts/perlmutter_phase5_brill_initial` | `evidence/phase5-brill-initial-analysis-recovery1` | `e5f232c672a088911ed5f4f967a7b9b52eaa0b168bba65e137aeab601a6c46e5` | passed without rerun |
| Phase 5 common-node analysis | none | `artifacts/perlmutter_phase5_brill_initial_common_node` | `evidence/phase5-brill-initial-analysis-recovery2` | `064cae1b89d8874aac8e6826ea96fff2dbeeca531147b9840d4c03d2478f43ea` | completed without rerun |
| Phase 6 failed harness attempt | 57447520 | `artifacts/perlmutter_phase6-fixed-brill` | `evidence/phase6-fixed-brill` | `fb37561e71438578dc47ff28691d374ec9670a0eaa8e7565da1851e4f3b7e090` | inherited `nlim=0`; cycle-zero stop; no evolution evidence |
| Phase 6 N128/N256/N512 fixed-grid retry | 57447597 | `artifacts/perlmutter_phase6-fixed-brill-retry1` | `evidence/phase6-fixed-brill-retry1` | `ba235137a57046a87fa6210d990a580f6a1b3a12f68930e8721e968a46059ebd` | all reached `tau_c>=3 M` |
| Phase 6 first analysis | none | `artifacts/perlmutter_phase6-fixed-brill-analysis` | `evidence/phase6-fixed-brill-analysis` | `b5ca5158fc4add2588f30ccfc074ce38a32c3efb555affd0e53ae9c0bab4d583` | completed |
| Phase 6 terminal restart sampler | 57447771 | `artifacts/perlmutter_phase6-fixed-brill-terminal-rhs` | `evidence/phase6-fixed-brill-terminal-rhs` | `7462bb5084293308c7de768f390bc340552ccf22b1271717b703eb0fb361074a` | one-stage diagnostic capture; no long evolution |
| Phase 6 regional terminal analysis | none | `artifacts/perlmutter_phase6-fixed-brill-terminal-rhs-analysis` | `evidence/phase6-fixed-brill-terminal-rhs-analysis` | `559a6b6f3de85b27305f8ae3e512482f1e363132f583d05d5b2ea0c74f74b373` | completed |

## Local deterministic analysis

| Artifact | Purpose |
|---|---|
| `analyze_vertex_transfer.py` | exact weights, Fourier/image response, repeated cycles, nonlinear profiles |
| `run_semidiscrete_interface.py` | production RHS comparison on static refined and level-matched uniform hierarchies |
| `plot_semidiscrete_interface.py` | Phase-2 plot |
| `analyze_brill_initial_data.py` | direct VC field, algebraic, Gamma, and constraint analysis |
| `analyze_fixed_brill.py` | common-central-proper-time history and terminal field differences |
| `analyze_fixed_brill_terminal_rhs.py` | role/region-resolved terminal state, RHS, and constraints |
| `plot_qualification_summary.py` | committed Phase-6 summary plots |
| `verify_rebased_artifacts.py` | local verification of original absolute-path manifests |

## Failure and qualification boundaries

- The Phase-6 initial `nlim=0` attempt is a harness failure, not a numerical result.
- The Phase-5 first analyzer failure is a corrected analysis-assumption failure; no science arrays were rerun.
- Historical Aurora data are not current-source backend qualification.
- The fixed-grid failure blocks common-tree, native-AMR, strong-field outcome, and performance claims.
- Exact shared-node or restart agreement is not a convergence result.
- No Figure-3 reproduction, threshold, horizon, critical exponent, or DSS claim is supported.
