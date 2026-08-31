# Perlmutter reference-motion hard-freeze discriminator

This bundle preserves the compact evidence from the bounded Ref-GH
continued-motion versus hard-freeze discriminator run on 2026-08-31.  It is
not evidence of stable or convergent trumpet evolution.

## Provenance and completion

- Source branch: `codex/ref-gh-relative-damped-single-hole-20260830`
- Source commit used for the GPU runs: `823f851d184185565fb4046927754810b551e24b`
- Perlmutter allocation: `57779143`, account `m3328_g`, QOS
  `gpu_interactive`
- Nodes: `nid001212`, `nid001213`, `nid001216`, `nid001217`
- Hardware: 16 distinct NVIDIA A100-SXM4-40GB GPUs, four per node
- Allocation result: `COMPLETED`, exit `0:0`, elapsed `01:36:09`
- Grid: `[-24M,24M]^3`, 32^3 active cells per MeshBlock, 328 MeshBlocks,
  three physical refinement levels, finest spacing `h=1/16M`
- Fresh common moving-reference seed: t=0 to t=2M, exit 0
- Continued-motion branch: t=2M to t=4.2M, exit 0
- Hard-freeze branch: t=2M to t=4.2M, exit 0
- All three runs report `bad-state=0` at their endpoints.

The seed and branches used all 16 ranks/GPUs.  The executable linked Cray's
CUDA-aware MPI transport (`libmpi_gtl_cuda.so.0`).  The authoritative mapping
is in `rank_gpu_mapping.txt`; earlier failed binding experiments are retained
separately and are not numerical evidence.

## Compact result

The common state at t=2M had GH/reduction/curl RMS norms
`1.0590e-4 / 1.7435e-5 / 3.0946e-4` with `xi=0.25` and
`xi_dot=0.125/M`.

| branch | first evolved t/M | first GH RMS | final GH RMS | final reduction RMS | final curl RMS | final xi | final xi_dot |
|---|---:|---:|---:|---:|---:|---:|---:|
| continued | 2.020653 | 1.0921e-4 | 4.1390e-3 | 9.8080e-4 | 1.2785e-2 | 0.525 | 0.125/M |
| hard freeze | 2.020653 | 1.9520e-2 | 1.2600e-2 | 2.9947e-4 | 2.7859e-3 | 0.25 | 0 |

The hard-freeze implementation zeros the runtime `reference_dt_frame` and
`reference_dt_connection` maxima exactly.  Its restart reprojection preserved
the represented spatial state to `2.22045e-15`, while the frame change altered
Pi by `1.00634`.  The abrupt reference/gauge stop produced an immediate GH
constraint jump, after which GH decayed with fitted log slope `-0.189/M` over
t=2.15--4.2M.  Reduction and curl still grew, with slopes `1.185/M` and
`0.958/M`, but ended 3.28 and 4.59 times below the moving continuation.

For the moving continuation, GH/reduction/curl fitted log slopes over the same
window are `1.612/M`, `1.747/M`, and `1.602/M`.  Its final constraint and RHS
maxima localize mainly around r=0.47--0.56M, the blend region.

This establishes that continued reference motion substantially amplifies the
growing reduction/curl and source sectors.  It does **not** establish that
ongoing motion is the sole cause: the frozen branch retains positive
reduction/curl growth, and the discontinuous hard stop contaminates its gauge
constraint.  A direct fixed-intermediate initialization, or a matched smooth
stop with no gauge impulse, is still required to distinguish an unstable fixed
operator from a transition-excited persistent mode.

## Artifact guide

- `reference_motion_freeze.json`: complete merged numerical summary
- `reference_motion_freeze_growth.tsv`: log-linear fits
- `reference_motion_freeze.png`: constraint/reference-motion history plot
- `seed_to_t2/`, `continued/`, `hard_freeze/`: compact histories and logs
- `source_oracle/source_unit.log`: CUDA source-unit and exact hard-freeze oracle
- `rank_gpu_mapping.txt`: authoritative 16-rank/16-GPU mapping
- `provenance.txt`: source, build, CMake, linkage, commands, and seed hash
- `gpu_final_snapshot.txt`, `slurm_job.txt`, `allocation_final.txt`: allocation evidence
- `restart_manifest.tsv`: selected common/final restart paths, sizes, and hashes
- `restart_sha256_manifest.txt`: raw `sha256sum` output for those checkpoints
- `cuda_build_attempt1.log`, `cuda_build_success_b5dad912.log`: CUDA portability evidence
- `seed_binding_none_incomplete/`, `seed_single_bind_failure/`: explicitly
  incomplete launch-binding attempts, retained only as diagnostic evidence

The remote analyzer initially failed because the Perlmutter Python environment
did not provide matplotlib.  The committed analyzer now treats plotting as an
optional postprocessing dependency, reports the true first evolved branch
sample rather than the common fork row, and was rerun locally to produce this
JSON/table/plot.
The branch `run_status.txt` files incorrectly report power-history time zero
because restart output retained the seed power-history basename.  The actual
power histories are present and were consumed by the analyzer.

## Large checkpoint locations

The large checkpoint files remain under:

`/pscratch/sd/h/hzhu/refgh-reference-motion-freeze-20260831.7Y5n8O/reference_motion_freeze_57779143`

Selected reproducibility checkpoints, sizes, and SHA-256 hashes are listed in
`restart_manifest.tsv`.  No restart or field dump is committed.
