# Compact Perlmutter fixed-reference discriminator evidence

This bundle supports the bounded report
`docs/ref_gh_fixed_reference_discriminator_perlmutter_20260831.md`.

- `source_oracle/source_unit.log`: complete A100 source-unit run, including the
  64-sample direct-fixed projection and smooth-stop reference-jet oracles.
- `direct_fixed/` and `smooth_stop/`: compact histories, max-location records,
  run transcripts, status snapshots, and restart path/size inventories.
- `fixed_reference_discriminator.json`: complete machine-readable endpoint,
  localization, shell-power, and growth-fit analysis, including the earlier
  continued-motion and hard-freeze cases.
- `fixed_reference_discriminator_growth.tsv`: fit windows, slopes, e-folding
  times, and R2 values.
- `scientific_status.txt`: compact claim boundary and endpoint values.
- `rank_gpu_mapping.txt`, `provenance.txt`, `configure.log`, `build.log`, and
  `slurm_job_final.txt`: allocation, device, source, build, and scheduler
  evidence.
- `compact_sha256.txt`: hashes of the original collected compact evidence.

Two raw records require care.  The GPU-run source printed zero initial
constraint fields before the output task had made them authoritative; use the
t=0 history values instead.  A bookkeeping-only source cleanup removes those
fields after the run.  Also, `smooth_stop/run_status.txt` reports a zero latest
power-history time because the restart retained the seed basename.  The file
`smooth_stop/refgh_reference_motion_seed.ref_gh_power.hst` reaches t=5.2M and
is the input used by the committed analyzer.  The harness now searches by
history suffix.

Large restart files are intentionally absent.  They remain at the Perlmutter
locations recorded in the report and the two `restart_sizes.tsv` files.

Claim boundary: this is one-resolution causal-discriminator evidence.  It does
not establish a stable or convergent trumpet.
