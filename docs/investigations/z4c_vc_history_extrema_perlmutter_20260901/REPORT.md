# Brill amplitude history-extrema reruns

## Verdict

The new slice-global history diagnostics are present and internally
consistent, and the existing data are sufficient for **partial comparison
plots**.  They do not constitute two completed, matched evolutions.

- The plotted `A=-0.047` native-AMR segment reaches
  `t=24.52458523488176 M`.  It then fails closed because the run wrapper
  incorrectly set `max_nmb_per_rank=256`; the next refinement required 260
  MeshBlocks.  This is an operational configuration limit, not a physical
  endpoint.
- The plotted `A=-0.050` native-AMR history, including its restart
  continuation, reaches `t=36.458530270670735 M`.  The continuation ends on
  its internal wall-clock limit before the requested `t=38.652331986867424 M`.
- A subsequent matched `A=-0.047` retry used the correct 4096-block limit and
  the same horizon settings as `A=-0.050`, but remained in setup after initial
  data import.  It produced only the `t=0` history row and was intentionally
  cancelled.  It is not plotted.

No convergence, completed Figure-3 reproduction, horizon formation, or
collapse/dispersal classification is claimed.

## Diagnostics

`src/outputs/history.cpp` reports:

- `maxAbsKret`: slice-global maximum of the absolute Kretschmann scalar when
  `history_kretschmann=true`;
- `minLapse`: slice-global minimum lapse over active evolution points.

For vertex-centred data, duplicated vertices are excluded with the canonical
diagnostic-ownership mask.  A nonfinite lapse contributes negative infinity,
so the diagnostic fails visibly rather than hiding invalid data.

The Perlmutter CUDA smoke gate completed and emitted both history labels.
The plotting reader additionally verifies every row is finite and monotone in
coordinate time, `minLapse <= axisLapse`, and
`maxAbsKret >= abs(axisKret)`.

## Existing plotted evidence

| Quantity | `A=-0.047` partial | `A=-0.050` partial |
|---|---:|---:|
| Final coordinate time | 24.524585 M | 36.458530 M |
| Final central proper time | 10.174928 M | 11.209853 M |
| Minimum slice lapse | 0.1034973 | 0.06096772 |
| Time of minimum slice lapse | 21.406875 M | 35.941276 M |
| Peak origin `abs(Kretschmann)` | 6.922216e3 | 1.577409e5 |
| Peak slice `abs(Kretschmann)` | 6.053563e4 | 5.544177e6 |

Both curvature maxima occur at the final retained sample, so neither curve
has demonstrated a resolved terminal peak.  The `A=-0.050` lapse reaches a
smaller value and its curvature grows to a larger value over its longer
available interval.  This is an observation, not by itself proof of black-hole
formation.

## Plots

- `analysis_partial_dynamic/central_kretschmann_figure3_style.{png,pdf}`:
  origin curvature against central proper time, with the published `A=-0.047`
  Figure-3 curves.
- `analysis_partial_dynamic/slice_max_kretschmann_coordinate_time.{png,pdf}`:
  slice-global maximum curvature against coordinate time.
- `analysis_partial_dynamic/slice_min_lapse_coordinate_time.{png,pdf}`:
  slice-global minimum lapse against coordinate time.
- `analysis_partial_dynamic/history_extrema_three_panel.{png,pdf}`:
  compact combined view.

Curvature is shown in raw code units; no ADM-mass normalization is applied.
Published reference curves correspond only to `A=-0.047`.

## Run disposition

| Job | Disposition |
|---|---|
| 57824147 | Cancelled build allocation; no science evolution. |
| 57826586 | Smoke passed; replayed `A=-0.047` failed source-ID authentication; first `A=-0.050` segment reached `t=30.92551 M`. |
| 57829349 | Partial native-AMR `A=-0.047`; stopped at the 256-block fail-closed gate and later cancelled during evidence hashing. |
| 57830474 | Clean `A=-0.050` restart continuation; reached `t=36.45853 M`. |
| 57833764 | Cancelled after 18 seconds when the setup-equivalence audit found mismatched horizon settings; no usable science data. |
| 57833859 | Correctly matched `A=-0.047` retry; setup/I/O stall, only `t=0`; intentionally cancelled. |

At handoff all allocations were revoked.  Slurm briefly retained job 57833859
in `COMPLETING`; no evolution process remained active.

## Validation and limitations

- Fresh CUDA executable SHA-256:
  `c3ef1c8b371eb3a447108d3b0acc115b34cfeaeb71337fda277e18d409a5b8c0`.
- `python3 tst/unit/z4c/z4c_history_deterministic_reduction_static_test.py --source-root .`: pass.
- Plot script byte-compilation and history invariants: pass.
- The domain-specific `z4c_cartoon_history_test.py` was not used as a
  production-history validator because its coordinate bounds are hard-coded
  for the small `rho,z in [0,8]x[-8,8]` test problem.
- The plotted `A=-0.047` run did not enable the horizon finder and had the
  erroneous MeshBlock cap.  Its PDE/gauge/AMR criteria otherwise match the
  native `A=-0.050` setup through its retained interval.
- The curves have unequal and incomplete endpoints.

