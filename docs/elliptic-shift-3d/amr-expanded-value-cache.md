# Radius 0.24 scalar-log halo qualification

This extends the value-only coarse/fine interpolation diagnostic from radius
0.12 to 0.24 on the same actual N6, seven-shell, refinement-three, 47,088-cell
wormhole mesh. The earlier coupled radius-0.12 pilot exposed a second inherited
coarse/fine defect near radius 0.16949. The expanded diagnostic qualifies the
reconstruction and evaluates the **complete initial-domain Hamiltonian**. It
does not run a coupled evolution or modify shared implementation headers.

## Frozen variant and exact scope

The new runner is `tst/elliptic_shift_3d/run_amr_expanded_values.py`; its standalone
source is `test_amr_expanded_value_cache.cpp`. It copies all required project
sources into the evidence directory before compilation. In that frozen copy
only, it raises the helper's engineering row-cap validation from 10,000 to 20,000.
The constructor defaults, scalar basis, rank/reproduction guards, L1 cap,
stamps, donor search, field values, and physical equations are unchanged. The
two string substitutions and their exact diff/hash are recorded in
`frozen-row-budget.patch` and `manifest.json`. The shared helper, evolution
engine, native code, pilot binary and CMake targets were not edited or rebuilt.

The tested call is equivalent to:

```cpp
PunctureLogValuePlans(grid, puncture, .24, 20000, 8, 175);
```

At the frozen run revision the shared helper still had its 10,000-row
validation. The parent subsequently accepted this measured qualification for
an opt-in pilot and owns the coordinated budget/configuration integration.
No live header change was made by this diagnostic subtask. The runner now also
accepts a shared header already supporting 20,000 rows: it freezes that version
byte-identically and records an empty budget patch.

The selection is still Cartesian receiving halos with Cartesian active donors
crossing refinement levels. All selected receiving assignments are retained.
The existing first/second stencils do not read triple-logical-corner padding.
The radius 0.24 selection cannot reach the Cartesian/shell boundary at Cartesian
coordinate magnitude 0.25; shell treatment remains separate. The existing
rho/h<=8 derivative override is unchanged.

## Counts, guards and costs

The count-only process completed before construction:

| Quantity | Result |
|---|---:|
| All active cells |47,088 |
| Selected receiving assignments |36,864 |
| Unique physical-query/spacing plans |18,560 |
| Explicit unique-plan cap |20,000 |
| Supports using 125 / 150 / 175 / 256 donors |17,604 /952 /4 /0 |
| Recorded rejected rank attempts |960 |
| Maximum full-basis condition |9.673e5 |
| Maximum VALUE L1 / fixed cap |6.807 /8 |
| Maximum internal polynomial/log reproduction error |3.90e-13 |
| Reproduction error at every actual receiving target |2.98e-14 |
| Maximum duplicate-query coordinate difference |2.41e-17 |
| Deterministic 1e-12 sample-noise output |2.24e-12 |
| Change in any active field value |exactly 0 |

No support exceeds 175 donors, so a 256-donor extension is unnecessary. Rank
rejections are counted before bounded support enlargement; there is no lower
polynomial degree, ordinary-stencil fallback, assignment truncation or excluded
active cell. Sharing identical geometric queries retains every receiving
assignment and does not snap a puncture position.

Measured cache build time was 29.345 seconds; one paired positive-log fill took
14.7 milliseconds. The whole diagnostic process took 34.56 seconds, including the
unchanged 2,728-row derivative cache and full-domain H calculations. Count and
compile processes took 1.49 and 2.52 seconds respectively. Every process has an
independent 180-second limit and uses one host worker, with no remote allocation
or GPU job. The retained 2,344,000 donor-index/value-weight pairs require about
37.5MB of payload, plus row, assignment and stamp metadata.

Compared with the nested radius 0.12 selection, expansion adds 8,944 unique
plans and 17,616 receiving assignments. Additional support counts are 8,534
with 125 donors, 408 with 150, and 2 with 175. This adds 1,128,300 donor-index/
value-weight pairs, or 18.05MB of payload. Measured cold-build and fill increments
are 13.90 seconds and 6.79 milliseconds. The derived arithmetic and source
evidence hashes are retained in `support-budget.json`.

For scheduling only, two measured cache constructions cost about 58.7 seconds.
Combining that with the parent's prior approximately 94-second coupled workload
suggests an approximately 153-second run. This estimate is not a measured
end-to-end restart result; the actual coupled pilot must keep its 180-second
bound and record total time.

## Full initial-domain Hamiltonian

Every active cell is evaluated with the actual `EvolvedConstraints` tensor
kernel. The state uses the actual initial chi, radial curvature and Khat values
from `amr-core-pilot-20261027/initial.dat`, the actual flat conformal metric of
this initial slice, mapped scalar derivatives, and the unchanged existing core
derivative override. The lapse/reference gauge is not changed. The Hamiltonian
in this kernel cancels all shift/normal dependence, so no elliptic solve is
needed for this **initial H diagnostic**. Momentum, Theta/Z, scri closure and
coupled evolution are not qualified by this calculation.

An independently written flat-metric identity, including the full curvature
and physical-trace terms,

    H = chi[-6 + 2 Omega x·grad(log chi)
                   + Omega²(2 Laplacian(log chi) - |grad(log chi)|²/2)]
        + (2/3) Kphys² - (3/2) Omega² Arr²,

agrees with the tensor result at every active cell to 1.50e-14. This is an
algebraic diagnostic check, not a subtraction or replacement of the physical
source. The physical-volume weights are the original mapped volumes divided
by chi^(3/2), with all 47,088 cells retained.

| Region | Cells | Original H RMS | Expanded-cache H RMS | Original maximum | Expanded maximum |
|---|---:|---:|---:|---:|---:|
| Entire domain |47,088 |.02981103801 |.02961995674 |.40630529274 |.04782253672 |
| Cartesian bulk |38,016 |.03176702962 |.03156139479 |.40630529274 |.04782253672 |
| All shell cells |9,072 |.00823855327 |.00823855327 |.04362334066 |.04362334066 |
| r<.12 |18,368 |.03217222205 |.03202868791 |.40630529274 |.04782253672 |
| .12<=r<.24 |12,096 |.01526483942 |.00090916197 |.19570589794 |.00406029802 |
| r>=.24 |16,624 |.00793879107 |.00793879107 |.04362334066 |.04362334066 |

The three radius partitions sum to the complete domain. Both the shell and
r>=.24 diagnostics remain bitwise unchanged. At the reported next-interface
point 25482, the **static initial** H changes from .19570589793 to .00086453168.
The parent's approximately .19555 value was measured in the coupled evolution;
it is not presented as an identical-time comparison here.

The new global maximum is still point 34919 at
`(-.002604166666666668,-.002604166666666668,-.013020833333333332)`, radius
`.013531646934131853`. Its current active-only enriched jet is identical before
and after this value-only interpolation change. Thus the remaining near-core
maximum is inherited; it has not been regularized or resolved by extending the
interface cache. The existing shell maximum .043623 also remains. These raw
values do not establish the scientific convergence targets of the original
milestone.

## Reproduce and next gate

```sh
python3 tst/elliptic_shift_3d/run_amr_expanded_values.py \
  --build build-shift-3d \
  --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat \
  --radius .24 --maximum-rows 20000 --maximum-donors 175 \
  --output docs/elliptic-shift-3d/evidence/amr-expanded-value-cache-reproduction
```

The authoritative completed evidence is
`evidence/amr-expanded-value-cache-20261027/manifest.json` and `results.json`.
Build/count/diagnostic stdout and stderr are retained separately. Source HEAD
was `f4b7b702b9f97e76b5101ab8ab23339698154320`; the frozen source hashes identify
the exact dirty-worktree contents and local-only budget variant. Shared source,
static libraries and initial data stayed unchanged during the run.

The next gate is the parent-owned optional coupled radius 0.24 pilot, retaining
all original regions and acceptance tolerances, lapse/shift diagnostics,
physical constraints and checkpoint/restart coverage. Only the static field
interpolation and complete initial H calculation pass here. No native task,
MPI, device, late-time front or moving-puncture result is claimed.


## Completed full 3D evolution control

The optional engine integration now exposes a support radius and persists it in
research checkpointv14. Oldv13 files retain .12. The .24 pilot, with the same
47,088-cell mesh, sources, dt=.0001 and2steps, finishes in155.12s and restarts
bit-identically. All compiled dependencies, the frozen executable and initial
data remain unchanged during the run. All-cell H maximum is .0478254 and RMS
.0296200; M RMS is .0328553. Original-vector RMS maximum2.727e-11 and point
maximum1.248e-9 pass their unchanged controls, while the tensor scri pole
1.4418e-5 still fails1e-7. This is an implemented constraint-interpolation fix,
not completed radial reproduction or generic puncture safety.

The new maximum is point34919 at r=.01353165 inside the existing derivative
enrichment. All-cell beta error versus the matched independent N2049 radial
reference remains .000412883, and angular scatter remains. Exact run/evidence:
`evidence/amr-expanded-halo-coupled-20261027`; same-mesh three-control
comparisons and raw arrays: `evidence/halo-support-radial-comparison-final-20261027`.

```sh
python3 tst/elliptic_shift_3d/run_coupled_halo_support.py \
  --radius .24 --output build-shift-3d/expanded-halo-reproduced
python3 tst/elliptic_shift_3d/compare_halo_support.py \
  --output build-shift-3d/halo-comparison-reproduced
```

Each evolution command has one worker, a50,000-cell ceiling and180s process
limit including reconstruction/restart. A scientific failure exit from the
underlying pilot is preserved independently of completing the numerical run.
