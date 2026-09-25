# Stationary background cache

Caching defaults to **on** for explicitly time-independent analytic providers:
all fixed `z4c_tov_ks` backgrounds (Kerr-Schild, Schwarzschild puncture/trumpet,
Kerr trumpet, and flat space), and the flat background of the linear-wave test.
Set `problem/cache_stationary_background=false` to opt out. Direct Z4c and
ADM-derived stationary backgrounds both qualify. Existing geometry-specific
input restrictions still apply.

The time-dependent gauge-wave provider defaults to off and rejects an explicit
`true`. Unknown providers remain uncached until they explicitly enroll through
`ConfigureStationaryBackgroundCache(pin, time_independent)`. Non-analytic
backgrounds cannot enable this cache.
It reuses the existing stationary ADM and once-projected Z4c background arrays.
Evolving fields, RHS values, constraints and matter response are not cached or
reset. No additional volume-sized arrays or per-stage fences are introduced.

`RefreshBackground` invalidates the cache. `UpdateBackgroundState` returns on a
valid hit before the two provider kernels and projection. Avoiding repeated
projection preserves the original floating-point cancellation. Enrollment,
including restart, starts invalid. `MeshBlockPack::AddMeshBlocks` and
`AddCoordinates` invalidate on layout replacement, including unchanged block
counts and allocator reuse. Provider pointers, array addresses, local block count
and conformal representation are also checked. Future mutations of background
parameters must invalidate or re-enroll. Time-dependent providers explicitly enroll with `time_independent=false`.

## Reproducible regressions

Build an MPI `z4c_tov_ks` executable, then use Python with NumPy:

```sh
python3 tst/regression/z4c_stationary_cache.py \
  --exe /absolute/build/src/athena --output /absolute/new/cache-results
python3 tst/regression/z4c_stationary_cache_amr.py --default \
  --exe /absolute/build/src/athena --output /absolute/new/amr-results
```

The first compares all saved active/ghost spacetime and background bytes and
histories for six default/explicit-on/uncached triples: uniform and static-refined vacuum,
lapse pulse, and atmosphere with matter feedback. It checks exact zero,
nonzero response, cache hits on every rank and the existing puncture/direct-provider guard.
`z4c_cache_provider_defaults.py` checks five additional stationary provider
variants; with a gauge-wave executable and `--gauge-wave`, it checks opt-out
equality, evolving background snapshots, and rejection of forced caching. The
`--linear-wave` mode tests the stationary flat provider in `built_in_pgens`.
The second exercises actual vacuum refinement and MPI load balancing followed
by restart. It compares complete MHD, magnetic and Z4c checkpoint payloads
between cached and uncached runs, and separately audits finite payloads,
raw active/ghost metric positivity and exact-zero vacuum. It does not compare
an uninterrupted matter trajectory to a restarted one.

## Evidence and limits

September 2026 GPU benchmarks of the same cache implementation gave short warm
speedups of 5.33x for the ordinary-coordinate provider and 10.45x for an
experimental mapped provider. Four cache off/on full-payload pairs were
bit-identical, and 15 subsequent vacuum GPU lifecycle checks passed. These
measurements exclude startup/final I/O and queues and are not sustained TDE cost
promises. The mapped provider is not included in this integration.

Long perturbed vacuum tests still exhibit late gauge/constraint growth for the
configurations reviewed so far. Exact stationarity and cache equality do not
certify perturbation stability. Full matter restart continuity, long atmosphere,
full-star AMR and the 2500M spinning TDE remain unvalidated. Existing inputs now use caching for eligible providers when run with a newly
built executable. Previously pinned campaign binaries remain unchanged.
