# Optional supported58 wrapper dispatch plan

Status: the optional engine dispatch and checkpointv15 are implemented and
coupled-tested. It is an experimental static-centered radial-first option,
never a default change or a claim of generic moving-puncture robustness. See
[the scientific gates and five translated hard failures](rho-log-robustness.md).

## Prepared component and contract

`src/z4c/hyperboloidal/supported_puncture_log_plans.hpp` owns only supported58
rows, selected donor IDs, full mesh descriptor and exact active-index snapshot.
It uses the original rho<radius and rho/h<=8 query selection and identical
nearest125 prefixes. It never calls the old derivative evaluator and retains no
baseline reconstruction plans. Every field sample preserves the actual query
value; it does not return the fitted value or an inferred power.

Before a sample, guards compare n/ng/padded/count/order, mesh cells/ghosts/face
subdivision/half width/radii/revision, every block face/layer/refinement/bounds,
active-index count **and contents**, and geometry-array shape. Geometry and
Jacobian arrays are immutable derived arrays under that descriptor contract;
arbitrary mutation of derived array contents without changing its descriptor is
outside the adapter contract. Full puncture position/velocity/time/stage/revision
must match the plan stamp. Bad field/components/state shape, nonfinite query or
donor data, and unsupported geometry throw before returning output. A temporary
grid cannot construct an adapter. A new position/topology requires rebuilding.

The first prototype retained baseline plans and discarded the result of old
Sample; its frozen evidence is retained, but this design was replaced. A first
lightweight test build failed because its test-only reference comparison lacked
an explicit baseline include; that attempt is also retained, then corrected.

The corrected lightweight contract on the actual47088-cell mesh reports:

- 2728 complete rows, eight enlarged to175, all other rows125; original query IDs
 and nearest125 prefixes exactly match a temporary test-only old wrapper.
- 32736 checks (six arbitrary log powers, two fields, every row); max h-scaled
 derivative error8.6844e-14, active values bitwise retained, input unchanged.
- 31 rejection contracts: all puncture stamps, logical fields, descriptor fields,
 block bounds/topology/count, active same-size reorder and shape, geometry shape,
 field shape, and nonfinite data. Constant derivatives are bitwise zero.
- Supported constructor4.264s, grid.883s, temporary baseline reference build3.818s.
 A5456-call two-field sample sweep takes.03480s on the same bounded Serial CPU.
- Exact scalar weight payload30043200bytes, donor-index payload2731200bytes,
 active-index snapshot188352bytes. These exclude allocator capacities and
 row/map/descriptor bookkeeping. Baseline30008000weight bytes are no longer
 retained; QR workspace is local to Build and not cached. Zero baseline Sample
 calls. No process-wide peak-RSS claim is made from payload counts.

The evidence is
[evidence/supported-puncture-adapter-light-fixed-20261027/manifest.json](evidence/supported-puncture-adapter-light-fixed-20261027/manifest.json).
Reproduce with a fresh output directory:

```sh
python3 tst/elliptic_shift_3d/run_rho_log_diagnostic.py --build build-shift-3d --adapter --output /tmp/supported-adapter
```

## Implemented engine and lifecycle dispatch

`CompositeTensorEvolution` retains the original `PunctureLogPlans` for default
scheme0. Optional trailing constructor argument `puncture_scheme=1` constructs
only `SupportedPunctureLogPlans`; no baseline plans or baseline Sample calls are
retained on that path. It requires the enriched positive-storage2 wormhole and
regular tensor scri. Both physical chi and lapse jets use the same fixed geometry
plans with their independently evolved stored logs; all other fields, beta,
physical equations and original vector residual are unchanged. Active values
are preserved. Unsupported rows throw; there is no lower-degree fallback.

Checkpointv15 appends this scheme, old versions infer0, and restart rebuilds
plans. CLIargv30 selects it explicitly. Tests cover version14/13 decoding,
checksummed bad metadata and incompatible modes. The engine test reuses one
set of plans across chi/lapse powers4/2 and2/1.09, including rho and polynomial
angular terms. Its224 independent AD physical-jet checks give max h-scaled
error1.72701e-16, with no active-value mutation. The previously frozen default
Minkowski volume and angular CSVs remain bitwise identical. This is a
representation check, not a demonstration of collapse or moving safety.

## Actual coupled result

The same47088-cell N6/7-shell/AMR3 t=.0002 pilot with .24 value halos completed
in159.03s under its180s bound. The fixed58-column basis uses2728 rows, eight
enlarged to175, max dimensionless gradient/Hessian L1=17.2997/63.0194 below the
unchanged32/64 guards. All-cell H RMS falls from.02961997 to.002984706; Hmax
falls from.0478254 to.0435195 and is now dominated outside the enriched core.
M RMS remains.03285528. Original vector RMSmax2.72400e-11 and pointmax1.24781e-9
pass the predeclared solve controls. Complete state/shift/scri restart is bitwise.
The unchanged tensor scri pole1.44183e-5 **fails** its1e-7 target. Process exit1
records scientific failure despite successful execution/restart.

Evidence `evidence/supported-puncture-coupled-20261027` freezes the executable,
actual compiled dependencies, initializer, every solve, all cells and64 angular
rays. Its manifest confirms source/binary/initializer stability. Reproduce with
fresh paths:

```sh
python3 tst/elliptic_shift_3d/run_coupled_halo_support.py --output /tmp/new-supported-pilot --radius .24 --puncture-scheme 1
```

All accepted-state norms retain every cell. This short test improves one spatial
error mechanism but does not reproduce late lapse fronts, qualify a universal
regularity exponent, or repair the five unsupported translated geometries.
