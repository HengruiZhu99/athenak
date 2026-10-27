# Frozen completed spherical collapse snapshots

The read-only source is the completed N4097 trajectory
`telegraph-scri-pchip-convergence-v1/n4097`, not a later floor/attenuation
candidate. Exactly three original NPZ snapshots are copied under
`evidence/radial-collapse-snapshots-20261027`: `profile-0021.npz` at t1,
`profile-0101.npz` at t5, and `profile-0201.npz` at t10. Their actual stored
times and steps are1/81920,5/409600,10/819200. `provenance.json` records original
paths, byte sizes, exact SHA256s, arguments, initial-data metadata and the
source/library binding. All115 frozen source/library files matched that
binding. No original file or job was changed.

The selected source parameters are M=1/3, isotropic-map k=.37754694621653295,
Minkowski reference, physical-log/multiplicative/collapsed-inner lapse, nu1,
Appendix-C Theta, constant kappa1=4/kappa2=-.99, flux kappa=.005/tau=.02, and
AF-regular flux support. The **actual completed input uses taper width .24**,
outer radius .8: its window is1 through r=.56 and vanishes by .8. Width .08
would be a different continuum configuration and must not be silently used
for this comparison. Source differentiation/quadrature width7, upwind
transport, local mapped KO order8/strength.5/radius.8, inner-exp-scri map with
stretch16, and CFL.05 are retained as source provenance. They are not assumed
identical to the new 3D discretization. This historical source predates the
later floor/attenuation candidates; those new defaults must not be inherited.

The original trajectory reports `finished`, t10,819200 steps, failure null.
The hashed spatial acceptance report has18/18 listed checks passing, including
finest H RMS1.52811e-8, scri pole4.21225e-9, clock0 and null4.44089e-16. Its own
overall `goal_status` is **incomplete**, with independent timestep,
quadrature, dissipation and adjacent-parameter controls listed as missing.
These snapshots do not certify all intermediate extrema, long-time
robustness, or the requested 3D milestone. The acceptance review/protocol
hashes were independently checked and their exact check values/statuses are
retained in the new provenance record.

## Positive physical reconstruction and measured regimes

The archived q array is the historical positive coefficient storage, not an
array of physical fields. Reading physical chi as `1+q_physical[:,0]` or lapse
as `Aref+q_physical[:,6]` loses the inner values through cancellation. The
exporter uses the **recorded historical factors only to deserialize** the
snapshot, then exports physical fields. It does not impose those powers on a
new 3D representation or evolution. The decoded chi/alpha agree with saved
physical diagnostic columns to2.3e-16 absolute.

The r=0 array slot is an independent AF coefficient endpoint, not an ordinary
physical point. The minimum positive radius is4.4131220689702443e-10. Every
positive-radius node, including scri, has positive finite chi/alpha:

| Time | Minimum chi | Minimum conformal alpha |
|---:|---:|---:|
|1|1.55443e-33|2.77132e-17|
|5|1.09963e-31|1.96048e-15|
|10|8.36184e-25|1.49131e-8|

`audit.json` retains log-slope fits over fixed physical radius bands, their
node counts, fit residuals and adjacent-slope ranges. They are diagnostics,
not imposed powers or a continuum endpoint theorem. At t10 the innermost
4.4e-10..5e-9 band still fits chi3.997 and alpha1.999, whereas r1e-4..1e-3 fits
chi1.992 and alpha.696. Across r1e-6..1e-5 the lapse fit is negative(-.959),
spanning a front/flux feature. A universal fitted lapse exponent1.09 would not
describe these data. A surviving small AF core and a different exterior
regularity can coexist; no shrinking unresolved core is called regular.

The full beta/B arrays remain inside each copied snapshot. The audit records
their extrema and locations: at t10 beta ranges-1..+.0449966 and B ranges
-13.6226..85.6386, with maximum |B| at r4.03942e-7. These inner features are
far below the minimum radius of the existing coarse Cartesian pilots. A
coarse instantaneous comparison must label which features it actually
resolves, rather than infer them from survival or an outer norm.

## Reusable physical Cartesian exporter

`tst/elliptic_shift_3d/radial_snapshot.py` provides three subcommands:

```sh
python3 tst/elliptic_shift_3d/radial_snapshot.py self-test
python3 tst/elliptic_shift_3d/radial_snapshot.py export \
  --snapshot docs/elliptic-shift-3d/evidence/radial-collapse-snapshots-20261027/profile-0201.npz \
  --radii 0.000001,0.001,0.02,0.25,0.67,0.95 \
  --output build-shift-3d/new-unused-cartesian-export.npz
```

An actual grid CSV containing x,y,z columns can instead be supplied with
`--points`. `--samples-only` emits physical radial values/first/second
derivatives; otherwise the output contains full Cartesian scalar, metric,
curvature, independent Lambda, beta and vector-flux value/d/dd arrays.
Derivative indices precede Cartesian component indices. The trace field is
physical Khat, not Khat+3. The sidecar metadata states array shapes and source
hashes. CSV output is also supported. Existing export/evidence files are not
overwritten.

For positive chi/alpha, interpolation is a local polynomial in log-field/log
radius, with complete radial derivative chain rules and no chosen exponent.
Signed fields use physical-value polynomials in radius. Widths5/7/9 are
available. At r=0 or outside the recorded positive node interval the exporter
rejects the request: it infers no endpoint limiting data, parity or excision.
Full Cartesian vector/tensor differentiation includes the angular dependence
of n_i and n_i n_j. The source metric is exactly the recorded spherical g=1
metric; a nonzero source metric-deviation field is rejected rather than
silently flattened. Subsequent 3D evolution must keep its tensor degrees of
freedom free; this exporter is only an initializer/comparison tool.

These are **interpolated numerical snapshot jets**, not exact analytic jets
or the archived evolution RHS. Upwind/KO discretization terms are not part of
the exported continuum jets. `export-checks.json` retains width5-versus7
differences at representative core/front/bulk/scri radii. Its Cartesian
component identities agree within1e-15. The independent manufactured export
test uses fractional physical powers2.3 and1.09 at ten translated points;
maximum scaled value/gradient/Hessian discrepancy is9.86e-13. Neither check
replaces reference-resolution or original-vector-equation residual controls.

The optional translated center is a coordinate-placement test, not proof that
the selected global-radius lapse source admits a translated AF puncture. That
separate incompatibility remains recorded in the chart audit. Beta is sampled
from the source; no beta=0 puncture boundary is added.
