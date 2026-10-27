# Configurable physical-lapse floor

The spherical telegrapher driver accepts `--lapse-floor VALUE`, or `lapse_floor` in a matrix JSON input's `common` or case `arguments` object. The user-requested default is `1e-10`; `0` disables clipping. Values must be finite and nonnegative, and representable in the AF logarithmic chart.

```json
{
  "common": {"lapse_floor": 1e-10},
  "cases": [{"name": "n1025", "arguments": {"n": 1025}}]
}
```

This excerpt supplies the floor setting, not the full strong-field configuration. The complete new long-run input is `inputs/elliptic_shift_1d/telegraph-lapse-floor-long-v1.json`.

The capped quantity is physical lapse `alpha/Omega`. At ordinary nodes `0 < r < 1`, values below the floor are set to it, up to an upward floating-point rounding margin. The implementation raises the stored logarithmic lapse to the corresponding threshold before each RK4 stage and after every accepted increment. The driver also applies it before its first report/checkpoint. Kahan compensation is reset only at accepted lapse components actually clipped. Other geometry variables and constraints are not projected.

The `r=0` entry stores an asymptotic coefficient, not an ordinary physical-lapse sample. It is not clipped. Actual scri retains its lapse-clock equation. This is a disclosed finite-grid regularization of the unresolved inner region, not a proof of asymptotic compatibility or convergence of a globally constant floor at the AF end.

The chosen value is included in manifests and exact restart checks. A continuation cannot change it silently. Old frozen runners and libraries retain their original behavior. Historical matrix inputs are explicitly pinned to `lapse_floor: 0` so rerunning them with the new driver does not silently change their method. `initial-data.npz` retains the analytic pre-floor initial-data construction; the first saved evolution profile contains the floor-adjusted lapse.

The original N=1025 run failed at t=43.836474609375. Offline replay localizes the stop to RK stage 4, positive-radius node 2 (r=3.628948e-9): the conformal lapse logarithm falls to -1109.306 and exp underflows to zero. The logarithmic telegrapher correction dominates the negative lapse rate as alpha becomes small; the metric deviation stays zero and chi stays positive. Native `lapse_floor=0` retains this failure. Isolated floor trials are controls, not amendments to the failed trajectory.

The replacement duration trial starts from fresh genuine wormhole data with the explicit default floor and otherwise the fixed N=1025 candidate. Its target remains t=500. Acceptance focuses on apparent-horizon-exterior constraints and apparent-horizon mass; puncture mass estimates remain diagnostic only. No long-time-stability result is claimed before that trial is assessed.
