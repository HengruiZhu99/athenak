# Initial floor versus total-RK clipping

This accounting check does not change or evolve the numerical state. It compares
the preserved analytic `initial-data.npz` with the actual evolved initial state
`profile-0001.npz` in the fresh N1025 attenuation trial. Frozen numerical bindings
are verified against the captured manifest through t401.95.

The initial floor modifies 80 positive-radius lapse samples, all at r<=6.29212e-7.
Its maximum conformal-lapse change is4.99995971398e-11 and maximum physical-lapse
change is9.99991942797e-11. All other initial fields, the AF coefficient endpoint,
and the scri lapse are verified unchanged. `review.npz` preserves the positive
interior radii, conformal factor, before/after lapse, and logarithmic changes.

The separate native `max_lapse_floor_conformal_correction` counter is updated by
`store_alpha` during total-RK stages and accepted updates. It does **not** include
the initial logarithmic `floor_lapse` operation. Its maximum through the captured
manifest is5.73768884405e-15, with zero nonpositive total-lapse clamps. The radius,
logarithmic-correction, and per-call count counters include initial flooring.
These populations must not be conflated in the manuscript. No acceptance limit,
floor value, source, or active-run parameter has changed.

Reproduce from the repository, choosing an unused output filename:

```sh
python3 -W error tst/elliptic_shift_1d/review_telegraph_floor.py \
  --case output/elliptic-shift-1d/telegraph-attenuated-duration-v1/n1025 \
  --export output/elliptic-shift-1d/telegraph-attenuated-duration-v1/monitor/validated-t400 \
  --output /tmp/telegraph-floor-review.json
```

`review.json` retains exact source hashes and counter scope. The initial maximum
is independently reconstructed from the saved log chart, compared with the saved
native lapse, and checked against max(alpha_initial, lapse_floor*Omega).
