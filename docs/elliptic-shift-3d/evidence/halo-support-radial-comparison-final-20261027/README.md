# Same-mesh halo controls versus independent radial evolution

All three genuine full3D RK4 cases reach t=.0002 and remain scientific FAIL.
Every one of the47,088 active cells and all64 raw angular samples at each
profile radius are retained. The halo support changes the largest H defect
.406127→.195554→.0478254 but barely changes physical radial profiles: beta RMS
error remains .000412883 against the matched independent selected.24 N2049
reference. Its N1025 control is much smaller. Raw errors, unnormalized L2,
all-region norms, weights, reference controls and scatter are in comparison.json
and the NPZ files. No late collapse or convergence inference follows.

The original wrapper stopped at a plotting KeyError; it is preserved in the
adjacent first-attempt directory. A second rendering had a stale differing-mesh
subtitle inherited from the general analyzer. This final rendering explicitly
labels the same-mesh controls. Neither correction reran or changed an evolution.
