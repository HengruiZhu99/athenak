# Production outer-boundary discrete audit

This self-contained diagnostic package uses NumPy, SciPy and Matplotlib. Run the
scripts from this directory. No absolute runtime paths, Aurora access, AthenaK
binary, checkpoint or production campaign data are required. See
`provenance.json` for the exact source snapshot and characteristic-reference
hashes. The scripts model a flat frozen background, not a full TDE evolution.

The important result is negative: the original zero_rate boundary has growing
modes, and none of the tested polynomial, source-order, localized-kappa, simple
SAT, all-configuration or outer-sponge alternatives passes the bounded
robustness checks. The sponge can delay growth and relocate the mode to its
transition; its coarse finite-domain passes are not a cure.

Read `RESULTS.md`, `SPONGE_AND_CONFIGURATION_CONTROLS.md` and `POINT_REVIEW.md`.
Representative evidence is under `results/`. `sponge-surviving-modes.png` shows
the mode moving from the physical face to the inner edge of the layer.

Examples:

```sh
OPENBLAS_NUM_THREADS=1 python3 corner_model.py --out baseline.json
OPENBLAS_NUM_THREADS=1 python3 sponge_model.py --n 16 --rate .02 --cells 8 --out sponge.json
OPENBLAS_NUM_THREADS=1 python3 sponge_strip.py --angle .125 --rate .02 --out long-strip.json
OPENBLAS_NUM_THREADS=1 python3 configuration_boundary_model.py --mode outgoing --out all-q.json
OPENBLAS_NUM_THREADS=1 python3 constraint_identity_check.py
```

The 16x16 full-field eigenproblem uses several GB of memory. Smaller 8x8 and
12x12 cases are available for fast checks. Set one BLAS thread to avoid
oversubscribing concurrent simulation work. Only summary JSON and a plot are
archived; dense matrices/eigenvectors are generated in memory and not saved.

The compiled C++ point test is separate, under
`tst/unit/z4c_damped_boundary/` from the repository root. It tests the new physical
constraint helper, not the original `NormalDerivative(rhs)` path. The original
path has an independently confirmed uncomputed tangential-RHS-ghost bug when a
non-diagonal metric tilts the normal at an internal block edge. Around a
conformally flat zero-residual background with zero-initialized RHS ghosts, that
error is generally quadratic in the perturbation, so it does not explain away
the linear modes modeled here. A NaN-poisoned ghost exposes it directly.

The low-order SBP, all-q, damping and physical-radiation switches in these Python
models are analysis controls; they are not production input options or endorsed
fixes. The separate continuum model is included solely to reproduce the bounded
constraint identities and root discriminator; the companion gauge audit has
the more extensive root validation and completion tests.
