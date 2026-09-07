# Complete Fourier and transient checkpoint

The compiled CPU/A100 Minkowski operator agrees with the independent exact
50-field source and three principal matrices. This is the stationary state
w=rho=alpha=1, zero shift, curvature, GH and auxiliary fields, sigma=1.
Production equations are unchanged. No evolution has been launched.

For arbitrary real k and nonnegative lambda, eta, kappa, the exact polynomial is

    (s+lambda)^30 (s^2+k^2)^2 (s^2+2k^2)
    (s^2+eta*s+k^2)^3 (s^2+kappa*s+k^2)^3 (s^2+2*kappa*s+k^2).

The proof checks C*A=-lambda*C for all 30 Fourier reductions and the invariant
20-dimensional reduction manifold. The triangular map from the state to its
20 primaries and 30 reductions has determinant one. The constrained 20-field
characteristic polynomial then gives the full polynomial above. Exact
rotational covariance of both the source matrix and principal matrices extends
the axis calculation to every direction. The oracle is imported from the pinned
candidate files, with hashes recorded; no reference main is run in that tree.

Each of three parameter triples (lambda,eta,kappa)=(1,2,1),(0,2,0),(1,0,0) has
150 compiled principal basis jets and 200 centered source perturbations at
steps 1e-4 and 5e-5, plus the zero-RHS check: 1053 point calls per backend.
Source and principal errors are <=1.34e-16 and 6.67e-17. CPU and A100 outputs
are numerically identical for this fixture. The exact polynomial has no
positive-real-part roots under these parameter signs, but this is not a
contractivity statement.

## Neutral modes and measured amplification

At k=0, exact nullities of J,J^2,J^3,J^4 are respectively:

| Parameters | Nullities | Length-two zero chains |
|---|---|---:|
| (1,2,1) | 7,13,13,13 | 6 |
| (0,2,0) | 40,47,47,47 | 7 |
| (1,0,0) | 11,20,20,20 | 9 |

For every rate triple, an explicit chain is
J e_K=(e_w-7 e_rho)/3 and J^2 e_K=0. Thus even damped GH/reductions do not
remove all homogeneous linear drift of primary variables. This is a neutral
Jordan effect, not a positive exponential eigenvalue. At k=1 the sampled
neutral roots i and sqrt(2)*i have equal first/second-power nullities, matching
their algebraic multiplicities; no Jordan extension is detected there.

For the oblique normal (1,2,3)/sqrt(14), k=0,0.01,0.1,1,10 and
 t=0,0.1,1,5,20, the largest sampled ||exp(tA)||2 is 72.5846709682 at
(lambda,eta,kappa)=(0,2,0), k=0.1, t=20. A separate 60-digit full matrix
exponential confirms that norm to 3.284e-15 normalized (the final SVD norm
uses double precision). The damped case reaches 58.4251 at k=0,t=20.
These are raw full-state Euclidean norms with M=1 normalization; they are not
an invariant energy, an all-time maximum or physical instability thresholds.

Compiled/reference exponential agreement is <=8.03e-14 normalized and
C exp(tA)=exp(-lambda*t) C holds to <=3.51e-15. Decaying defining constraints
can coexist with primary-variable transient amplification. The exact neutral
structure must therefore be retained when interpreting future RK spectra;
small floating positive real parts alone are not evidence of runaway.

## Numerical environment and reproduction

The first local analysis stopped under warnings-as-errors inside SciPy's expm
matrix-product path. That failure and the raw CPU outputs are preserved. The
same saved CPU batch passes on Linux (numpy 1.20.3, scipy 1.7.1, sympy 1.9), as
does the CUDA batch. Both used one CPU thread for analysis. No tolerance was
relaxed or warning suppressed. The 60-digit independent control also passes.
This isolates an analysis-environment issue; it does not prove a library bug.

```sh
python -W error analysis/pc_gh_clean_reduction/check_intrinsic_fourier.py \
  --binary /absolute/path/to/intrinsic_rhs \
  --reference-dir /absolute/path/to/candidate_20260906/analysis \
  --output /absolute/path/to/new-directory
python analysis/pc_gh_clean_reduction/check_fourier_exact.py \
  --reference-dir /absolute/path/to/candidate_20260906/analysis \
  --output /absolute/path/to/new-exact.json
python -W error analysis/pc_gh_clean_reduction/check_fourier_high_precision.py \
  --reference-dir /absolute/path/to/candidate_20260906/analysis \
  --cases /absolute/path/to/new-directory/cases.json \
  --output /absolute/path/to/new-high-precision.json
```

Replay adds `--replay /absolute/path/to/saved-batch` to the first command;
it requires identical regenerated input bytes and reads the saved output,
run.log, kokkos.txt and binary-sha256.txt without executing the supplied binary.
A100 execution and Linux analysis took about 2.50 seconds combined; CPU-output
Linux analysis took 2.57 seconds. Remote ownership and GPU occupancy are saved.

Results, exact and high-precision checks, all roots and sampled norms, plot,
warning log, source/build identities and hashed external raw inventories are in
qualification-runs-20260907/pcgh-clean-reduction/intrinsic-fourier-001/.

The next work is the coupled discrete operator including RK3 and KO, followed
by intrinsic mesh/transfer/restart integration and ordered physical qualification.
This Fourier result does not qualify variable backgrounds, interfaces or punctures.
