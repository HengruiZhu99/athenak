# Independent finite-rb continuum constraint-rate readback

All declared rb=.98 pointwise rate gates passed. This validates the selected
stationary-reference continuum fields and their coefficient-aware constraint
rates. It supplies neither a discrete Bianchi identity nor an energy or boundary
stability theorem. No radial operator, eigenvalues, propagation, native pulse or
black-hole evolution was performed here.

The unchanged actual C0/spatial-norm point bridge uses S=1, a=.5, geometric layer
.05–.95, physical-P lapse/storage, xi=2, kappa_input=10 and kappa2=0. The actual
Release executable is `2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15`;
the new API header is `d60ea8266abebf5263eb78b81997b60af472dc7b1fe102bd6f2c21baca1a6017`.
It is included after the already source-gated actual bridge. It adds no equations
or reference subtraction. The physical Cartesian ordering throughout is
H, Mx, My, Mz, Zx, Zy, Zz, Theta_physical, with no Omega rescaling or moving frame.

| Gate | Cases | Maximum final scaled L2 discrepancy | Time |
| --- | ---: | ---: | ---: |
| Exact polynomial core source | 4,431 | 2.201e-16 | 445.02 s |
| Exact polynomial core actual constraint rate | 4,431 | 9.443e-16 | included above |
| Gauge-only zero constraint rate | 714 | 1.136e-7 absolute L2 | 16.66 s |
| Shell actual rate versus subsidiary | 2,751 | 1.979e-8 | 84.93 s |

The exact symbolic core proof gives all 160 zero entries of
`C(d) L(d) T = B(d) C(d) T`, where d_i are commuting Cartesian derivatives and T
is the complete algebraic det/trace tangent chart. For arbitrary kappa_input,

```
H_t     = -2 div M
M_i,t   = -.5 partial_i H + Delta Z_i - partial_i div Z
          + 2 kappa_input partial_i Theta
Z_i,t   = M_i + partial_i Theta - kappa_input Z_i
Theta_t = H/2 + div Z - 2 kappa_input Theta.
```

The physical core lift has chi=-tau/sqrt(3), g=h_STF and independent tracefree A;
H=divdiv(h_STF)-2 Delta(tau)/sqrt(3), M=div(A)-2 grad(P+2Theta)/3, and
Z=(Lambda-div(h_STF))/2. The actual core gauge is alpha_t=-3P,
beta_t=3Lambda/8. The proof's formal raw22 extension is not identified with the
actual off-normal constraint functional. Literature independently reviewed the
source/signs without a correction; it did not rerun the proof or kernel.

The actual core gate covers J0/1/2, all nonnegative m and nonzero real/imaginary
phases, W=1,rho,rho²,rho³ and the declared alternating mixed polynomial. Exact
Cartesian polynomial source/constraint jets are evaluated at r=0,.025,.049;
there is no r^L division or FD stencil crossing the geometric transition. The
actual initial constraint discrepancy is at most 1.852e-16 scaled; the frozen
subsidiary's polynomial rate discrepancy is at most 2.144e-16 scaled. Negative-m
conjugation remains in the already frozen basis controls, rather than being
claimed as a new numerical check here.

For the transition/collar gate, complete source F(x) and initial physical q(x)
are independently evaluated at every required Cartesian sample, then separately
differentiated at the frozen five h values. First derivatives and standard
diagonal second derivatives are fourth order; mixed derivatives compose the
first derivative stencil. Each h uses the full 61-point stencil, with identical
coordinate samples reused across h levels. The actual DC_ref[F] consumes the
full raw22 jet without an inserted algebraic projection. The subsidiary consumes
the independent physical q jet and the stationary analytic reference, including
all coefficient gradients and the actual C0 connection/shift terms. This does
not replace them by an isotropic frozen coefficient.

The initial gauge constraints and corresponding subsidiary rates are exactly
zero. The largest actual final rate L2 is 1.13584546e-7 and largest extrapolated
rate is 1.19024473e-7, both below the fixed 2e-7 tolerance. Of 714 actual sequences,
512 have the required >=8 reduction in at least one pair of non-floor
increments; 202 pass the fixed absolute gate with order explicitly unclassified.
All 714 exactly-zero subsidiary sequences are order unclassified.

The shell includes each P,Theta,Lambda,metric trace,metric STF and independent A
channel, plus the declared alternating nongauge mixture. No stronger Theta or Z
falloff is imposed. The maximum extrapolated actual/subsidiary scaled L2 error is
2.10749594e-8. Last scaled increments are at most 1.926e-8 actual and 8.107e-9
subsidiary. Actual sequences classify 2,100 fourth-order cases and 651 cases within
tolerance with unclassified order; subsidiary counts are 2,021 and 730. The
largest final absolute component errors are 1.166e-7 in H, 6.290e-8 among M,
2.966e-11 among Z and2.139e-11 in Theta. Every case preserves all five vectors,
increments, observed orders, extrapolation and absolute/scaled discrepancies.
Unclassified order is not called a proof of fourth-order convergence.

There are 7,896 cases and 23,688 successful API processes, 546.61 s total. Every
process has empty stderr. The original core driver and its 14 input hashes remain
unchanged. A fresh additive v2 driver batches all samples per witness and appends
call receipts, avoiding repeated serialization of the growing core log; it also
tightens the exact-zero initial gauge check to absolute 5e-11 and records launch
HEAD/command directly. Scientific fields, stencil, five h values and tolerances
are unchanged. The core launch HEAD 2e0aa3b0 is separately reconstructed from the
reflog in an additive receipt; the two FD stages directly record that HEAD.
The public runtime implementation remains 27c19d20. All 14 source/executable pins
match before/after each stage and on independent saved-data readback.

Three API-mode comparisons, each with origin/core/transition representatives,
also pass Release versus ASan/UBSan with bit-identical output and empty stderr.
The first Debug launch used a retained byte-copy lacking executable permission;
its failed source/input/Release output/receipt are preserved separately. A fresh
v2 script uses the hash-identical executable working Debug 75193a3b and succeeds;
no binary bytes, equations, scientific inputs or tolerances changed.

The summary's RMS/peak values are unweighted sample statistics over the declared
fields/points, not integrated Penrose or physical covector norms. Boundary owns
the radial quadrature, energy identity, source projection defect and SAT
constraint production. rb=.995 has not been tested in this bundle. These gates
do not establish nonlinear live subsidiary closure, full-jet exact-scri
regularity, radiative data admission or the later single-BH wormhole-to-trumpet
transition. The Minkowski hyperboloidal reference remains the baseline for that
later goal.
