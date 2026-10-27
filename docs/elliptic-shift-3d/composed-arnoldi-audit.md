# Composed metric operator: independent Arnoldi audit

This is a standalone diagnostic of the proposed composed operator. Production
headers, equations, gauge sources, boundaries and tolerances were not changed.
The earlier native adaptive package was left unchanged and committed separately.

## Implementation inspection

The right-preconditioned FGMRES implementation correctly stores each preconditioned
basis vector, performs two modified Gram–Schmidt passes, applies Givens rotations,
and recomputes the original residual before acceptance. The MMS harness correctly
subtracts the inhomogeneous lift once and certifies the same original equation.
I found no sign, component-order, double-lift, or Arnoldi-update defect explaining
the 120-iteration failures. `MetricRate` stores global Cartesian metric components;
its six-component packing/unpacking in the composed probe is consistent.

There is a concrete discretization/preconditioner mismatch. The composed operator
forms `ShiftMetricRate`, interpolates that active rate across interfaces, and takes
another centered first derivative. Its outer rate trace also depends on the
Hermite reconstruction of beta derivatives. Scalar `CompositePoisson` instead
uses a direct order-two second derivative and its own lower-order ghost lift.
In a uniform flat interior, the fourth-order first derivative has symbol
`i sin(theta)(4-cos(theta))/(3 h)`, so its square vanishes at theta=pi. The direct
second derivative used by the scalar preconditioner is 4/h² there. Thus scalar
MG need not resolve the composed operator's high-frequency branches. This local
symbol observation alone does not establish a global nullspace.

The MMS coefficients use analytic metric jets. They are a valid variable-metric
consistency control, but calling that particular setup the exact Frechet derivative
of a fully sampled metric-built connection would require replacing analytic metric
first derivatives by the same discrete differentiation. The independently audited
actual-state source/operator identity is separate and remains valid.

## One bounded full-action diagnostic

Command:

```sh
python3 tst/elliptic_shift_3d/run_composed_arnoldi.py \
  --output docs/elliptic-shift-3d/evidence/composed-arnoldi-20261027
python3 tst/elliptic_shift_3d/analyze_composed_arnoldi.py \
  docs/elliptic-shift-3d/evidence/composed-arnoldi-20261027
```

Both operators use the same fixed flat N6 Cartesian core plus seven shell layers,
10,800 active cells, zero vector boundary data, existing scalar MG, and the actual
`CompositeShiftStage::Dot` physical measure. The start vector is identical and
includes deterministic noise. Each operator takes 80 unrestarted Arnoldi steps
with double MGS. Full Q and Hessenberg data are retained. Source dependencies are
frozen before compilation, the compiler dependency probe finds no live project
headers, and binary/library/source hashes remain unchanged. No shared build target
is rebuilt. Build took 3.23 s; expanded/composed Arnoldi took 13.35/18.95 s; fresh
singular witnesses took 1.44/1.63 s. All subprocesses have explicit bounds of at
most 180 s and use one Serial worker.

| Quantity | Expanded AP | Composed AP |
| --- | ---: | ---: |
| Orthogonality error | 8.88e-16 | 8.88e-16 |
| Best relative residual at 16 directions | 1.75e-8 | 0.19813 |
| At 32 directions | 2.68e-15 | 0.18117 |
| At 48 directions | 2.22e-15 | 0.17574 |
| At 80 directions | 1.96e-15 | 0.16692 |
| Smallest singular value on this Arnoldi space | 0.58269 | 0.21481 |
| Restricted singular condition | 4.80795 | 16.02572 |
| Relative projected nonnormal commutator | 0.04420 | 0.10426 |
| Projected numerical-range real interval | [0.24525,2.47864] | [-2.79394,1.78601] |

The fresh full `AP v` actions reproduce the singular predictions to 1.11e-16
and 2.78e-17 absolute error. The composed weakest singular witness gives
`||A z||/||z||=646.12`, with `z=P v`. It does not demonstrate a physical near-null
mode. There is no full spectral-gap claim, and these spaces do not exhaust all
possible localized modes.

A second fresh action verifies the most negative projected numerical-range
vector. Its actual full weighted `<v,AP v>` is -2.793939041462449, versus projected
-2.793939041462415, with `||v||=1.000000000000006`. For its physical correction,
`<z,A z>/<z,z>=-857.3183784732684`, while `<v,P v>` is positive. Consequently the
composed global discretization itself is noncoercive in this physical norm; this
is not merely a projection or right-preconditioning artifact. This does not prove
singularity, negative eigenvalues, or ill-posedness of its discrete boundary problem.

The negative-direction output norm is 3.00471. Cartesian-core and first-transition
contributions are 2.80903 and 1.04761; outer-shell contribution is 0.04805. The output
maximum is in Cartesian block 5 at (0.2291667,-0.0208333,0.1875), logical index
(5,5,4), adjacent to the Cartesian/shell interface. The fixed mesh has no coarse-fine
interface; zero coarse-fine coverage is explicit. All cells remain included in
norms and CSVs. This localizes an actionable failure mechanism to compound interface
differentiation and transfer, with little evidence that the scri trace dominates
this particular flat-grid stagnation.

The first NumPy analysis emitted floating-status warnings from small BLAS matrix
products despite finite results. The independent analysis uses scalar `math.fsum`
products and LAPACK factorizations, emits no warnings, independently reconstructs
physical-volume weights, and verifies the same quantities and fresh-action
witnesses. No numerical result depends on suppressing those warnings.

## Next falsifiable engineering step

Increasing restart length alone is insufficient for this start vector: even the
unrestarted space stagnates. A suitable next control preserves the composed A and
its original residual, but changes the preconditioner near the Cartesian/shell
interface. A small overlapping Schwarz correction must assemble the *actual*
composed response, including active-only transfer and intermediate metric-rate
halos; a scalar direct-D2 interface block would repeat the mismatch. Alternatively,
a bounded correction/deflation space built from the verified troublesome physical
vectors can test whether those interface directions explain stagnation before
implementing a complete composed multigrid hierarchy.

If the final discretization is changed, a compatible first-derivative/interface
pair (for example a summation-by-parts construction with compatible transfers)
needs its own metric-built-V variation and boundary audit. Replacing the physical
operator by the expanded control, removing metric terms, changing the gauge,
subtracting a physical source, or weakening tolerances is not justified here.
