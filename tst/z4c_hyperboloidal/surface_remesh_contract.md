# Surface AMR transfer gates (declared before execution)

The independent operator test uses the native 106-leaf and outer-refined
113-leaf chart layouts. It tests both directions and validates only local new
faces. No field snapshot is globally gathered.

Changed state uses tensor degree-five raw point-value interpolation on the old
containing replacement-family outer patch. Source and target points use the
actual BlockMap geometry; the analytic polynomial comparison uses global chart
angles independently of the interpolation's physical inverse map. All 36
monomials u^a v^b, 0<=a,b<=5, must satisfy

    |computed - exact| <= 2048*epsilon*max(1,sum(abs(weights)))

on [-1,1]^2. The envelope covers tensor summation, product weights and the
forward/inverse angular map's roundoff, without using measured errors to set a
threshold. Constants are included. Smooth sin(5.1u)*cos(4.7v) is tested at
N=12,24,48, with a required L-infinity convergence order >=5.5 for each doubling
and for both refinement and coarsening (sixth-order interpolation).

Sparse transport tests use native old/new modules, seed every field and every
ChiRecord member, and poison all unowned staging. Unchanged leaf fields and all
eight record bits must match exactly, including changed owners/GIDs. Changed
independent fields follow the same coefficient-norm roundoff bound, dependent
metric/curvature slots are reconstructed, and dormant slots are zero.

Each old surface history packet routes once to the nearest new replacement
family sample in chart coordinates, normalized by the old patch span. Exact
index ties choose lower index, then newGID/k/j; packet merges use oldGID/k/j
order. Counts and correction sums are additive, minima/maxima/first time use
extrema; no-ancestor targets retain ChiRecord defaults. Tests seed integer or
power-of-two values so every additive comparison is exact in binary64 and
check every member, unique global packet coverage and global totals/extrema.

Both 2 and 8 local MPI ranks are required, including ranks with zero old or new
faces, radial inner siblings with no face, and unchanged faces adjacent to chart
seams. Coupled compatibility and actual native regrid lifecycle are independent
root-owned gates; this operator test does not claim coupled AMR stability.

## Diagnostic extension after the original rate gate failed

The first run reproduced all polynomial modes but failed the original N12→24
rate threshold: refinement order was 5.471545, below 5.5. That failure is retained.
An independent 60-decimal-digit calculation reproduces the same finite-N errors;
the six-point stencil is still asymptotically sixth order. The same waveform is
extended to N96 and N192 without changing its coarser measurements or threshold.

For each target, let `Ba = omega_a^6/720 * product_i |x_a-node_ai|` and let `Lu`
be the one-dimensional absolute row sum in u. Since both analytic factors have
amplitude at most one, tensor interpolation obeys `|error| <= Bu + Lu*Bv`.
The diagnostic checks this independently derived bound plus the existing
coefficient-norm roundoff allowance at every point/resolution. It retains the
original rate failure as an overall failure while allowing all state transport
and conservative-history cases to execute. `--diagnostic` is explicitly required;
a diagnostic run cannot be mistaken for a passing original convergence gate.

The root scientific review in `docs/z4c_hyperboloidal/amr-interpolation-review.md`
requires polynomial reproduction, all five finite-N remainder gates, unchanged
5.5 order threshold on N48→96→192, and all transport/history/migration gates.
`--reviewed-admission` explicitly selects that admission and reports
`original_gate_passed` and `revised_admission_passed` separately. The original
coarse failure artifacts remain retained. A deeper central-refinement fixture
forces actual unchanged-surface owner migration through native equal-weight
partitioning; the first interior-shell fixture changed GIDs without moving
surface owners at two ranks and is retained as a failed coverage diagnostic.
