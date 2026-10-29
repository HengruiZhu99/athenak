# V4 immediate ghost-geometry rejection

The captured V4 failed stage has **positive-definite metrics at every scri
node**, but its unchanged componentwise 6×6 Lagrange interpolation produces
indefinite ghost-boundary metrics. This identifies the immediate rejection
mechanism. It does not identify the first growing mode or establish that
changing interpolation would stabilize the coupled scheme.

The independently bound run is Aurora job `8983855`, frozen source-v11
`d98e2a99bbf1d5546956be3d724c82715ca704ec`, n16, dt=0.0005, degree 7 angular
dissipation with epsilon 2. The accepted rollback is t=0.5224999999999979,
step 1045. The failed attempted interval is [0.5225,0.5230]; its precise RK
stage abscissa was not recorded. The last fresh diagnostic is t=0.5 and is
stale relative to the accepted rollback. The checkpoint is explicitly
unadmitted and was not restarted.

[Evidence verification](results/evidence-verification.json) independently
replays all 33 worker gates, retaining all nine failures, and checks all five
artifact hashes, 2,613 frozen source/Kokkos blobs, the exact executable,
epsilon 2 admission, actual launch, twelve-tile placement, terminal F/exit1
and allocation release. The V4 checkpoint uses the strict 88-byte schema
with epsilon 2, degree 7, reserved 0. Its scri-state offset is `88 + 16*nv`;
the unchanged failed-stage scri offset is `8*nv`, where `nv=31*19*n^3`.
Both scientific inputs remain unchanged after analysis.

All 1,536 captured scri nodes have finite fields and positive chi. The minimum
nodal chi is 0.8088679492, minimum metric eigenvalue 0.1224263323, and minimum
determinant 0.9979086151. The accepted rollback also has positive nodal
metrics; its minimum eigenvalue is 0.2338941778. The failed snapshot is an
intermediate stage before final-stage algebraic projection: its determinant
and tracefree residuals need not equal those of an accepted projected state.

The native host replay evaluates 6,616 globally deduplicated exact-coordinate
queries. Twelve queries fail the metric-validity test, with no nonfinite-field,
nonpositive-chi, or independent inverse-normal rejection. These twelve exact
floating-point query coordinates represent four directions after rounding
coordinates to twelve decimals. They are not twelve separate physical
locations, nor a sum of rank-local invalid counters. All four directions use
+z scri donors (face 4, block 17) and feed radial ghost indices 16, 17, 18 of four
side blocks:

| Recipient block | Representative query | Failed minimum eigenvalue | Failed determinant | Same-query rollback minimum eigenvalue |
|---|---:|---:|---:|---:|
| +x, block 13 | 1015 | −0.404381381 | −0.483428886 | +0.412882679 |
| −x, block 14 | 2142 | −0.148545168 | −0.177344144 | +0.475235328 |
| +y, block 15 | 2545 | −0.146768688 | −0.174553516 | +0.476691154 |
| −y, block 16 | 4123 | −0.397498174 | −0.471324052 | +0.418869283 |

For the worst query, interpolated chi remains positive at 0.8337340883.
Its 36 donor metrics are all positive definite, with minimum donor eigenvalue
0.1461455924. The interpolation weights sum to 1 within roundoff, but their
L1 norm is 2.7709695913 and their minimum is −0.3744739747. Thus this linear
componentwise interpolation does not preserve the positive-definite cone.
The reported `q=0` is a consequence of `Geometry` returning before inverse
construction when the metric is invalid; it is not a second independent
failure mechanism. Exact coordinates, 36 weights, donor IDs, full interpolated
fields and recipient ghost indices are retained in
[interpolation-audit.json](results/interpolation-audit.json); independent
NumPy reconstruction and eigenvalues are in
[interpolation-summary.json](results/interpolation-summary.json).

The audit reuses the earlier read-only C++ helper. All six geometry,
mesh-mapping, stencil and exchange headers are byte-identical between frozen
V3, frozen V4 and the checked local sources. It calls the native `Geometry`
check and reproduces the exact frozen stencil construction and anchor-relative
sum. This is a CPU binary64 replay, not a bitwise replay of SYCL arithmetic.
The finite rejection margin and independent Python reconstruction support the
same immediate positivity failure. Four synthetic controls pass: constant
positive identity, negative chi, indefinite metric and nonfinite auxiliary
field. No projection, positivity repair, clamping, evolution or restart is
performed by this audit.

To reproduce using the preserved local failed-stage/checkpoint payloads and
configured local Serial Kokkos build, run from the repository root and choose a new output directory. Prerequisites are Python with NumPy, `/usr/bin/c++`, the retained raw payloads, frozen Git objects and the existing `build-hyperbolic-mpi` configuration. The recorded compile command, flags and linked Kokkos libraries are in `results/audit-build.json` and `results/audit-provenance.json`:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/stability-dissipative-minkowski-v1/positivity-audit/run_audit.py \
  --output output/hyperbolic-core-shell/analysis/v4-positivity-recheck
```

`--raw-dir` and `--build-dir` override those local inputs. Only small JSON
results are retained here; temporary synthetic payloads and the audit
executable are removed automatically. The overall history comparison is
maintained separately in the parent campaign's analysis directory.
