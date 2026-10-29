# V4 angular-dissipation GPU preflight protocol

This protocol precedes the Aurora result. It defines a bounded implementation
and restart preflight, **not a successful run or stability qualification**.
The earlier V1, V2 and V3 Minkowski duration attempts failed. The next optional
candidate adds degree-7 angular scri dissipation to final-stage algebraic
projection and cross-face angular reconstruction, while rebuilding dependent
gauge traces from the complete geometric rate.

Frozen source-v11 is commit
`d98e2a99bbf1d5546956be3d724c82715ca704ec`, with Kokkos pin
`6739bc623081648af9e752b616d9671527922cbf`. Its full 2,613-file manifest
(1,127 main-repository files and 1,486 Kokkos files) has SHA-256
`0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f`.
The fresh MPI/SYCL/PVC build uses 64 build workers. Actual compiler, cache,
build outcome and all nine executable hashes must be retained and verified;
this protocol does not presume build success.

The reviewed plan SHA-256 is
`672dccc8fa28eba2380c3752dfc19220166341210c1ba73556d8bab22b6e7269`.
It schedules fifteen ordered cases on **one debug node, twelve MPI ranks and
twelve distinct GPU tiles**, using CompactBinaryMerger. The case ceilings sum
to 3,450 seconds within a 3,510-second worker and one-hour allocation. Planned
new output is 750 MB with a separate 100 MB atomic-checkpoint reserve, under
the 5 GB campaign-home cap. The coordinator alone submits, collects and
releases the allocation. A failed case stops the plan; missing downstream
cases must remain explicitly unstarted.

1. Bulk CPU/device parity.
2. Halo/exchange CPU/device parity.
3. Original scri CPU/device parity.
4. Algebraic projection checks.
5. Cross-face angular reconstruction checks.
6. Additive scri dissipation/closure checks, including moving-normal dependent D.
7. Angular dissipation reference, energy, moment, constant and partition checks.
8. Four-step V4 epsilon-zero pulse control.
9. Four-step V3 pulse control using the same initial state and timestep.
10. Four-step V4 epsilon-2 Minkowski equilibrium.
11. Two-step V4 epsilon-2 pulse.
12. The same pulse restarted for two further steps.
13. Four uninterrupted steps of the V4 epsilon-2 pulse.
14. Exact direct/restarted checkpoint comparison.
15. Four-step V4 epsilon-2 isotropic puncture, including FastFlow at
    lmax=12, ntheta=24, nphi=48 and accepted expansion RMS <=0.001.

Evolution uses n16, dt=0.0005; the two-step prefix ends at t=0.001 and the
four-step cases at t=0.002. All **221 plan checks** must be replayed against
the reports without replacing failed checks. They include actual SYCL and
twelve ranks, current diagnostics, positive geometry, exact clocks/steps,
roundoff-scale null and algebraic residuals, the selected scheme, and V4's
epsilon/degree/header/CFL fields. The seven component cases additionally
require all 84 individual rank results and their hash-bound logs. No
communication benchmark is added because its path is unchanged.

V3 uses its unchanged 72-byte header. V4 uses the same base header plus
16 bytes `<dii>`: finite nonnegative epsilon, degree 7 and reserved zero.
Its arrays begin at byte 88 even when epsilon is zero. The verifier checks
all four arrays, including both RK compensation arrays, for finite values,
sizes, offsets and hashes. The uninterrupted and resumed epsilon-2 files
must be **identical in every byte**, including headers. Separately, the V3
and V4 epsilon-zero controls must have identical base-header fields except
the magic and identical four-array payloads. That cross-schema control is
an implementation comparison; it does not authorize cross-schema restart.

The puncture input is the previously generated n16 isotropic data, SHA-256
`c8ffa46d9007ee8b643db41980cfa4a08f9ab3f2f1c2383978a002f067f9db31`,
14,225,648 bytes. The verifier hashes the actual local input copy and checks
the worker binding. It never regenerates or evolves initial data.

After full download, `verify.py` checks the frozen source against Git objects,
build/cache/binary bindings, immutable reviewed plan, launcher hashes, job and
terminal identity, twelve-tile affinity, case order, generated-input chain,
every artifact and check, native checkpoint headers, all four arrays and the
two exact comparisons. It records failures, absent artifacts and unstarted
cases explicitly. Verification of a failed run is never scientific admission.
Missing or failed evidence yields nonzero exit status and `passed=false`.

A successful, independently verified result may support a reviewed admission
for **epsilon 2 only**, at the same low resolution, for bounded duration tests.
The epsilon-zero control has no separate GPU restart qualification in this
plan. Epsilon 0 or 1 duration runs need their own supported configuration
admission. This preflight cannot qualify a 100M wormhole-to-trumpet evolution,
convergence, Cartesian AMR or general nonspherical stability.

The verifier is read-only except for its required fresh JSON output. Run it
from a checkout containing the frozen main-repository and Kokkos Git objects:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/aurora/preflight-dissipative-v1/verify.py \
  --output output/hyperbolic-core-shell/analysis/preflight-dissipative-verification-v1.json
```

Use `--raw-dir`, `--build-dir` and `--initial-data` to select preserved local
copies. It hashes Git blobs rather than trusting the current working tree.
Component metrics are replayed with the exact source-bound parser; the
checkpoint decoder and streaming byte comparisons are implemented separately.
It does not fetch Git objects, invoke a test binary, evolve a state or contact
Aurora. `test_verify.py` exercises malformed and truncated checkpoints,
configuration/schema mismatch, all four arrays, nonfinite values, altered
clocks, rank placement and artifact-path rejection using synthetic fixtures.
