# V5 launcher and checkpoint guard checks

These are offline Python checks and unbound draft plans for the experimental
sampled-normal closure. They are not Aurora execution results, an admission,
or evidence of duration stability. No SSH, scheduler operation or native
evolution was performed by this check suite.

The 66 focused tests pass. They cover V1–V5 checkpoint parsing and export,
V5's 104-byte header with exact normal record `(1,6,4,0)`, explicit matching
report metadata, rejection of truncation/trailing/nonfinite data and
cross-schema restarts, and separate exact `normal_derivative_configs`
admission. The component parser requires both equilibrium and nonspherical
n6/epsilon-2 cases with finite bounded errors, exact finite-D invariance,
twelve-way partition/default/rejection flags, one actual SYCL backend line,
and the exact native success marker. Synthetic parser fixtures are not GPU
measurements. Existing V1–V4 checkpoint bytes are unchanged.

From the repository root, reproduce the focused checks with:

```sh
cd tst/hyperbolic_shift_3d/aurora
python3 -m unittest test_control.ControllerTests test_preflight.ComponentParserTests test_preflight.CheckpointTests test_preflight.ExportCheckpointTests
```

`tests.log` and `verification.json` retain the result, command and source
hashes. The draft plans both remain `reviewed: false`; they cannot be bound
or submitted in that state. The coordinator must review the final frozen
source, executable and component output contract before making a concrete
plan, and independently admit the actual preflight results before duration
testing. Future source-v12/build-v12 paths in these drafts are placeholders
until that freeze and build exist.

The actual local CTest output is retained as `serial-component-ctest.log`.
Its sampled-normal case completed with backend **Serial**. The parser
accepts its finite metrics and exact format, while correctly returning
`passed: false` and `gpu_cpu_parity: false` because SYCL was not exercised.
`serial-contract-verification.json` records that result without changing
the backend text. Reproduce this check from the repository root with:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/scri-sampled-normal-launcher-v1/check_serial_contract.py
```

The seven-case preflight draft contains sampled-normal device parity, four
Minkowski steps, a two-plus-two-step pulse restart and uninterrupted control,
their exact full-checkpoint comparison, and four puncture steps using the
previously audited 12/24 horizon angular resolution. Its 167 report gates
retain the existing scientific limits. It requests one debug node and all
twelve ranks for 30 minutes, with 1,710 worker seconds and 1,490 total case
ceilings. Its additional output estimate is 500 MB plus a 100 MB atomic
checkpoint reserve, under the 5 GB campaign cap.

The separate duration draft uses fresh analytic Minkowski data, n16,
dt=0.0005 and target t=0.6. The existing floating-point target loop takes
1,201 accepted steps, including the final small remainder. Its 36 gates
retain state error at most 1e-8 and H/M norms at most 1e-7, alongside the
unchanged null, algebraic, CFL, current-diagnostic and completion checks.
It requires a new source-bound V5/epsilon-2 preflight admission; no failed
V4 checkpoint is a starting point. Requested resources are one 12-rank
debug node for 20 minutes, 1,110 worker seconds and a 1,050-second case
ceiling, with 130 MB additional output plus a 100 MB reserve. Neither draft
claims puncture stability to 100 M or convergence.
