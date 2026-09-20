# Strong-field static-SMR Aurora diagnostic — NOT SUBMITTED

This is a prepared, guarded next test. No job was submitted, uploaded, restarted or modified. Aurora authentication is unavailable; the allocation, environment and readiness must be rechecked after access is renewed. `READY.json` is intentionally absent and the PBS script refuses to evolve without it. There is no submission helper or automatic resubmission.

The deck requests **MHDTidal / debug, two nodes, 24 MPI ranks (12 per node), one-hour PBS walltime**. It preserves the immutable GPU executable

```
/lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/outer-boundary-fix-20260920/athena-boundary-fixed
SHA256 a6c3af79571819fba5dc2252ceb9abacb31279feec440f2c542e342dfee43639
```

That executable includes the repaired active-only boundary stencil but predates the new compact-constraint-seed diagnostic. This diagnostic uses only its already-existing compact **lapse** seed and original `zero_rate` boundary. It does not require, or claim to test, an unbuilt GPU source revision. SHA checks run before both applications.

## Physics and sequence

Both fresh inputs have 232 static blocks of16³ cells in [-32,32]³M, with levels0–3 and finest dx.125M throughout the sphere r≤4M. The centered M=R0=1 Schwarzschild trumpet has coordinate horizon r=1M, resolved by16 cells across its diameter. The original G1 background-adapted gauge is retained, with kappa1=0, kappa2=0, eta=.02 and lapse damping=.01. Linear extrapolation and sixth-order volume derivatives are unchanged.

The radial C2 sponge starts at8M, ramps over20M and reaches rate.05/M at28M. Its source timestep cap is20M. All inner freeze/projector/excision/sponge options remain off. This is decoupled vacuum: the numerical fluid floor has no matter feedback; no star or production checkpoint is involved.

1. Fresh zero control: three cycles, application cap two minutes. Every one of24 final restart files must be present with matching headers, cycle3 and the same time. The entire residual payload, including all levels and ghost cells, must be exactly zero; the entire raw payload must be finite; reconstructed full alpha, chi and conformal-metric Sylvester minors must remain positive. Histories must be finite and invalid-metric counts zero. Any failure prevents the pulse launch.
2. Fresh compact lapse pulse: amplitude1e-8, center(2.5,0,0)M, support.75M, zero outside support. It uses `-i`, never a restart from the gate. Target1000M, application cap **`-t 00:55:00`**. Binary and restart outputs are per rank; histories remain globally reduced. The driver flushes its endpoint checkpoint on a clean walltime stop.

The pulse starts only if the gate and validation complete within the first three minutes, leaving two minutes after its55-minute cap for final validation. `mkdir` refuses to overwrite an existing job run directory. Input/helper hashes, executable SHA,24-rank count,232-block count, MPI geometry and stopping reason are checked. No physics or target override is supplied on the command line.

## Unequal rank partition and validator test

The equal-cost contiguous partition assigns nine blocks to ranks0–7 and ten blocks to ranks8–23, mean9.67 blocks/rank. The SMR reader uses actual logical levels and reconstructs all leaf/ghost coordinates; the uniform-grid checker must not be used.

The24-rank parser was tested by privately repartitioning the existing eight-rank zero preflight's raw saved blocks without changing any block data. All24 synthetic rank files passed exact zero, finite payload, common headers, independently audited leaf geometry and raw ghost metric checks. Missing rank23 and a mixed cycle header were rejected. A single indefinite fine-level ghost at rank2/gid21, x=(-4.0625,-2.9375,-2.9375)M was correctly rejected. Every original eight-rank source file remained byte-identical. This is a validator/ownership test, **not a24-rank numerical evolution**; the private fixture is marked ineligible for evolution and is not packaged.

`check_run.py` distinguishes requested target completion from a clean walltime stop below target and from failure. It never declares stability merely because fields remain finite. The saved full headers include the implicit global gid ordering; restart payloads do not contain independent per-block gid signatures, so matching same-sized rank payloads cannot be authenticated without their provenance hashes. All hashes are retained.

## Runtime and scope

The local eight-rank CPU preflight measured8.92–9.03 seconds/cycle with dt.025M over only three cycles. At unchanged CPU throughput,40000 cycles to1000M would take roughly99–100 hours. This is a short CPU observation with substantial uncertainty; it is **not a GPU forecast**. The two-node GPU test must measure its own seconds/cycle and actual timestep before estimating its finish time. A55-minute walltime stop before1000M is an expected possible outcome and must not be called target completion.

The three-cycle local preflight passed previously, but no long strong-field or24-rank GPU stability claim follows. The near-horizon background finite-difference Hamiltonian is nonzero even with an exactly zero residual. Refinement-interface, near-horizon and outer-layer constraint evolution need separate review after this test. No successor may be submitted automatically.

## Readiness and offline checks

`package-manifest.json` hashes the frozen prepared files. `check_package.py` is read-only and cannot submit or arm a job. After access is renewed and the actual queue/environment are reviewed, the coordinating agent may create a readiness record containing campaign `strongfield_smr232`, the exact SHA256 of `package-manifest.json`, and `reviewed_after_access_renewal=true`. No such record is distributed here. This prevents an old archive from being blindly rerun; it is not a request for additional user authorization.

Offline checks:

```
bash -n submit.pbs
python3 check_package.py
python3 check_package.py --require-ready  # must reject the unarmed prepared deck
```

`validator24-regression.json` records the completed private-fixture checks. The input parser and mesh construction passed in the parent strong-field plan; no new evolution is performed by this preparation.
