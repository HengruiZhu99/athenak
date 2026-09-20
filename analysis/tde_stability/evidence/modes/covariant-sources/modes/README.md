# Independent full-map checks of the covariant-source candidate

The candidate binary and source validation are owned by the parent `covariant-sources` experiment. This directory independently tests the complete RK/physical-boundary/projection map using the validated full-state hook. No production source, job or queue was changed.

Binary: `../athena-covariant-modehook`, SHA256 `ef1a5473e3bfb5722240cf96a65284a6a5a78c8987509d6de0719a572f84dad1`. The flag `ccz4_covariant_sources=true` adds the coupled geometric source terms and changes Gamma damping to the printed CCZ4 coefficient `−sigma Q`; this is not the earlier lapse-scaled-only hybrid with coefficient2.

Conditions: vacuum Schwarzschild R0=M=1 trumpet, adapted residual gauge with G=2/f=1,16^3single block, box±2M,dx=.25M,sixth-order/RK3,dt=.0375M,KO=.5,cubic ghosts and original zero-rate characteristic closure. Complete raw state includes25fields and all ghosts. Matter feedback, inner excision and outer sponge are off.

## Map sanity

All three damping profiles pass:

- exactly zero residual remains bitwise zero across20steps, including ghosts;
- one step composed with another equals two uninterrupted steps bitwise;
- response to old discrete eigenmodes at peak amplitudes1e-3 and3e-4 agrees to a few parts per million.

See `hook-validation.json`, `manifest.json` and call logs. `prepare_and_probe.py` reproduces the checks.

## Old-mode responses are not new eigenvalues

The old cubic-G2 eigenvectors (growth .0464840 and .0357344/M) were supplied as independent directions. Each central-difference map advances3M (80steps). The new operator changes their shapes, so projected log gains must not be called its eigenvalues.

| coordinate damping product | active norm gain, old faster mode | active norm gain, old slower mode |
|---|---:|---:|
| sigma=.1alpha |1.10824|1.05719|
| sigma=.1 |1.10392|1.05299|
| sigma=.3 |1.08584|1.03465|

The constant .3 product gives the smallest active norm gains in this short comparison, but not complete-state decay. Its faster direction has physical H/M/Q norm ratios .960/.950/.846; its slower direction has1.064/.865/.825. This illustrates why neither a lower Theta amplitude nor short-time decay of selected constraints establishes complete stability. See `old-mode-responses.json` and `old-mode-physical-constraint-responses.json`.

Physical constraints use central signed double-precision probes of the existing ADM constraint kernel. The fixed background discretization defect cancels. Q is independently linearized with the same sixth-order derivative on the conformally flat background. These are discrete diagnostics, not proof of continuum constraint convergence. The probe binary itself does not evolve the states.

## New covariant sigma=.3 modes: both still grow

A single32-vector Arnoldi test, seeded by both old mode directions, identified two positive real branches. It approximates the full discrete tangent map by central differences; it does not survey the entire spectrum. All25fields and ghost values participate, and the map includes the actual RK stages, physical boundaries and algebraic projection.

| branch | multiplier over3M | growth gamma/M | e-fold time | direct full-state relative residual |
|---|---:|---:|---:|---:|
| faster |1.071681097|.02307618|43.34M|1.36e-6|
| slower |1.027922996|.00918009|108.93M|2.14e-6|

Independent80-step responses at peak amplitudes1e-3 and3e-4 agree within3.76e-6 in the full state, including ghosts, and1.62e-9 relative in active cells. One-step active projected rates are .02307615 and .00918070/M. The slower branch is less precisely converged; avoid quoting its growth rate beyond about .00918/M.

Signed physical H, covariant M_i and conformal Q^i each reproduce the corresponding growing multiplier. Their relative eigen-shape residuals are below2.2e-5. The faster branch's physical H peak is at(.375,-.375,-.125)M, M at(.125,-.125,.125)M and Q at(1.875,.125,.125)M. For the slower branch H/Q peak in the outer part of the box, while M peaks near the hole. These are locations of a global discrete eigenfunction, not proof of where initial errors are injected. They do not establish a continuum puncture instability.

The coupled source change and stronger damping reduce these two growth rates relative to the original (.0464840,.0357344)/M and lapse-scaled-only sigma=.3 (.0351450,.0243533)/M cases. They **do not cure the observed discrete constraint instability**.

Reproduction: `arnoldi_candidate.py covariant_constant03 32`, then `validate_candidate.py`. See `covariant_constant03-arnoldi32-results.json`, `candidate-validated-modes.json` and the compressed physical-constraint fields. The candidate executable hash is recorded above and in `manifest.json`. The physical-constraint probe uses a separate unchanged constraint kernel at zero evolution steps; it does not evolve the candidate with the old equations.

## Sigma1 directional probes: transient comparison only

As a bounded follow-up, these same two sigma=.3 eigenvectors were evolved for3M with the covariant sigma1 operator. There was **no sigma1 eigensolve**. Two amplitudes agree within3.2e-6 globally; active maximum discrepancies are below6.1e-11. Exactly zero state remains bitwise zero through20steps.

| input sigma=.3 direction | active-state norm ratio | physical H norm ratio | M norm ratio | Q norm ratio |
|---|---:|---:|---:|---:|
| faster |1.04416|1.15721|.85752|.66072|
| slower |1.00396|1.38432|.72803|.62671|

Active shapes change by about6.6%/5.6%; physical constraint shapes change substantially. These are transient responses, **not sigma1 eigenvalues or a stability verdict**. Sigma1 initially reduces M and Q in these directions while H increases. Complete evolution and its own mode analysis would be needed to assess late behavior; neither was added here.

Reproduction: `sigma1_responses.py`; data: `sigma1-directional-responses.json`; comparison plot: `covariant-mode-comparison.png`/`.pdf`. `plot_comparison.py` regenerates the figure. This remains a single-block CPU proof of concept; no refinement, MPI, GPU, matter or production-stability claim follows.

## Later authorized sigma1 Arnoldi follow-up

The slowing but persistent growth in the separate long sigma1 control motivated one additional bounded32-vector solve. This is a subsequent experiment; it does not turn the earlier directional responses into eigenvalues. Its seed is the normalized sum of the two sigma=.3 eigenvectors and the finite sigma1 checkpoint residual at500.034375M, cycle13368. Each component was normalized before addition. Only the25Z4c fields, including ghosts, were taken from that checkpoint; fluid payload was skipped. The checkpoint direction was rescaled around the exact vacuum background, not evolved as a nonzero base state. Its saved metric/payload checks pass. Seed paths and hashes are in `covariant_constant10-arnoldi32-seed.json` and `sigma1-seed-checkpoint-validation.json`.

The same immutable candidate executable, G=2, dt=.0375M and3M complete map were used. A weak oscillatory growing pair is approximately resolved:

```
mu(3M) = 1.00255712 +/- 0.07520986 i
|mu| = 1.00537421
gamma ~= 0.0018 / M       (e-fold time about560M)
|omega| ~= 0.02496 / M    (oscillation period about252M)
```

The numerical values should not be interpreted more precisely than their convergence supports. At dimension32, its Arnoldi residual is2.41e-5. Independent direct3M responses give relative eigen-shape residuals2.41e-5 globally and8.50e-5 on active cells. Peak-amplitude1e-3 versus3e-4 responses agree4.11e-6 globally and3.23e-9 active. A separate one-step response gives gamma=.00178439/M and omega=.02495442/M, close to the3M estimates. This supports weak oscillatory discrete growth; it is not solely an apparent fit to a transient Theta history.

The complex H/M/Q eigenfields also approximately reproduce the same rotation and amplification:

| diagnostic | complex norm ratio after3M | relative complex eigen-shape residual |
|---|---:|---:|
| signed physical H |1.00537596|1.13e-4|
| covariant M_i |1.00551724|6.79e-4|
| conformal Q^i |1.00539293|2.67e-4|

These combine independently mapped real and imaginary parts of the eigenfunction; a real evolution oscillates between them. The physical-constraint residuals, especially momentum, are less precisely converged than the complete-state mode. H peaks near(-.375,.375,-.125)M, M near(-.125,.125,-.125)M, and Q at an outer face near(-1.875,.125,-.125)M. Peak locations describe this discrete eigenfunction, not the first source of perturbation injection.

A real Ritz candidate near gamma=.00470/M remains **unconverged**: direct residuals are.00383 globally and.00791 active. It is not recorded as a validated eigenmode. Its rate could be relevant to the long control, but this32-vector experiment cannot establish that association or exclude other faster modes. Absence of an additional converged positive branch would not be a stability pass. No second solve, changed timestep, gauge, boundary condition, source variant or production job was added.

Reproduction: `arnoldi_sigma1.py`, then `validate_sigma1_modes.py`. Data: `covariant_constant10-arnoldi32-results.json`, `sigma1-validated-modes.json`, `covariant10-mode1-physical-constraints.npz`. Convergence plot: `sigma1-mode-convergence.png`/`.pdf`, generated by `plot_sigma1_convergence.py`. The earlier comparison plot intentionally continues to label its sigma1 bars as directional responses.
