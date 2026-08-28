# AthenaK gauge implementation audit

## Mapping

AthenaK stores the Z4c trace variable

```text
Khat = K - 2 Theta,
```

so the physical mean curvature used by Bona-Masso slicing is reconstructed as

```text
K = Khat + 2 Theta.
```

`src/z4c/z4c_calcrhs.cpp` implements the reference lapse as

```text
rhs(alpha) = beta^i partial_i alpha
           - (alpha^2 + lapse_shock_avoiding_kappa) (Khat + 2 Theta).
```

For this campaign:

```text
lapse_shock_avoiding = true
lapse_shock_avoiding_kappa = 1
telegraph_lapse = false
slow_start_lapse = false
lapse_oplog = 0
lapse_harmonic = 0
shift_mode = prescribed_zero
```

The prescribed-zero mode zeros all evolved shift components at initialization
and every RK update.  Gamma-driver coefficients are also set to zero in the
campaign input for fail-visible provenance, although `shift_mode` is the
authority.

## Defect found at the base commit

At base commit `c67edf54ce97536ad9410adcf6029a4478038139`, the lapse RHS
already matched the continuum equation and used `Khat+2*Theta`.  However,
both passes through `z4c_newdt.cpp` rejected `alpha<=0`, and the stage-state
admissibility scan also required positive lapse unconditionally.  This would
terminate a mathematically valid reference evolution at the first zero
crossing.

The narrow correction:

1. keeps positive lapse mandatory for every non-shock-avoiding gauge;
2. allows any finite lapse for `lapse_shock_avoiding=true`;
3. uses `abs(alpha)` only for the physical light-speed CFL magnitude;
4. retains `sqrt((alpha^2+kappa) gamma^{nn})` for the gauge speed;
5. leaves the lapse RHS unchanged and unregularized.

The default-off behavior and every other gauge family remain unchanged.

## Focused host verification

The required focused tests are:

```text
athena.z4c_state_admissibility
athena.z4c_timestep_contract
athena.z4c_shock_avoiding_gauge_static
```

They cover negative-lapse state policy, negative-lapse light-speed magnitude,
the exact shock-avoiding RHS source, `Khat+2*Theta`, and the production
timestep source contract.  Perlmutter CUDA execution is still required before
the N256 production gate.

## Remaining qualification boundary

This source audit does not prove Figure-3 reproduction, stability, or
convergence.  The N256 run is the first scientific gate.  N128/N512 replay is
forbidden unless N256 reaches the published first peak, descent, deep minimum,
and rebound within reasonable numerical error.
