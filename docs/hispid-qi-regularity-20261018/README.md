# Moderate QI checkpoint: migration and horizon validation

The new axis-tau checkpoint exposed an outdated Python proof whitelist before
any finder work. Commit e54102e5 admits the explicit C4-v4 and C2tauC4-v5
identifiers, preserving exact producer/consumer basis, maps, source image,
checkpoint and sampled-field bindings. Unknown versions and nonfinite or
out-of-range maps remain rejected. Four provenance regression tests pass;
invalid-map tests also match those invalid values in the proof and workers,
so a simple map mismatch cannot hide a missing range check.

Perlmutter jobs 59957080/59957101 use the existing executable, SHA256
ff751123adcf4ea043c2b62d1f55ba10e5e966f462d7c606ef7b01e4decc57a5.
The first job's 2454-point native sampler migration passes with identical fields
and metric gradients. The second job reuses that proof and passes harmonic
orders 8/12/16 plus an independent quadrature increase. ADM/Z4c import error
is 4.083e-16, with no evolution. Final horizon quantities are:

| Hole | Horizon mass | Irreducible mass | Coordinate spin chi | Expansion RMS |
| --- | ---: | ---: | ---: | ---: |
| + | 0.615673713 | 0.604595771 | 0.370895278 | 2.041e-8 |
| - | 0.416086392 | 0.409618329 | 0.345813441 | 5.626e-9 |

After the observed angular refinement buffer, continuous enclosure margins
for the modified balls are 0.03205/0.01867. Across all consecutive angular/quadrature pairs, mass and spin-vector changes
are below 2.01e-12/2.79e-11. These are coordinate rotation integrals, not AKV
spins. The checkpoint remains diagnostic: the fixed native QI 160/192/224
sequence meets absolute exterior constraints but fails its strict bulk RMS
trend. No trumpet or full-binary qualification is inferred from these horizons.

Full inputs, logs, raw migration witnesses and harmonic coefficients are
retained on the TwoPuncturesC branch codex/hispid-trumpet under
validation/trumpet/regularity-family-control/horizons/. Native solve/observer
records and the updated PRD-style report are retained alongside that evidence.
