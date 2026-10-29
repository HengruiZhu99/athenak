# V6 offline diagnostic parity and matched 1D reference

Offline checkpoint reanalysis now explicitly selects the seven-node bulk
radial derivative for V6. Previously its constructor and cell diagnostic loop
would have used the legacy radial operator. The exporter supplies and checks
the operator selector; an older analyzer cannot silently process a V6 request.
V1–V5 retain their existing invocation. Ray diagnostics remain independent
interpolated jets, with this distinction recorded in the exported metadata.
This changes offline diagnostics, not the frozen source-v13 evolution job.

The collected four-step n16 puncture checkpoint from Aurora job 8984220 was
exported with worker/source binding and reanalyzed locally. Normalized
coordinate-volume RMS values agree with the native GPU diagnostics:

| Quantity | Reanalyzed RMS | Absolute difference from native |
|---|---:|---:|
| H, whole volume | .04794012017650396 | 2.78e-17 |
| M, whole volume | .05767082469458413 | 1.39e-16 |
| H, exterior | .024830170348307947 | 7.63e-17 |
| M, exterior | .05767817169886768 | 1.39e-16 |

The exterior cutoff is the native contemporaneous mean horizon radius.
M uses the Penrose/conformal metric norm, not the radial component or the
physical norm. This checks diagnostic implementation parity, not evolution
accuracy. The export completed in 9.35 s, within its 250 s analyzer / 400 s external
bound. The 82 existing offline regressions also pass. Source, executable,
input and output hashes are retained under [reanalysis](reanalysis).

Reproduce the final check after exporting the complete checkpoint with the
updated `checkpoint_export.py --reanalyze`:

```sh
python3 tst/hyperbolic_shift_3d/check_reanalysis.py \
  --export output/hyperbolic-core-shell/v6-reanalysis-check-v1/export \
  --out output/v6-reanalysis-parity-new.json
```

A fresh **unchanged N513 1D donor** run reached coordinate t=1=3M in 158.04 s,
with dt=.025/512 and 20481 steps including the final remainder. Its library
SHA256 and 22 dependency source hashes match the preserved donor used for
the previous comparisons. The run used a 400 s native / 420 s external bound,
one process, fresh output, and no restart. The controller verifies that the
sources remained unchanged. Its final local-polynomial apparent-horizon
radius is **.07507475825193644**, areal radius **.6666666666843031**, and mass
**.33333333334215154**. Its null residual is 2.22e-16.

[Initial/final profiles and reports](donor/n513) provide the matched-time
reference for a future successful V6 puncture duration. The full 1D state and
compensation archive remains in local ignored output and is hash-bound in
the execution record. This is a 3 M reference, not a new 100 M validation or a
claim that the 3D puncture has reached this time. The 1D reported exterior
radial constraint RMS must not be equated directly with the 3D invariant
momentum norm without the documented metric conversion.

The fresh endpoint also reproduces the archived 100 M campaign at its saved
t=1 row: horizon-radius difference 4.25e-14, mass difference 1.05e-15,
and exterior H/M radial RMS differences below 4.8e-15.
[archive-comparison.json](donor/archive-comparison.json) binds that history
and retains every difference; this is not a full-state bitwise replay.
