# Half-timestep continuation failed

Job 8984609 is terminal (PBS F, exit1), independently verified, and released.
The same V7 n16 formulation at dt=.00025 failed at coordinate t=.96975
(2.90925M), versus .9695 (2.9085M) at dt=.0005, at the same Cartesian cell1143.
The captured intermediate-stage lapse is -2.1596772971199663e-5. No lapse floor
was applied. This strongly disfavors a timestep-size explanation of the failure.

The last fresh diagnostic is at t=.95; the final report explicitly marks its
diagnostics stale. Its positive diagnostic minimum lapse must not be confused
with the negative lapse in the failed stage. Horizon acceptance is lost from
t=.65 onward. Scri surface/ghost null residual maxima remain below1.3e-15.

The preserved parent at t=.5 remains a valid checkpoint, but this failed
continuation is not admitted for restart. Verification replays all47 scientific
gates and checks raw artifact hashes, all four checkpoint arrays, source, binary,
plan, input and scheduler identities. Scientific acceptance is false.
