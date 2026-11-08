# N12 common-time history and horizon comparison

This offline comparator does not run evolution or change runtime output. The
pinned original FastFlow CSV has SHA256
`1a9cad154bf199e0e28f25384ad5185c17e85066d2afd337cc366b45f4a84435`.
Require all11 reference cycles through5M:
0,356,711,1066,1421,1776,2131,2486,2841,3196,3551.
For every cycle require native shared-checkpoint metadata admitted by the
strict same-platform sidecar, exact repeated-addition nominal time/dt and the
unchanged fixed-origin FastFlow settings. All ordinary history rows are finite,
have positive full/exterior volumes and nonnegative RMS values. Additional
nominal-step output rows are retained separately; they cannot substitute for
a missing common state.

Each of full/exterior H,M,Theta,Z uses the adopted
`5e-10+1e-6*abs(reference)` RMS gate. Time0 roundoff diagnostics remain separately
reported, with no convergence claim. Native summary area gives areal radius
`3*sqrt(area/(4*pi))` in M units and irreducible mass equal to half that radius;
both use `max(1e-10,1e-7*abs(reference))`. Compact mean/minimum radii, expansion
RMS, summary total mass and shape coefficients remain reported diagnostics;
no previously failed extra compact-radius gate is silently repurposed.

Native summary time is printed with `%g`, so its integer cycle must match an
exact-time `.hst` row and its displayed time must match six-significant-digit
formatting of the exact coordinate time. This is not permission to round the
history/checkpoint time. Every summary must have a corresponding complete169
finite harmonic-coefficient shape row (lmax12).

Ordinary native output does not serialize explicit found/complete flags or
per-point mask membership. Successful history generation calls the reviewed,
hash-bound `Diagnostics::RequireAll(found && complete_surface && surface.Valid)`
guard and computes its exterior mask with the full harmonic surface. The
comparator records successful status as this source-verified inference, backed
by finite positive area/mass and complete shape output. It does not invent
mask bits or claim exact mask comparison. The pinned historical successful
horizon statuses remain required.

Retain every initial/final/restart duplicate row. Each must independently pass
the reference gates, and every same-time pair must be consistent under the
same adopted RMS/areal/mass gates. A favorable duplicate cannot hide a failed
one. No compact-radius equality requirement is introduced for restart searches.

`n12_long_history_controls.py` uses a small synthetic11-time fixture to verify
parsing and rejection of missing common states, late exterior-Z errors,
areal-radius errors, inconsistent duplicates, wrong exact/printed times and
incomplete harmonic shape records. This validates tooling only; no long native
trajectory or active-floor behavior is inferred. Full-state/chi-ledger
comparison remains the separate checkpoint tool, with historical per-cell chi
records explicitly unavailable.
