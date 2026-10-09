# Failed half-timestep N24 physical-reference wave-map run

The original native process terminates with return code -6 at mesh time
0.79280598958281545, cycle 24355. Its stderr reports an invalid physical ADM
state, with negative lapse and chi at xyz=(.1375,-.9625,-.2291666666666667),
Omega=.0021701388888885099. Exact values remain in native.stderr.
The original command, executable/source/input binding, stdout hash, stderr,
before/after protected-input identities, output inventory and native history
are preserved. Source hashes stayed equal; the native process gate failed.

All copied sources/logs/receipts are byte-exact. Restart/visualization arrays
and any >1MiB console log are hash/size/origin metadata only. This capsule
does not replay arrays or perform a partial-field analysis. It does not turn
the failed t2 run into completed acceptance or identify a PDE/boundary cause.
Other cases in the fixed t2 matrix remain independently recorded.
