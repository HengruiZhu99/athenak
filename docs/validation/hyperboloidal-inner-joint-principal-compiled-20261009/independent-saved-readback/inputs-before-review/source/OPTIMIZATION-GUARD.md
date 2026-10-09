# Optimization guard correction

The original v2 source/index/readiness and independent admission failure remain
unchanged. Its exact and saved-JSON checks used assert while the wrapper inherited
PYTHONOPTIMIZE. Thus an uncontrolled optimized interpreter could skip checks and
print passed=true. No v2 scientific gate was executed.

This fresh source-only revision rejects sys.flags.optimize!=0 in the outer
runner and both checker direct entrypoints. Python child commands use -I, so
inherited PYTHONOPTIMIZE/PYTHONPATH cannot change assertion semantics or module
resolution. The runner records its optimize/isolated flags. The recommended
root command also uses -I. The exact diff changes guards/launch flags only;
the probe grid,118-case registry,18 Fraction cases, source formulas, all matrix
thresholds and required Release/ASanUB byte equality remain unchanged.

No generated source was imported, compiled or scientifically executed during
preparation. Exact root review/authorization and independent guard review are
still required. This revision does not authorize a kernel query or evolution.
