# Independent guard-only v3 review

PASS for source-only execution-admission review. The reviewed v3 source index
is ea06abf5a285ede3fb4224904a7b8932f2664dbc07c5b31718c19353570c7b4a;
the recipe is 9123c2182d8df050f4fe41a8545667c3d1626d3ed40149cda3aa22ceb0a4a4a5.
The original v2 conditional review remains preserved with its uncontrolled
Python-optimization admission failure.

run_gate.py now rejects sys.flags.optimize!=0 with an explicit if/raise
inside its checkpointed exception/finalization wrapper. It records optimize
and isolated flags before argument parsing. check_exact.py and analyze.py
both reject optimized direct execution before their scientific main entry.
Their pinned child invocations, and the proposed runner launch, use -I:
inherited PYTHONOPTIMIZE and PYTHONPATH cannot disable the assertions or
redirect Python module resolution. Explicit command-line optimization is
also rejected by the direct-entry guards. These changes close the reported
guard gap without running any optimized or normal scientific command.

The saved optimization-guard-only.diff agrees with all three source edits.
probe.cpp, gauge_proposal.hpp, reference_wave_map.hpp, PLAN.md,
PREPARATION.md, COUNT-CORRECTION.md and count-only.diff remain byte-identical
to v2. All prior scientific recipe settings remain unchanged; the sole new
recipe field records guard_revision. The 118 actual20 cases, 18 Fraction
cases, tolerances, literal rows, explicit basis/inverse and Release/ASanUB
raw-output byte equality have not changed. The preceding mathematical
source review therefore carries over, rather than being rerun as science.

All 1,454 unique declared input paths were rehashed unchanged. No proposed
module was imported, no compiler/Python executable invoked, and no CAS,
numerical calculation, kernel query, array load, evolution or production
source change occurred. This review does not itself release execution or
claim an actual20 numerical pass, nonflat source/fixed-point gate, lower-order
stability, puncture or black-hole admission.

One mechanical metadata-reader failure is preserved in history: the first
recipe comparison iterated over new v3 keys and indexed v2's absent
guard_revision key. It raised KeyError after successful pin and unchanged
file verification. A corrected comparison over the prior v2 keys passed and
reported the sole additional guard_revision field. No candidate was imported
or executed, and no source/threshold changed in response.
