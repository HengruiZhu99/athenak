# Ordinary executable AMR restart gates

These gates precede the first run of `amr_restart.py`. The runner uses the actual
native executable and ordinary dynamic adaptive input. It neither replaces main
nor bypasses the adaptive admission gate, changes initial fields, or supplies
another evolution driver. Run it only after the remaining AMR admission gates
allow ordinary adaptive evolution.

Start with the preserved N12 reference fixture, central SMR patch, KO=.1,
chi floor=1e-10, classical RK4 and exact dt=0.00046944203735617445. Add the native
location criterion at (.99,0,0), radius .005, maximum physical refinement level
one and zero refinement waiting interval. It must refine only the outer +X
parent after accepted cycle 1. The native tree grows from 106 to 113 leaves; the
surface grows from 6 to 9 patches. There is no derefinement criterion in this test.
All load-balancing costs must remain one.

Compare an uninterrupted four-cycle run on two ranks with two cycles on two
ranks followed by a shared-file restart on eight ranks and two more cycles.
Use dcycle=1 native restart outputs. Because Driver output precedes each AMR
call, the regular cycle 1 checkpoint still has 106 leaves; cycle 2 and later have
113. The split checkpoint is therefore already on the refined tree. Final
outputs may duplicate a cycle; their complete payloads must match the regular
output at that cycle exactly.

Audit every checkpoint's logical locations, shared-file rank metadata, record
stride, atlas/physics identity, exact dt/time and equal costs. Independently
enumerate the original 106-leaf topology and the eight children replacing the
outer +X parent. Enumerate every active bulk/surface slot from those logical
locations and the N12 overlap policy; require 241488 bulk points and 1296 scri
points on the refined tree, finite all 31+8 active values, and zero unused
padding. Chi floor/nonpositive/nonfinite event counts and corrections are
expected to stay zero over these four reference-initialized cycles.

The existing `restart_roundtrip.py` parser, payload accessor and byte comparison
check the complete 31-field/eight-record payload, including padding. The split
cycle 2 checkpoint must equal the uninterrupted cycle 2 checkpoint; resumed
cycles 3 and 4 must equal uninterrupted cycles 3 and 4 bit for bit. Preserve source,
native object and executable hashes and reject a changed artifact during the
run. This establishes bounded AMR restart continuity across decompositions;
it does not establish long-time stability, convergence or GPU performance.
