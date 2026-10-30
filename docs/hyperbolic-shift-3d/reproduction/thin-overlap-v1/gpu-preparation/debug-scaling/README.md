# Thin-overlap scheduling on debug-scaling

At the user’s request, subsequent thin-overlap GPU tests use `debug-scaling`,
CompactBinaryMerger. The queue permits one-node jobs and walltimes from five
minutes to one hour (queried on 2026-10-30 UTC). This launcher retains the current
one-node, 12-rank configuration, with all 12 tiles checked by the runtime affinity
probe. The user allows up to two nodes; a 24-rank launch path and scaling test are
not implemented or claimed by this change.

A distinct job name (`hyper-scri-thin`) and queue-specific allocation lock keep
this task separate from the other agent’s preserved `debug` task. There remains
one active allocation per lane; same-lane races and uncertain submissions retain
their locks. Worker startup verifies the actual PBS queue matches the bound plan.
The original debug lane keeps its name and lock. Neither running allocations nor
frozen source trees are changed.

51 offline controller tests passed, including scaling queue rendering, exact
scheduler selection, queue mismatch rejection, separate locks, and terminal-only
release that preserves the other lane’s lock. No GPU efficiency or numerical
stability result is inferred from these launcher tests.
