# Clean intrinsic reduction implementation status

2026-09-07. Work in progress. No new evolution or qualification claim.

The remote source was fetched at `62945657b4f2828a481abb5e7708e0e3e06dbd8d`,
exactly the independently verified SHA. The requested new branch
`codex/pcgh-clean-reduction-transfer-20260907` was created there, then fast-forwarded
to the originating checkout's committed descendant `249929fd` after inspecting
the intervening commit inventory and production diff. This retains the direct
lapse-gradient helper, tracker and CUDA fixes, and negative qualification evidence.
The originating checkout has dirty analysis and extensive untracked raw evidence;
none of those files was modified or merged. The parent main checkout is untouched.

Separate detached legacy control: `../athenak-pcgh-legacy-control-20260907`,
commit `b81b44d658f3b81584e94ce79b92656c112ff908`. It is an evidence-only checkout.
The mathematical reference is detached at
`7ef9c61c0c2bd12a46b28f334a53a1aabcd1842e` in
`../pcgh-candidate-reference-20260907`; its files are not changed by tests.
The candidate suite executes in `../pcgh-clean-reduction-tests-20260907/candidate-suite`
using a separate virtual environment. All eight original checks returned zero.
This is reproduction of candidate algebra/point jets, not independent compiled
AthenaK verification, a puncture theorem, or an evolution result.

The direct-lapse report is recovered at
`qualification-runs-20260905/direct-lapse-gradient/REPORT.md`. Preserve its failure
at 5.167923M versus 5.187818M, unavailable first-invalid intermediate state,
and scalar-max/stale-ghost limitations. The R16 20M ladder has anti-aligned field
differences and does not pass the binary gate. Existing transfer brackets do not
establish a causal projection-feedback explanation of late growth.

Code inspection confirms GH projection every stage, reduction projection only
on the final stage, followed by restriction/exchange/BC/prolongation. Both ordinary
and postprojection transfers require explicit coverage in the new operator work.
The inherited direct lapse helper returns **twice** the gradient: intrinsic l
must account for that convention and cannot reinterpret legacy L.

Della access works using the preserved multiplexed ControlPath. Initial occupancy:
A100-PCIE-40GB, 40960 MiB total, 3496 MiB occupied, 0% sampled utilization; no
user Slurm jobs. This does not imply the occupied memory is available. Scheduler
gputest reports a 15-day limit and heterogeneous GPU node types. No remote build
or evolution launched yet. Inspect occupancy and exact requested node resources
again before allocation. User authorization permits direct vis1 testing or up to
eight 80GB A100s through gputest; request only resources needed by each staged test.

Next: complete provenance/operator audit, freeze detailed fixtures and tolerances,
implement coherent transfer tests before changing bulk equations, then intrinsic
map and production kernel with full independent oracles.
