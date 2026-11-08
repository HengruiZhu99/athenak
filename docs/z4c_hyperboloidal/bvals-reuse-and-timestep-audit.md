# Boundary reuse and AMR timestep source audit

This is a bounded source audit, not an implementation change or a new numerical
admission. The linked evidence JSON records the exact inspected source hashes.

## Copy-compatible boundary subset

Aligned, same-level Cartesian interiors contain genuine one-donor rows. The
Cartesian selector finds the finest covering core leaf and snaps an aligned
sample to a length-one stencil in each axis (`mapped_halo.cpp:151`). For example,
a coarse core face point whose tangential coordinates stay inside the nominal
block and whose neighbor is another unrefined core parent has a direct aligned
neighbor donor. Fine siblings inside the central refined core have analogous
interior faces. A geometric label alone is insufficient: a nearby finer leaf or
independently evolved overlap may change the selected owner.

For shell children, same chart/layer/level and shared dyadic angular/radial grids
are candidates. Cross-layer interfaces are excluded because radial spacing and
endpoint offsets can differ. Across seams, charts differ. Even within one layer,
`Interpolate` tries Cartesian coverage first, then overlap and shell selection.
The shell branch evaluates angular/radial interpolation weights after floating
coordinate inversion; it does not universally snap aligned coordinates. Thus
same chart and same level do not prove a byte-compatible copy.

The precise safe routing predicate must examine the already selected, coalesced
row: exactly one evolved donor with coefficient exactly 1, selected owner equal
to the proposed native neighbor, matching level/chart/layer where applicable,
and the target outside the receiver's evolved region. Keep every other row on
the existing mapped path. Do not change owner priority, tie ordering, retained
small weights, radial plane selection or stencil bounds to enlarge this subset.
No count or performance benefit for this subset has been measured here.

Even that predicate only permits transport reuse. Current interpolation computes
`anchor + weight*(donor-anchor)`, which can normalize a negative zero; a raw copy
can preserve its sign instead. Keep the one-entry arithmetic in the unpack/fill
step if exact current behavior is required. Maintain 26 independent residual
fields and 31 raw physical fields as separate channels; dependent residual slots
are zeroed explicitly. No algebraic projection may be inserted in the raw halo
path used by physical KO.

## Minimal native infrastructure extension

Existing CC buffers derive indices from shared `Mesh::mb_indcs` with origin 4,
whereas the hyperboloidal arrays use origin 7 and block-dependent evolved overlap.
`InitSendIndices`/`InitRecvIndices` store one rectangular region per neighbor
index, shared across blocks. Simply passing the larger module array would read
and write the wrong indices. Shifting all indices by 3 fixes the origin only;
it does not cover tangential overlap or irregular selected-donor subsets.
The shell neighbor table already removes cross-chart and cross-layer entries.

The smallest conservative CC route is an optional, module-specific descriptor
for selected same-level neighbors with an explicit storage-origin offset and a
precomputed destination-row mask/list. A whole rectangular buffer is eligible
only when its needed rows all satisfy the predicate above. Otherwise retain the
mapped rows. The same mask/count must control metadata construction, receives,
packing and unpacking; do not mutate the shared mesh neighbor table. Preserve
native rank batching, peer aggregation, posted receives and incomplete polling,
and invalidate descriptors with the native mesh sequence. Field selection and
the existing one-entry arithmetic need explicit support. This extends the native
communication infrastructure; calling the current CC API on unchanged indices
is not a valid adapter.

For irregular one-donor subsets, a transport component shared by CC and mapped
boundary classes is cleaner than expanding CC box assumptions to arbitrary
points. The existing `bvals::MappedExchange` already uses absolute module
indices, native block ownership, persistent peer buffers and nonblocking task
polling. Sharing that transport implementation is a code-reuse opportunity;
it is not evidence that replacing the mapped selector or its interpolation
arithmetic improves performance. Keep this work outside the frozen N12
qualification until a separately validated change is justified.

## Fixed-step AMR admission, not an adaptive controller

The objective file requires preserving the existing timestep. The available
predecessor AMR driver does not change dt after every cycle. In configured fresh
nonproduction mode, it can select dt once at initialization. For restart and
after an accepted regrid, it calls `SelectTimestep` and rejects a step above the
returned cap. With continuation=true the cap includes `.4/rate` and `.005`, plus
an optional angular-dissipation bound (`coupled_mpi.cpp:1440`). The post-regrid
check is explicit in `amr_evolution.cpp:777`; it does not replace dt with a new
value there.

Native `NewTimeStep` validates classical RK4 and the relaxation limit, then sets
`dtnew=requested_dt`. It records characteristic CFL through the normal RHS, but
has no counterpart to the old AMR post-regrid characteristic admission. This is
a remaining **fixed-step AMR admission parity** item. “Mapped adaptive timestep”
would incorrectly imply a new adaptive controller is required. The source
comparison is with the available predecessor AMR helper, not a recovered
historical N12 dynamic-AMR binary.

The distinction is observable in existing bounded CPU evidence: refined cycles 2
and 3 reach CFL approximately 0.4767786203 at both two and eight ranks, while cycles 1
and 4 remain about 0.2376923023. Those tests intentionally establish current native
fixed-step transfer/evolution consistency and have no new CFL rejection gate;
they do not establish parity with the predecessor AMR admission policy. Do not
relabel them or silently change their dt. The pending GPU AMR bundle has the same
bounded consistency scope. Fixed N12 uses a fixed SMR tree and its pinned device
policy, so this separate AMR parity item does not alter or block that comparison.
