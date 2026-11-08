`bvals::MappedExchange` transports required donor vectors from native
`MeshBlockPack` ownership. It does not select interpolation donors, own a second
mesh, repartition cells, or allocate global field arrays.

Construction duplicates the supplied MPI communicator collectively. `Rebuild`
is collective and must be called on every rank, including ranks with no donors.
The native overload takes a pack and uses its contiguous local GID range,
`Mesh::rank_eachmb`, and `GetAMRLoadBalanceUpdateSeq()`. The current native mesh
has one pack per rank; the overload rejects other layouts rather than silently
assuming that another pack's block can be indexed into this pack's state.
Ownership rank numbers must refer to the exchange communicator.

A donor key is `(gid,k,j,i)`, where the last three indices are absolute indices
in the module's local storage. They may include the hyperboloidal evolved-overlap
padding. They are not flattened global-cell offsets. `fields` selects field
indices in a specified order and must agree across ranks. The constructor/plan
accepts all 31 fields or any distinct selected set, including the 26 independent
fields. The exchange does not infer field semantics.

`Rebuild` deduplicates requested donor keys in first-use order. `RequestOrdinals()`
maps each input request, including duplicates, to its unique donor ordinal.
`Values()(ordinal,field_slot)` contains local and imported values in that order,
and `Fields()[field_slot]` names the original state field. `UniqueDonors()` makes
the mapping explicit. Views and host mappings remain stable until the next rebuild.
A metadata-only rebuild overload supports testing without an auxiliary Mesh.

The stage sequence follows native boundary tasks:

1. `StartReceives()` posts one nonblocking receive per supplying peer.
2. `PackAndSend(u0)` batches all direct/local copies and off-rank donor packing in
   one device kernel, fences outgoing buffers before handing them to MPI, and
   posts one nonblocking send per requesting peer.
3. `Receive()` tests requests, returns `TaskStatus::incomplete` while needed data
   is pending, then batches the imported-vector scatter into `Values()`.
4. `ClearRecv()` and `ClearSend()` both must complete before starting another stage
   or rebuilding. They poll; neither calls an MPI wait or collective.

All collective metadata traffic occurs in construction/rebuild. Runtime transport
uses persistent peer offsets and payload-only `MPI_Irecv`/`MPI_Isend` on a private
communicator, following the native rank-packed boundary approach. MPI consumes
Kokkos device buffers directly, so a GPU build requires the same GPU-aware MPI
support as existing native boundary transport.

A receive buffer may still be read by the asynchronous device scatter after
`Receive()` has returned complete. The next `StartReceives()` fences that device
work before MPI can overwrite the buffer. Rebuild/destruction also preserve view
lifetimes. These are local device dependencies, not rank collectives. GPU event
polling or overlap improvements need separate profiling and validation; the
current tests establish Serial/MPI behavior only. An output view is ready for
subsequent work on the default execution space after `Receive()` completes.

`exchange_tests.cpp` covers exact selected-field payloads, duplicates and stable
ordinals, self donors, zero-peer and empty-request ranks, poisoned unused storage,
repeated stages with changing data, reversed rank/GID ownership on rebuild, and
an empty plan. An independent control-channel handshake deliberately withholds a
peer send until another rank proves `Receive()`/`ClearRecv()` return incomplete.
The test therefore exercises actual incomplete MPI polling, not only an early
call before packing.

Validated with current native Serial Kokkos/Open MPI configuration:

- 1 rank: 660 exact comparisons.
- 2 ranks: 1,942 exact comparisons.
- 3 ranks: 3,222 exact comparisons, including a zero-peer rank.
- MPI-disabled compilation and Serial execution: 660 exact comparisons.

The tests do not yet establish a coupled hyperboloidal RHS, interpolation
accuracy, refinement donor eligibility, Aurora GPU transport, or performance.
