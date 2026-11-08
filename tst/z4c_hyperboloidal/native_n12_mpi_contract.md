# Bounded N12 MPI attribution

This measurement-only pair runs the frozen native executable from the admitted
cycle-2486 checkpoint through cycle 2490 on two Aurora nodes, 24 tile-bound MPI
ranks. The ordinary run and the APS run use the same inputs, compiler/runtime,
timestep, equations and original Driver. No numerical source is changed.

The APS command uses `--collection-mode=mpi --stat-level=3 --mpi-imbalance=0`
on all ranks. Level 3 collects function, communicator and message-size
statistics. Imbalance mode 0 disables the tool's imbalance calculation; mode 2,
which inserts barriers, is excluded. These meanings are specified in
[Intel's APS documentation](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide-application-snapshot-linux/2025-0/controlling-amount-of-collected-data.html)
and the installed command-line options were checked before execution.

The existing external process monitor observes ranks 0, 8 and 23 in both runs,
preserving PMI descriptors. The actual executable, MPI, SYCL and C++ runtime
identities must remain unchanged. The APS executable, report tool and known
injection dependencies are pinned before and after the pair. The exact loader
cache exception from the previous profile remains explicit; all other observed
ordinary runtime images must still match.

Acceptance requires complete padded 31-field plus eight-record checkpoint
equality and ordinary history/horizon-output equality against the paired
ordinary run, the same source checkpoint and output controls, successful solver
exits, correct tile binding, frozen source/object guards, and actual MPI records
covering all 24 ranks with Isend, Irecv and Testall data. Tool launch success
alone is insufficient. All raw statistics, report commands, report exits and
generated tables are retained. An unexpected report format fails closed and
may be inspected offline without repeating evolution.

Reports cover the whole process, including initialization and final diagnostics.
They attribute MPI calls and communication; they do not directly establish the
RK4 critical path or replace unprofiled complete-step throughput measurements.
Communicator numbers are not interpreted as native exchange-plan identities
without matching creation order and payload evidence. No extra barriers,
observer hooks, per-call Chrome logging or globally replicated fields are added.

Each MPI case has a 660-second external bound and each report a 60-second bound;
the two-node allocation has a 1680-second bound. Raw checkpoints stay in the
private remote case. Compact collection retains their hashes and all MPI data.
