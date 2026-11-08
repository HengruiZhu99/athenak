# Two-node selected-rank profile

This is a separate diagnostic experiment under the active two-node N12 goal.
It uses unchanged `f3a85db7…` from the reviewed cycle2486 checkpoint, first
unprofiled and then with unitrace on global ranks8 and23. Each run advances
four complete RK4 cycles to2490 with the original timestep, equations, output
cadence and saved FastFlow seed. Only nlim2490, tlim2 and ndiag1000000000 are
overridden. A30-minute, two-node allocation is proposed; Donor owns queue
checks and submission. No profiling executable is a throughput measurement.

Rank8 has the maximum static bulk point count; rank23 owns a complete outer
surface. These selections are workload hypotheses, not measured critical ranks.
The first pass uses audited absolute unitrace2.3.0 under baseline
oneapi/release/2025.3.1, without loading the PTI module. Host/device timing,
kernel submission, Chrome LevelZero calls, kernels and SYCL events are enabled.
Verbose shapes are requested. Hardware metrics and Chrome MPI/Testall logging
are disabled.4096 events is a per-thread buffer size, not a total trace cap.

A Python wrapper launches the same unmodified solver in both cases. On the two
selected ranks it samples only its own solver process tree at0.25-second
intervals, records actual PID/executable, rank/tile, selected environment and
loaded runtime-library mappings, and hashes mapped images after completion.
The child spawn preserves inherited PMI/PALS descriptors (`close_fds=False`),
with no extra open files. Monitor failure terminates and reaps its solver/tool
process tree and retains a failure record. The monitor is bounded to600seconds; the MPI process timeout is660seconds.
Nonselected ranks exec f3 directly after recording their binding/argv. Actual
loaded MPI, SYCL, C++ and GCC libraries must match pinned baseline bytes; the
profiled run must also preserve every runtime image observed in the control.
Tool/library injection and identities are verified rather than inferred from
module names. Any collection/ABI/finalization failure remains a failure.

The remote stdlib verifier compares all24,704,784 padded31-field/eight-record
scalars bytewise and checks finiteness. It compares initialized mesh metadata,
complete final parameter maps, all history/horizon output bytes, exact2490
time/dt, and all169 shape coefficients. Nine unused global coarse header
integers remain excluded and are reported. Frozen source/object/executable and
admitted checkpoint checks run before/after via the existing reviewed guard.
The control and profile start independently from2486; neither starts from the
other's final state. Source/runtime files are never changed.

Raw traces, checkpoints, outputs and complete manifests stay on Aurora.
A bounded20MiB compact archive contains terminal job evidence, commands,
runtime maps/hashes, qualification and event-schema/group summaries. It contains
no raw checkpoint or Chrome trace. The collector rehashes every raw artifact
before producing that compact proof. Raw failure artifacts remain available
remotely even when no success archive is admitted.

Initial analysis inventories actual categories, kernel names, argument/shape
examples, counts and local durations. Kernel functor signatures must be mapped
to the frozen source before attributing cost. All four bulk subsets share one
CalcRHS callsite. The normal solve and surface evaluation have separate lambdas;
the frozen Kokkos SYCL Range-reduction implementation has hash
`a3c7d632a7bd903ccd3d6cec270c11b1dcaf20eb90b5d3f84ea4aa58017f55c8`.
Its group-size rule suggests the144-point surface may launch one workgroup;
only actual trace shapes can confirm this.

The summary does not label arbitrary events as GPU work, assign RK boundaries
without evidence, align clocks across ranks, or sum overlapping maxima into a
critical path. Actual same-rank kernel and host wait intervals can reveal local
idle gaps once the tool schema is inspected. Missing MPI trace means MPI wait
or progress cannot be inferred merely from a GPU idle interval. A later bounded
follow-up is justified only by these observations and the independent unprofiled
measurements. The original timing object is unchanged; no per-rank timer
instrumentation or optimization is introduced here.
