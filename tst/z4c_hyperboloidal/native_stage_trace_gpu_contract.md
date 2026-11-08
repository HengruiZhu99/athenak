# Native GPU stage observer contract

This test translation unit replaces main and links the frozen native runtime
objects. It uses the original Driver initialization, classical RK4 and native
TaskLists. There is no legacy CPU or GPU evolution operator in the executable.
Only observer callbacks are appended to existing before_stagen, after_stagen
and after_timeintegrator lists, depending on all original tasks in each list.
No scheduler dependency is removed or reordered.

Runtime: `native_stage_trace_gpu INPUT clean|poison`, output in current directory.
The input is the fixed N12 native fixture with dynamic classical RK4 and nlim8.
Both modes initialize the same fresh state and perform eight complete steps.
Poison mode overwrites all RHS and residual workspaces plus only bulk u0 cells
outside each block's inclusive evolved bounds with NaN before every stage.
The owned work list is checked against those bounds, whose point count must
match; evolved overlap is preserved. rhs_status is reset to a sentinel. Every
stage must overwrite all owned RHS values and statuses with valid finite
results. No owned state, RK base, accumulator or history is deliberately changed.

The first four stage inputs and outputs include all 31 fields and all eight
ChiRecord members. The first full bulk/scri RHS is captured after_stagen.
This is the just-consumed RHS: ExpRKUpdate reads u_rhs/s_rhs and writes only
state/accumulators/history; the following ProjectSurface writes s0, and the
remaining clearing tasks retire communication buffers. These functions do not
modify the RHS arrays. No extra RHS evaluation is performed for tracing.
Accepted states0/1/8 and mapped geometry are also saved. Initialization snapshot0
is taken after Driver::Initialize has filled the native boundary state.

Each of the32 stages records cycle, stage, RHS/state time, global maximum
compatibility and clock residual, global minimum owned bulk/scri chi and maximum
absolute owned RHS. Ordering, exact requested dt, eight accepted cycles, finite
owned fields/RHS, compatibility<=1e-7 and minimum chi>=floor are required.
Zero-surface ranks participate in all callbacks, reductions and empty dump files.
These diagnostic transfers make the observer unsuitable for performance timing.

## Binary schema

Files are `LABEL-bulk-rankR.bin` and `LABEL-scri-rankR.bin` for every MPI rank,
including zero-row scri files. Required labels are geometry; accepted0,1,8;
stage1..4-input/output; and first-rhs (13 labels total).

Header:16 bytes `NATIVE_TRACE_V1` plus one NUL; uint64 localcount; uint32[4]
`{6,nvalues,region,1}`. The header is40 bytes. Region0 is bulk,1 scri. Each row
writes int32[6] keys followed by nvalues float64 values, explicitly with no
padding. Keys are canonical ordinal, native gid, k-origin, j-origin, i-origin,
chart face. Bulk ordinals are native GID/work-list order and include evolved
overlap. Scri ordinal is face*144+k*12+j; scri i is12. Values are xyz for geometry
(nvalues3); stored31 fields for first-rhs; or31 fields followed by
`minimum,accepted_minimum,floor_count,nonpositive_count,nonfinite_count,max_correction,sum_correction,first_floor`
(nvalues39). Storage is little endian on the tested platforms.

The comparator requires complete unique canonical coverage of229392 bulk and
864 scri points in every label, finite values, exact key identity and32 valid
stage records. It checks stage1 input against accepted0, each subsequent first
step input against the preceding output, and stage4 output against accepted1.
With a reference run, all geometry, fields, RHS, histories and stage records
must be byte-identical for clean/poison and native24/96 comparisons; differences
are reported without relaxing the gate. Original-GPU versus native scientific
comparison remains a separate existing tolerance-based test.

The portable builder derives compiler, definitions, include paths, FP/device
flags and link objects from the frozen native CMake build. It replaces only
main.cpp.o, builds with make-j64, and records source/object/executable hashes.
No runtime edits or scheduler submissions are made by this test package.
