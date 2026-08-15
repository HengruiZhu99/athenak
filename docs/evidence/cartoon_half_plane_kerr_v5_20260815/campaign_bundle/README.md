# Half-plane Kerr CUDA qualification v5

This frozen campaign builds pushed source `f95a0580` once with CUDA+MPI and
runs exactly three fresh spin-0.5 Kerr evolutions to `5M` at finest spacings
`M/32`, `M/48`, and `M/64`.

All three cases use AthenaK's explicit default moving-puncture gauge: advective
1+log lapse (`lapse_oplog=2`, `lapse_harmonicf=1`) and advective Gamma-driver
shift (`shift_Gamma=1`, `shift_eta=2`). Telegraph lapse, slow-start lapse, and
scale-invariant shift damping are off. Initial lapse is pre-collapsed.

The build and enabled CTest suite run in the allocation before science. The
three science cases use one node, four MPI ranks, four A100 GPUs, and eight CPU
cores per rank. Cases run sequentially and fail closed. The analyzer retains a
qualification claim only if all time-dependent constraint, axis-layer,
horizon-residual, reflection, and convergence gates pass.

V1 is preserved as an immutable allocation-free login-preflight failure: its
bare `/usr/bin/python3` was Python 3.6 and rejected the committed Python 3.11
analyzer syntax. V2 then bound `/usr/bin/python3.11`, but that bare interpreter
lacked NumPy. V3 loads the fixed NERSC `python/3.11-24.1.0` module and binds its
interpreter hash and NumPy 2.1.3. Scientific and resource contracts are
unchanged.

V3 passed login preflight and began a fresh CUDA build, then failed before any
CTest or science step because nvcc rejected captures first referenced inside a
Kerr initializer `if constexpr`. Source `258fbcfe` hoisted that common capture
expression ahead of the compile-time branch and added a static source guard.

V4 then built the full CUDA executable successfully and ran all 37 enabled
tests. Thirty-four passed, including the production-kernel CUDA tests and the
long restart-carrier test. Three failed before science: the Python environment
had SymPy 1.12 instead of the generator's pinned 1.14, its h5py extension was
binary-incompatible with NumPy 2.1.3, and the axis-parity unit test invoked a
host-created function pointer indirectly on CUDA. Source `f95a0580` changes
only that test to use a compile-time device-inline field-family selector. V5
also pins `python/3.12-26.1.0`, whose bound interpreter supplies NumPy 2.3.5,
h5py 3.15.1, and SymPy 1.14.0. No production equation, analytic data, gauge,
case, threshold, or resource value changed.
