# Driver invocation correction only

Source001 remains byte-exact, held and ineligible after a precompile source
review found loss of the C++ driver basename. This is not an actual compile
failure. clang++ and clang share a binary, but argv[0] selects driver behavior.

Fresh source002 invokes literal
`/Library/Developer/CommandLineTools/usr/bin/clang++`, binds the contents at
that invocation path and its resolved clang target separately, and verifies
both before and after through the protected set. The runner explicitly checks
the invocation spelling, resolved target, and both content digests. Its
command array retains clang++ rather than a normalized resolved path.

Header, probe, Fraction oracle, all70 registry/case bytes, mathematical PLAN,
IMPLEMENTATION, authorization schema and original preparation record are
byte-exact001. The runner differs only by eight source-level guard lines plus
its comment; an exact reverse text/AST proof is saved. Recipe/external/readiness
metadata rebind this fresh source/index and preserve all001 files as pins.
No compiler, target arithmetic, oracle, query or numerical import has run.
