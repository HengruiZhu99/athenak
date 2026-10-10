This held standalone CPU gate tests an exact dyadic-integer backend and final
binary64 rational rounding. It contains no RWM/native/operator integration.
All three reviewed WIP sources are copied byte for byte: the131072-bit backend,
integer-only probe and independent fixed74 Fraction module. An additional
carry_range_units.py introduces exactly8 controls, for82 total. No candidate
module, registry, Fraction expression, compiler, probe or unit has executed.

The new controls are positive/negative pairs for minimum-normal minus half a
minimum subnormal (tie carry to minimum-normal), largest binary64 below1 plus
2^-54 (tie carry to1), MAX+MINsub (strong exact overflow despite IEEE rounded
maxfinite), and MAX-MINsub (admitted rounded maxfinite). They reuse the original
Fraction expected/canonical oracle, without changing its74-case registry.

Each finite/overflow row compares full canonical numerator and denominator
hex strings and final binary64 bits. All76 such rows must match;6 explicit
resource/domain negative controls require their declared nonempty failure
category. The registry and exact protocol are generated only after admission
and after both compiler dependency closures pass. Release and ASan/UBSan Debug
must each pass every82 controls and produce byte-identical native output and
identical complete oracle rows. This selected suite is not a global proof or
an admission of arbitrary future gauge expression capacities.

The canonical-object contract from the prior mathematical review is unchanged:
callers use binary64 atoms and canonical factory/operation outputs. Public raw
mutated objects or arbitrary int64 helper/constructor exponent arguments are
outside scope. Conservative4096-limb/131072-bit and +/-131072 normalized
exponent limits may fail explicitly. Exact magnitudes above maxfinite reject
before rounding; exact zero is+0 and nonzero negative underflow is-0.

The source review, completed70-unit actual compiler dependencies/runtime
inventory, audited source002 runner, its2400-input baseline, and whole-row
pencil are bound before compilation. The old debug MachO load commands are
read as metadata only, and available actual CLT runtime-library files are
pinned. System shared-cache library names are disclosed without pretending
they are separately file-pinned. No old executable is run.

The literal CLT clang++ path and its resolved clang target are separately
bound. CLT Python must run-I-B with optimization0. Fixed environment sets
PYTHONOPTIMIZE0, bytecodeoff, OMP/OPENBLAS/VECLIB1 and C locale; Python/compiler
include/library/SDK injection variables are removed. Both compilations use
the preserved flags and SDK. Generated-MD dependencies are parsed and each
external header must already belong to the frozen closed baseline; local
probe/header copies must match their indexed originals. Both lists must pass
before any probe or Fraction import. A newly discovered header is a preserved
mechanical admission failure requiring a separately reviewed additive attempt.

Every command has a60s cap (or the remaining internal120s budget). The separate
root outer launcher enforces a120s process-group cap and cleans that group even
after failure, capturing all stdout/stderr/true exit/elapsed/output hashes.
The one-shot child records each command before running it and preserves early
admission, compile, dependency, registry, probe and oracle failures. Static and
dynamic pins are checked before and after. Existing attempt destinations are
never reused. The root preparation/launcher remain unexecuted and require a
new independent exact-index driver review plus root source/math approval.

Source002 corrects only the JSON-registry structural guard, as documented in JSON-REGISTRY-ERRATUM.md. Source001 is preserved ineligible and unexecuted.
