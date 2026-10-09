# Count-corrected held gate preparation

The actual tensor extraction and general-coefficient formulas are unchanged
from the first proposal. Only the count assertion/prose changes from262 to118,
as shown in count-only.diff. The first proposal and its source index remain
unchanged. ASSESSMENT-v2 is authoritative; the original displayed A_t beta
transcription and corrected Lambda term are preserved separately.

The separately added saved-JSON analyzer has no NumPy/CAS/eigensolver imports.
It checks the fixed118-case registry, independently writes all20 expected rows,
and constructs a full analytic left basis and explicit two-sided inverse.
Thus completeness is checked by inverse identities rather than a numerical
nullity cutoff at repeated roots. The scalar block is also transformed to four
wave pairs with speeds squared f,q,1,1. The Fraction checker remains byte-exact
from the first proposal, with18 rational cases and exact conjugacy/inverses.

The exact fixed tolerances are matrix maxabsolute2e-12; transformed scalar
maxabsolute1e-10 and per-entry scale max(1,abs(actual),abs(expected)) relative
1e-11; left-basis maxabsolute/scaled1e-10; both basis-inverse products maxabsolute
1e-10. No threshold is adjusted by this count correction. Original PLAN's
suggested repeated-root nullity check is superseded by the stronger explicit
inverse and all20 left-eigenfield checks, without a numerical rank decision.

run_gate.py requires root's future authorization with execution_released true,
the exact local recipe/source-index hashes and matching scope. It refuses a
nonlocal recipe before parsing it. The sole fresh attempt is attempts/gate001;
all compile/run/analysis errors and source drift are saved there. An already
existing attempt is refused without overwrite. Release/ASanUB use the pinned
compiler/headers/runtime, record actual compiler dependencies and executable
hashes, and require byte-identical raw matrix output. The authorization itself
is copied and protected after admission. A source-only preparation/readiness
record does not authorize this runner.

This is a constant-reference finite-positive alpha/chi principal test only.
No nonflat gauge fixed-point, lower-order source, finite-frequency, puncture,
black-hole, native or evolution gate is admitted. The later single-black-hole
goal remains wormhole-to-trumpet with the Minkowski hyperboloidal reference.

Preparation used standard-library file/hash/AST operations only. A metadata
read first tried the nonexistent historical recipe.json; the actual pinned
historical file is release-recipe.json. That read made no scientific call or
change. Generated gate modules have not been imported or executed.
