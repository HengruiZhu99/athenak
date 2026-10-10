# Independent source002 review

The corrected recipe keeps the literal `clang++` invocation in both compiler
commands. It separately pins the resolved shared `clang` target, verifies both
hashes and the symlink target before compilation, and includes those inputs in
the final protected-input check. The basename is therefore retained where the
driver selects C++ linking. No compiler invocation was needed for this review.

The added guard is the only runner change; its removal reproduces source001
bytes and AST exactly. Recipe changes are confined to the C++ driver fields and
five correction-history filenames. The arithmetic header, probe, independent
Fraction oracle, 70-case registry and serialized inputs are byte-identical to the
prior reviewed source. Release/Debug flags, exact domain, thresholds, dependency
checks, isolated unoptimized Python guards and one-shot attempt scope are
unchanged. The runner requires an explicitly passed receipt binding this exact
source index and root authorization before its compiler or oracle calls.

The prior integer arithmetic review carries over through exact source equality:
136 limbs cover the bounded four-factor/128-generated-term domain; full dual
product-rule terms are retained; signed sums are accumulated before one
nearest-even rounding; exact out-of-domain sums are rejected before rounding;
and all consumed atoms are validated before zero shortcuts. The Fraction oracle
uses an independent pair recurrence and CPython Fraction conversion. This review
does not calculate those targets or qualify any unit result. The source001
driver finding and its unexecuted status remain explicit.

The resulting PASS is source/math/admission readiness for this standalone CPU
primitive only. It does not establish a unit result, repair a metric inverse,
adopt an RWM gauge, or upgrade the original far-dual failure.
