# Source layout and deferred checks

* `signed_products.hpp`: standalone integer-only product, signed accumulation,
  exact-domain comparison, final binary64 rounding, complete dual expansion.
* `probe.cpp`: fixed textual raw-bit input parser and exact input/result echo.
* `registry.json` / `cases.txt`: one fixed70-case registry in two representations.
* `fraction_oracle.py`: independent exact polynomial/Fraction target and audit.
* `run_gate.py`: fresh one-shot release gate, two compiles, two probes, two
  independent oracles; failures remain FAILED with original streams/trace.
* `recipe.json` / `external-pins.json` / `source-index.json`: exact runtime,
  source/context/dependency admission. `authorization-schema.json` is FALSE
  and cannot authorize a run.

No candidate import, C++ compilation, Fraction target evaluation, unit run,
scientific query or saved-array read occurs in source preparation. The only
preparation checks are standard-library text/JSON/AST/hash/size metadata.

The probe parses at most33 terms so the invalid public count32+1 control can
reach the API safely. It always stores four atom pairs; an arity5 control is
rejected before access. Invalid atoms in unused slots are intentional controls
of the stated consumed-domain semantics. For valid duals, generated-term
count includes zero derivative terms and must equal the sum of all arities.

Overflow/status must be tested before consuming result bits. On global dual
validation failure, the default result objects are not evaluated outputs.
The scalar/dual public interfaces are the reviewed API; `detail::Sum` is an
implementation function and its internal128 limit is not an enlarged public
scalar domain. No caller may claim an unbounded exact accumulator.

Correctness of passing doubles as memcpy storage on this native IEEE binary64
CPU and the integer compiler implementation remains part of the compiled
unit stage. The source explicitly forbids fast-math flags. The new primitive
does not call floating multiplication/addition, change a rounding mode, use
an FMA, or infer a derivative from a known seed label.
