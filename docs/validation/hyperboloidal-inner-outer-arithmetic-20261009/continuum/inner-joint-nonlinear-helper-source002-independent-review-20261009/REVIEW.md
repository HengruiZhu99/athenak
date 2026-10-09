# Independent declaration-only source002 review

PASS for the narrow source correction and mechanical rebinding. This is not a
successful compile or an execution authorization. Source002 still requires the
new exact root release and its actual Release/Debug gates.

The source001 compile failed before any scientific query, with two instantiations
of the same error: `Parts(q)` returns `std::array<T,8>`, while `Fields(f)` returns
`std::array<T,4>`. One `auto` declaration cannot deduce both types. My earlier
source review missed this C++ type constraint. Its mathematical analysis remains
applicable, but its no-source-blocker assessment did not establish compilability.
The original review, failed source, compiler stderr, child receipt and root outer
receipt are preserved without rewriting their conclusions.

Independent byte comparison confirms that probe.cpp differs by precisely one
replacement:

```cpp
const auto qp=Parts(q);const auto fp=Fields(f);
```

Each declaration now independently deduces its intended array type. Evaluation
order and all subsequent uses are unchanged. No helper, source equation, oracle,
lift, case, query count, tolerance, FD rule or compiler option changed. Twenty
copied files declared byte-identical were independently compared with source001.
Other multiple-auto declarations in the inspected probe have matching return
types; no additional mixed-array declaration was found by source inspection.

The new recipe equals the original after only the fresh private-prefix path
replacement and the four explicit correction/prior-failure provenance fields.
The unchanged runner resolves its own local recipe/index, so the old release
cannot authorize this fresh source. Its optimization, runtime, dependency,
failure-preservation and postcheck behavior is unchanged. The original preparation
script is explicitly historical; the new preparation script performs only
copy/hash/JSON/diff work and is not a scientific caller.

All 35 indexed files and 2,550 unique protected inputs were checked before and
after this review. The original source001 indexed files were separately rehashed.
The preserved compile stderr exactly matches the declared failure hash and shows
only the array8/array4 auto errors, for double and D. No candidate import,
compiler/syntax check, CAS, numerical calculation, array load, kernel query,
operator or evolution was performed in this review.

The source001 mathematical and validation limits carry forward unchanged:
declared fixed families only; far-from-reference Dchi/Dalpha cancellation and
arbitrary absolute-dual seed accuracy are unverified; exact W=1 frozen RWM
retains its existing limitations. No native/BH/stability conclusion is admitted.
