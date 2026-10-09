# Held implementation proposal001

The eight copied held-plan files remain byte-for-byte unchanged. This source
proposal implements their equations, fixed cases and thresholds; it has not
been compiled, imported or executed. The independent plan review is a protected
input. A fresh exact root release is required for any scientific command.

`inner_gauge.hpp` is a private CPU helper with explicit scalar adapters for
double and the frozen field dual D. Every product scales its factors and each
dual product-rule term before final scalbn. A zero primal does not discard a
nonzero derivative. It records counts/exponents rather than suppressing errors.
Bounded k uses positive mantissas and integer exponents. At W=0 the tiny
numerator mantissa is divided before its final scaling, avoiding premature zero.
There is no universal correctly-rounded nonzero-k claim; all native/oracle
rounded hex values are retained. The declared exact k=.5/rounded-zero checks and
absolute/dual tolerances apply. The negative control intentionally erases k's
dual derivative, and a fixed nondegenerate case must detect that defect.

Genericity here is for live-field directional derivatives at a fixed external
reference position and fixed W/G0. The helper explicitly rejects a nonzero
reference-radius tangent. It does not claim to differentiate the spatial cutoff
or missing higher reference jets. Complete native spatial coefficient jets are
used as values, including every nonflat reference connection slot. There is no
double-only replacement of a live-field derivative.

The isolated alpha=chi=1e100 Lambda witness uses the independently simplified
core target G0*Lambda for its required relative check. No240/280-digit subtraction
of two O(1e297) terms is claimed to resolve that O(1e-3) answer. The second core
witness uses the positive closed target2 alpha^2 k^2 gradchi. Both are evaluated
independently at110 digits. The main high-contrast families keep240/280 digits.

`probe.cpp` has exactly eight fixed modes: reference, sources, coefficients,
core-witnesses, principal, duals, invalid and nonrepresentable. Each emits full
per-case JSONL rather than only maxima. The 20 dual directions use the unchanged
frozen Direction/Consistent lift. The all22 local C0 source call uses actual
ConformalRHS with live kappa1=10/alpha and k2=0 and the production analytic
Minkowski subtraction with reference kappa1=10. It adds the candidate gauge or
unchanged RWM gauge. This is an analytic-jet local source comparison; it contains
no native Cartesian derivative, upwind, KO, ghost or final-stage RK operation.

The raw22 order is alpha,chi,P,Theta,beta_xyz,gxx,gxy,gxz,gyy,gyz,gzz,
Axx,Axy,Axz,Ayy,Ayz,Azz,Lambda_xyz. The independent oracle verifies all four
gauge rows/derivatives and exact equality of every geometry row against the
same actual C0 kernel with baseline RWM. Input determinant and A-trace normals
are retained without a new undeclared numerical threshold. The complete fixed
FD sequence is retained and only its final level plus convergence/floor rule
gates acceptance. There is no independent continuum/source-physics assertion
from geometry-row equality alone.

The MP oracle independently evaluates the literal unfactored split, reference
identity residual and proposal correction. Reference, both connection oracles,
outer bit patterns, principal coefficients, fullsource rows, field-dual rows,
core closed targets, bounds/rejections and nonrepresentable classifications are
separate checks. All absolute/relative/scaled errors and near-zero targets remain
visible. The fullsource threshold remains2e-10, connection2e-11, finalFD5e-7;
there is no relaxed high-contrast tolerance.

The one-shot runner requires isolated unoptimized Python, bytecodeoff and all
three thread-count environments1. Standard-library exact source/prerequisite
guards precede the MP import. Unknown generated compiler dependencies stop the
attempt before scientific queries. Both unique binaries and exact compile
dependencies, command outputs, FD data and failures are retained. Root must
capture the outer invocation as well, so a pre-attempt authorization/input guard
failure cannot disappear. No retries into the same destination are permitted.

This stage admits no production/native integration, puncture/BH construction,
global operator, spectrum, time evolution or stability conclusion. Exact outer
RWM arithmetic retains its known stationary/scri limitations; this inner helper
does not repair them.
