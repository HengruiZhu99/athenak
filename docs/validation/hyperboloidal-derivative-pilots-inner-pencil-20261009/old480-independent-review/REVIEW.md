# Independent tiny timing and outer-harness source review

PASS by source/math inspection, with the partial-group retention limitation
below. No source was imported or executed, no syntax/CAS/numerical test was
run, and no scientific arrays or rays were read or queried. This review is
not execution authorization, angular accuracy or full-derivative acceptance.

The timing candidate48486f9a, its outer690f4ee7 and the full-v2 outer4281a38d
were copied and pinned before their contents were read. All indexed source
bytes and113 unique dependency/package/runtime paths were rehashed unchanged.
The timing derivative_core is byte-identical to the reviewed v2 main; no
mathematical formula or full-gate setting was changed.

The tiny recipe has three fixed events (initial origin, future transition,
short-time exact outer), two fixed boosts,60/80 digits and2x4/4x8 sphere
samples. This gives480 single-ray jets and24 groups. The4744 checks count
correctly:3360 local metrics,480 Hessian-symmetry,24 Lorentz,160 exact initial
zero-value and720 fixed-ray precision block checks. All four fields, four
gradients and ten Hessians are retained. The initial-origin branch avoids
radial divisions; the other two events cover capped transition-root work and
the analytic outer branch. Root width1e-50,512 iteration caps and the stated
1e-40/1e-35/1e-30 pilot gates are separate from the held full80/110-digit
493568-ray gate.

Every ray uses fixed k during its analytic differentiation. The two boost
grids provide distinct ray samples, not two completed angular quadratures.
The source correctly claims local identities/timing and fixed-ray precision
consistency only. There is no scalar wave-trace or angular convergence claim,
no inverse/native target-time/Jacobian test and no evolution. Per-ray timing
isolates ray_integrand; group timing includes sampling and bookkeeping and
does not include the preceding NativeGraph construction/height-prefix work.
Full runtime extrapolation would therefore remain conditional.

The child binds the consumed resolved timing recipe/hash to the authorized
local recipe, enforces exact source/output/runtime/environment pins and
rehashes after execution. The standard-library outer verifies all candidate
pins before launching at most once, binds exact argv/paths/counts/cost,
captures streams and source copies, and requires the exact child success,
scope and unchanged-output declarations. The independently generated outer
diff contains scope/path/source-count/expected-count substitutions only;
one-shot, drift/failure and protected-stream logic are retained. The full-v2
outer's100/100/28 rows and2360 checks also match its unchanged core loops.

Preservation limitation: partial-rays.json is written at completed-group
boundaries. If a ray raises midgroup, previous groups and the traceback are
retained, but already completed rays in the unfinished group are not saved.
This is not a formula/admission blocker for a bounded timing pilot, but the
record must not claim persistence of every completed ray after an exception.
A later source-only additive persistence improvement could write on exception
without changing any mathematical setting. Existing sources remain unchanged.

Positive local denominators and these tests cannot establish angular
integration accuracy, global gauge regularity, coordinate-map invertibility,
native stability or later wormhole-to-trumpet survival. The Minkowski
reference remains fixed for that later requirement.
