# Additive v2 admission correction, source only

Original candidate index9649d93e6f34a1c9e2b79c542509258b6a51af73eadfbfbc3cc9b7d1fc808146
and original launcher index3ca1a5ff remain byte-exact and have not been executed.
The independent source/math review e7858751e2fb4132c3e2523a5d956f1e192ea6812b3bf1dddf9a05cb8c1c37ea
found no formula blocker, but correctly failed standalone recipe admission:
the original main could consume a different --recipe while pinning only its
local derivative-recipe.json. That failed admission remains failed.

This fresh v2 requires the resolved consumed recipe path to equal the local
indexed derivative-recipe.json, hashes the bytes actually parsed, checks
that digest against its required local source pin, records it in the receipt
and protects the same resolved/local file in the existing before/after set.
No field, formula, quadrature, event, threshold, precision, count or full-gate
setting changed. The local recipe changes only held command paths and adds
original candidate/review context pins. PLAN.md and both mathematical source
dependencies are byte-exact copies. The full fixed493568-ray gate remains
held; no source import, syntax check, numeric/CAS/query or inverse work ran.

The independent review also confirms the cost includes163840 independent
coarea angular/radial node evaluations in addition to the493568 derivative
rays, with extra nested capped transition roots/radius/height work. Neither
number is measured runtime. A separate small timing gate will not relax or
stand in for any full-gate requirement.
