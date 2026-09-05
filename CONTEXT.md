# PC-GH puncture hybrid terminology

This glossary records the distinctions agreed during the 2026-09-05
`grill-with-doc` planning session for the current cell-centered research branch.

## Language

**Finite inner relaxation**: A bounded coordinate-time damping rate that increases
smoothly near a puncture and acts continuously through the existing RK equations.
It is not a discrete projection.

**Reduction projection**: A discrete correction of the independently evolved
p/Q/L/B variables toward their defining spatial derivative targets. The agreed
hybrid fully resets the core and blends the correction through a smooth taper.

**Algebraic enforcement**: The existing conformal determinant and trace
corrections. This is distinct from reduction projection and remains enabled.

**Core and taper**: Fixed physical radii defining full correction and its smooth
transition to zero. Their units refer to the individual puncture mass; they are
not a fixed number of grid cells or an assumed causal boundary.

**Qualified hybrid**: A candidate passing consistency, constraint behavior,
resolution, mask-width, time-step, and long-evolution gates. Survival alone is a
partial improvement, not qualification.

Example: “Did projection stabilize the core?” — “Finite relaxation improved
survival; the discrete reduction projection still needs its taper-curl and
convergence gates. Algebraic enforcement was enabled in both.”
