# N16 failure mechanism: zero chi contaminates logarithmic dissipation

This is a read-only forensic evaluation of job9009718's saved rejected stage.
No evolution was restarted, no scientific kernel was changed, and the frozen
Aurora N24 run remains unchanged. The new inspect_failed_rhs.cpp diagnostic
loads the raw volume/surface capture, fills the original halos, and invokes the
native bulk kernel on one local CPU rank. The unmodified mode reproduces the
Aurora first-error cell3071/reason7 exactly. The capture SHA256 is unchanged.

## Where and how

The eight cells nearest the puncture, at (plus/minus.005625)^3 and
r=.0097427858, have chi exactly0 in the rejected stage. All stored volume
values are finite, and the minimum conformal lapse remains.0209212. The final
accepted state saved for failure analysis has minimum chi2.44488022e-7 at
9.1575M. That checkpoint is explicitly unadmitted, and was used only to inspect
stored arrays; it is not an allowed restart.

The first reported bad cell3071 is at (-.005625,-.005625,-.050625), r=.05124619.
Its physical/conformal lapse variable is positive,.02714535; field18 itself
stores a perturbation and is negative. The native KO stencil explicitly
connects cell3071 to zero-chi donor4095 along z, four core spacings away
(dx=.01125). Both are inside the last candidate horizon radius.1066377,
and far inside the spherical-shell inner boundary.16.

The exact source path is:
1. coupled_mpi.cpp uses log-RK rates RHS_chi/chi and exponentiates intermediate
   candidates. This GW1 run has no chi floor.
2. hyperbolic_composite_exchange.hpp:135 calls PositiveLog(chi,reference).
   positive_variables.hpp:14 computes log(chi/reference) without a lower bound.
3. Zero chi produces negative infinity in that dissipation chart.
4. hyperbolic_composite_device.hpp:166–183 applies the weighted KO operator to
   the log chart, then multiplies by the local chi. Nonzero neighboring chi
   cannot remove the infinity/NaN from the stencil.
5. The final finite-RHS check reports reason7 at the first affected cell.

## Controlled snapshot evaluations

| Read-only mode | Nonfinite RHS cells | Remaining invalid cells | First reason |
| --- | ---: | ---: | --- |
| Original log-variable KO |96 (chi RHS only)|104 total|3071: nonfinite RHS |
| KO disabled |0|8 zero-chi inputs|4095: invalid input |
| KO on physical variables |0|8 zero-chi inputs|4095: invalid input |
| Original log KO, replacing only the8 zero-chi snapshot values by1e-8 |0|0|success |

At cell3071 the non-dissipative chi RHS is finite (.00304570723); with physical
variable KO it is .00227020457. The zero-chi snapshot repair gives a finite
bulk RHS everywhere, but this is NOT a time-evolution test or proof that a
floor fixes long-time stability. Changing snapshot values is confined to
an in-memory diagnostic copy and is never saved as an evolution checkpoint.

The immediate failure is thus established as zero chi plus logarithmic KO,
not a nonfinite metric/lapse at the first reported cell, a scri null-closure
failure, or a demonstrated loss of hyperbolicity. The observed zero is produced
during the log-RK update. The saved capture does not include the intermediate
logarithm or every RK rate, so it does not distinguish a finite exponent
underflow from an earlier nonfinite log-rate, nor explain fully why the
puncture profile collapsed in that step. Further stage-level instrumentation
would be needed for that earlier cause.

## Why standard Z4c is not the same numerical scheme

The inspected standard reference is origin/main commit
c8d5c33e39089f005e416bf839652f6796fa6e82; file hashes are recorded in
standard-z4c-reference.json. Its z4c_update.cpp advances stored chi and alpha
directly with RK coefficients. Its z4c_calcrhs.cpp adds KO to u0 directly.
It does not form RHS_chi/chi for a log-RK update or log(chi/reference) for KO.
Consequently the zero-to-logarithm contamination mechanism above is absent,
even if the regular bulk equations otherwise agree in a suitable limit.

Standard Z4c also exposes chi_div_floor in RHS divisions and an optional
post-update floor_chi/chi_min_floor safeguard. Do not claim these are always
active: this reference defaults chi_div_floor=-1000 and floor_chi=false.
The GW1 run intentionally kept chi flooring disabled to match its3M donor.
The strong-field gauge is also not identical to default moving-puncture Z4c:
the hyperboloidal run retains the telegrapher vector, independent shift
auxiliary, reference/source terms and its chosen damping/driver coefficients.
At the failure radii both old and smooth w equal1, so the local coefficient
change in the outer transition is not itself active there (inward propagation
from the outer region is still possible).

A variable transformation can be equivalent for an exact positive solution
without producing equivalent finite-step/discrete dissipation schemes. Small
chi magnifies RHS_chi/chi, and floating-point exp does not guarantee a strictly
positive representable value. This is a concrete difference from the standard
implementation, not evidence that standard Z4c should fail similarly.

The earlier original-weight N16 success to20M switched at3M to direct chi and
lapse with a chi floor. The current smooth-weight continuation kept log variables
and no chi floor throughout. Those longer trajectories therefore do not isolate
the effect of the changed weight. A controlled next evolution should preserve
the smooth weight while testing the puncture variable/floor treatment separately,
starting from the admitted3M state. None was launched during this audit.

## Reproduction

    cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_inspect_failed_rhs -j8
    build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_inspect_failed_rhs 16 output/hyperbolic-core-shell/smooth-gauge-20m-v1/campaigns/n16-smooth-gauge-to20m-v1-segment01/output/puncture.failed-stage.bin

Execution was bounded to180 seconds. native-replay.json records all four
controls, individual nonfinite cell IDs and the exact zero donor connection;
execution.json records executable/capture hashes and runtime. snapshot-analysis.json
records zero-chi locations and the accepted-state comparison. These data refine
the earlier first-cell report, which could not identify the originating cells.
