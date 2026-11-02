# Native initialization, numerical repairs and shared restoration: final assessment

The authorized A–E N16 control matrix is disposed. Native initialization and the separately tested consistency repairs are implemented and preserved, but **high-order convergence of the evolved puncture solution has not been achieved or demonstrated**. The shared-restoration runs both produce invalid shell geometry near9.39M, with or without the repaired numerical operators. No failed state was restarted. No new long N24/N32 runs were launched.

## Controlled evolution results

All controls start fresh from native C++ wormhole data, with direct chi/lapse, dt=.0005, fixed floors/damping/horizon settings and12GPU ranks. C/D/E use the separately repaired fifth-donor MPI exchange. SG1 is the preserved external-initializer baseline.

| Case | Change from native baseline | Disposition | Common r>.1099 H RMS at10M | M RMS at10M |
|---|---|---|---:|---:|
| SG1 | External initializer, original collar/KO8/scri | Completed10M | .0154149702 | .0290076410 |
| A | Native initializer, original collar/KO8/scri | Completed10M | .0154149684 | .0290076414 |
| B | Shared restoration, KO8/original scri | Failed9.3915M | unavailable | unavailable |
| C | Original collar, KO10/original scri | Completed10M | .0165178658 | .0299467550 |
| D | Original collar, KO10/first6 scri | Completed10M | .0166453350 | .0408930995 |
| E | Shared restoration, KO10/first6 scri | Failed9.3900M; concurrent storage guard | unavailable | unavailable |

A−SG1 isolates native initialization: evolved exterior norms are essentially unchanged. C−A isolates KO10 with correct distributed donors: H/M increase7.15%/3.24%, while absolute .775–.9 squared residuals barely change(+.019%/+.054%). The increase is concentrated partly in.16–.30, where squared H/M increase47.01%/39.38%. A smaller outer fraction is not an outer improvement.

D−C changes the scri first derivatives and the matching native diagnostic sampling. Each native operator gives H/M changes+.77%/+36.55% and .775–.9 squared H/M changes+3.58%/+114.60%. The additional common bulk diagnostic sampling control gives exterior M changes+33.22% (first4) or+32.13% (first6), and outer squared M changes+103.81%/+97.08%. The deterioration is therefore not solely a change in the bulk diagnostic stencil. Native boundary refresh and donor-seven last-three-row closures remain per case; this is a bounded common-bulk control, not a completely common boundary operator.

B−A and E−D show that the specified shared-W restoration is unstable in these matched N16 trials. B and E fail at the same eight symmetric shell points, r=.609375, with last accepted times only.0015M apart. This localizes the immediate invalid geometry and associates the failure with the shared-gauge experiments; it does not prove continuum ill-posedness or identify the preceding dynamical mechanism.

## Failure and diagnostic qualifications

E's native report and rejected capture demonstrate a finite but indefinite metric. At firstcell57352, xyz(.3748364114,−.3397319203,−.3397319203), eigenvalues are approximately(−.00281727,.787311,477.082); chi11.8570 and conformal lapse.00429280 remain positive. Surface metric minimum eigenvalue.28465 is positive. Null residuals remain≤1.56e-15 while pole/clock maxima rise to.74280/.10816. This is not a chi/lapse sign stop or failure of the algebraic null condition.

E also triggered the worker's storage-reserve guard while failure outputs were written. The guard sent SIGTERM; exact ordering relative to native failure handling is not recorded. Native return2, complete failed report/log, unadmitted checkpoint and indefinite captured geometry independently establish numerical failure, but the worker outcome must retain the infrastructure qualification. The worker aborted before hashing artifacts. A separate post-terminal inventory of19files was copied and SHA256/length verified; original worker metadata was not rewritten. The exact-case postmortem reader preserves this exception rather than relaxing the ordinary loader's infrastructure checks. Six already verified smoke checkpoint copies were rehashed remotely and retired to restore headroom. E failure evidence remains preserved and is never restartable.

The last B/E diagnostic rows are9.3M; the rejected RK stage abscissa was not recorded. E's last unaccepted candidate cutoff is.740995, outside the failed shell. Its apparently small candidate-exterior H/M(.01120/.01459) excludes that shell, while full H/M/Mr are15.97/486.49/191.18. Neither candidate-exterior improvement nor accurate horizon mass can be inferred from that row. No B/E10M endpoint or common-operator endpoint reconstruction exists.

## Horizon and floor evidence

| Case | Accepted horizon rows | Accepted mass error and measurement time | Negative chi stage/final | Chi floors stage/final |
|---|---:|---|---:|---:|
| SG1 |45/68|+.341326% at10M|56/8|56/8|
| A |45/68|+.341320% at10M|56/8|56/8|
| B |38/63|last−.009138% at8.4M; endpoint missing|104/32|112/32|
| C |41/68|last+.210556% at9.9M;10M missing|24/8|24/8|
| D |41/68|last+.210513% at9.9M;10M missing|24/8|24/8|
| E |39/63|last−.004763% at8.4M; endpoint missing|80/32|88/40|

Counts include rejected attempts and remain segment-local sums. All chi events occur at the nearest eight puncture cells, separate from the shell failure. Zero candidates, lapse/surface floor interventions and boundary-interpolation negative/zero counts are zero. Boundary counters remain separate in the machine-readable summary. Missing masses and finder flags are preserved without interpolation.

## Full specification assessment

- **Shared W:** implemented once, with the requested inner plateau, mu and auxiliary-D equation. The scri limit and dependent-D restoration terms are derived in STATUS.md; restoration was not silently deleted. Both prescribed shared-gauge trials failed, so this formulation is not validated for long evolution.
- **Native initializer:** production C++ direct quadrature/inversion, true grid coordinates, original slice/mass/conventions, and independent tolerance controls completed. A demonstrates that changing initialization alone does not fix the late outer error.
- **Initial-data convergence:** N16/N24/N32 and independent continuum/projection/exchange/cutoff/norm controls completed. Continuum H/M<2.6e-14. Coarse H orders1.50/1.98 reflect the slicing transition; bounded finer controls approach fourth order. The coarse grids are not an established asymptotic regime.
- **KO consistency:** the separately identifiable KO10 energy-form repair passes polynomial and full-patch consistency controls, with bounded outer order3.915 atN256. The actual12-rank smoke exposed a missing fifth MPI donor; repaired12-rank local states match one rank byte-for-byte and corrected GPU parity passes. KO10 alone does not remove evolved outer errors.
- **Scri consistency:** nontrivial manufactured surface/adjacent-row/normal-derivative tests demonstrate the original third-order defect. First6 substantially improves these tests (~48x RMS atN32;48→64 RMS orders4.722/4.730 and normal order5.671), with finite-grid seam-maximum order≈3.82 still qualified. These operator results do not establish evolved-solution accuracy; D worsens measured outer momentum residuals.
- **Controlled evolutions and preservation:** all five fresh N16 trials completed or failed numerically. Full/candidate histories, horizon gaps, sign/floor counters, null/regularity measures, admitted common-cutoff regional endpoints and the bounded common-operator control are preserved. Failed states were localized without replay or admission. No further resolution evolution is authorized by this matrix.

The remaining obstruction is evolved gauge/outer-layer error and shared-gauge shell instability, not an unimplemented initializer or untested repaired MPI path. A useful next study would isolate the growth leading to the shared-gauge shell metric failure and the original-collar outer residual, with fixed-coordinate diagnostics independent of an accepted horizon. That requires a new experimental plan; no timestep, floor, gauge coefficient or long-resolution campaign was changed automatically.

## Reproducible evidence

- [Matrix history plots and machine-readable summary](aurora/matrix-final/physics-summary.json), generated by `final_matrix.py` and `summarize_matrix.py` from preserved mirrors.
- [All histories](aurora/matrix-final/histories.png), [full constraints and boundary measures](aurora/matrix-final/full-and-boundary.png), [candidate-cutoff qualification](aurora/matrix-final/candidate-cutoff.png).
- [Admitted native endpoint regional reconstruction](aurora/abcd-versus-sg1/radial-comparison.json) and [radial plot](aurora/abcd-versus-sg1/radial.png).
- [Common bulk diagnostic control](aurora/common-operator/README.md), source/binary/checkpoint hashes and inspected figure.
- [E dual-failure disposition](aurora/e-failure/README.md), complete inventory/verification, failed metric locations, original scheduler/worker/native evidence.
- Initial-data/KO/scri controls and limitations: STATUS.md, EVOLUTION-MATRIX.md and checks/; frozen scientific build records under aurora/ko-dependency-build.

The monitor can be removed after this disposition. This closes the authorized control matrix and documents the measured obstruction; it does not mark the broader high-order evolution-convergence objective achieved.
