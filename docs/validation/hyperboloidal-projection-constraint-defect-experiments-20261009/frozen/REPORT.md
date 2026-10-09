# J0 finite-rb projection constraint readback

The analytic projected-constraint point gate passes and measures substantial
constraint production on the sampled gauge seeds. It does not pass the complete
nongauge continuum-comparator gate. Both ordinary-FD attempts remain failed;
no generator spectrum, propagation, CPBC, integrated energy or production change
is admitted by this stage.

All calculations use the same J0/N8/rb=.98 radial operator, S=1, a=.5,
geometric layer .05–.95, physical-P/spatial-norm gauge xi=2, kappa_input=10,
kappa2=0 and Minkowski reference. Modal coefficients are channel-major degree 0–7;
the basis already includes r^L. The physical-eight ordering is H, Cartesian
Mx/My/Mz, Cartesian Zx/Zy/Zz, physical Theta. The table below uses sample Euclidean
norms of those eight coordinate components, with the fixed witness amplitudes;
these quantities are not physical energies or normalized growth rates.

| Gauge witness | Maximum sampled bulk constraint-rate norm | Maximum sampled SAT constraint-rate norm |
| --- | ---: | ---: |
| Constant lapse | 986.11286 | .0112036 |
| Polynomial beta, W=rho^3 | 2651.1084 | .214873 |
| Lapse, W=exp(-8rho) | 14.3114 | .000615920 |
| Beta, W=exp(-8rho) | 5.89942 | .00216420 |

The same polynomial interpolant X is used to form Jbulk X and Jsat X. The
analytically seeded actual constraint map evaluates C_ref[Phi(X)] and
C_ref[Phi(Y)] after the complete reference/physical lift, including its
coefficient and angular jets. All 84 gauge initial vectors are exactly zero
in these outputs. Their continuum source constraint-rate zero is a separately
derived positive-Omega Einstein-sector identity: the physical constraints do
not depend on lapse/shift; ADM lapse/shift variations at an Einstein reference
are tangent to H=M=0; C0 additions vanish at Theta=Z=0. This uses the previously
audited actual/ADM source equivalence and frozen gauge-rate controls. It is not
a numerical pass of either failed FD attempt. For general nongauge seeds,
C_ref[L_actual Phi(X)] remains unresolved here.

The original ordinary-FD attempt retains all five prescribed levels and stops
at case 16, constant lapse at r=.96 on the x axis. Its continuum last increment
is 1.8406042e-6, exceeding 2e-7, although observed orders 4.0143/4.0030/4.0072 and
both final/extrapolated gauge-zero tests pass. The extrapolated continuum norm
is 2.42650e-9; the bulk and SAT norms there are 81.8918 and .00386403. The source map
and direct-linearity controls pass before that stop. Its 16 cases / 32 API calls
take 9.11804 s with unchanged sources and empty stderr.

The separate authorized half-step attempt shifts all five h values by a factor
of 1/2 and changes no fields, stencil, operator or threshold. It stops at case 4,
constant lapse at r=.30 on the x axis: continuum increments 2.18e-8,2.67e-8,
1.04e-7,2.76e-7 show cancellation amplification; the final increment and
Richardson zero test fail (extrapolated norm 2.09000e-7). The projected bulk/SAT
checks still pass. This 4-case / 8-call attempt takes 2.92786 s. Its full records remain
failed. Ordinary binary64 FD has competing truncation and cancellation errors
in these two attempts; no tolerance is relaxed or result selected to erase that.

The distinct analytic point gate uses the unchanged manufactured-rate API.
At fixed point/channel its envelopes 1, rho, rho^2 span W,Wrho,Wrhorho, so a triangular
linear recovery supplies the actual RHS22 and constraint8 maps. Direct held
envelopes rho^3, exp(-8rho), the fixed shell, and the mixed cubic validate the map.
The constraints8 have their own error scale, independent of the RHS22 scale.
The recovered RHS22 map also matches the prior source-batch map. All 1,176 rows
and 294 analytic points pass in .109248 s; maximum held constraint-row scaled
error is 1.41843e-14, source RHS22 cross-binding 6.38994e-14. Sources are unchanged
and stderr is empty. This is an analytically seeded map evaluated in binary64,
not an exact-arithmetic claim about floating output.

Every available old projected/initial FD result is compared with this new point
map without reclassifying the old gate. Largest scaled Richardson differences
for bulk/total are 1.86106e-8 / 9.13199e-9 in the original attempt and
2.60697e-8 / 2.95738e-8 in the half-step attempt. Root independently reconstructs
all 294 analytic rows with separate scalar modal loops and math.fsum, obtaining
maximum scaled difference 1.30085e-14, total linearity 8.1755e-15 and 84 exact
gauge initial zeros. The additive root review is retained. Literature independently rehashes all 24
run source/data pins and call input/output, recovers the saved triangular maps
bit-for-bit, and confirms the 294-row / 84-gauge-zero scope without a correction
or new scientific query. Its independent receipt is retained as well.

The original shared source maps cover 4,809 Cartesian points,115,416 basis
queries and504 direct controls. Half-step refinement reuses 3,927 points and
adds 882. Saved-data postprocessing, with no kernel queries, preserves both the
original map NPZ and a 5,691-point union, explicit Cartesian ordering and schema.
The original map NPZ SHA256 is 8f419ece7c5e6b0b6318adefd4fca621d200bfb3019410d14ead849ce086231c;
the union is 63f46fc55243f8898d9cd0559f33447190fa9562a5c8c6834e2aacb7f2c3649f.
Their lineage, inputs, source outputs and readback receipts remain local;
large payloads may be metadata only in the published compact archive.

All scientific launches are at 9bc9fc71b057bc74d1ead0c3b34119390591c1bc, with
compiled public implementation 27c19d20 and unchanged Release executable
2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15.
The operator SHA256 is 2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742.
The first interpreter launch lacked SciPy and stopped before attempt creation
or any kernel query; its logs are preserved. Successful launches use the saved
CommandLineTools Python/PYTHONPATH environment, with NumPy 2.0.2 and SciPy 1.13.1,
and floating warnings are errors. All sources, root reviews, authorizations,
commands, point ordering, five-level FD records and failure receipts are frozen.

This finite matrix/point stage establishes no constraint closure for the radial
projection/SAT operator and no continuum instability theorem. It does not supply
a scri boundary prescription. The eventual single-BH wormhole-to-trumpet
transition must still retain the Minkowski hyperboloidal reference; no BH
integration is performed here.
