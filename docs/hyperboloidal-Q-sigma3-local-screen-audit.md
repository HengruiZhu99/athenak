# σ3 local Fourier comparison: no accepted stabilization or adoption

This screen does not establish an accepted improvement or a new gauge option. Late σ3 retains the same target maximum as σ5, Re λ=22.467615596732056 at a=.5,r=.85,k4,radial. The late feedback V=Smooth(.85,.95) is exactly zero at r=.85, so this onset-gap maximum is independent of σ. Earlier feedback V=Wgauge(.45,.85) reduces some sampled maxima, but positive primitive roots remain and its k0 maximum is slightly worse for σ3. There was no native/global σ3 run or source adoption.

The calculation uses the actual C0 geometric RHS with physical P storage/evolution unchanged, κ1=10/α, κ2=0, and the frozen robust Q/preferred-source helpers. Both new forms retain the physical-inner alpha blend and explicit ξ=1/a. The Q helper has outer η=1; the separately named physical-P/source-off spatial-norm comparator has ρ=1.5. The source-only σ3−σ5 difference changes beta value rows and adds no principal derivatives. This does not imply that all old/new gauge forms have identical principal matrices.

The predeclared grid consists of a=.5,.75,1,2; r=.45,.65,.8,.85,.9,.95,.98; k=0,4,16,64,256; radial and(.36,-.48,.8) directions; and seven forms. It contains 1960 parameter rows for each matrix convention. Forms 0–4 preserve global-Q σ0, global-Q σ5 late, physical-inner Q σ5 late, physical-P/source-off spatial-norm, and physical-inner Q σ5 early-W controls. Forms 5/6 are physical-inner Q σ3 late/early-W. All 5880 spectra/121,520 roots remain locally indexed.

At a=.5, maxima of Re λ over the sampled radii/directions for the intrinsic20 convention are:

| Coordinate k | P/source-off norm | Q σ5 late | Q σ5 early W | Q σ3 late | Q σ3 early W |
|---:|---:|---:|---:|---:|---:|
|0|1.51801637|20.3993442|1.51801637|20.3993442|1.74418968|
|4|1.53195234|22.4676156|3.34310561|22.4676156|2.53615219|
|16|1.26965195|21.2980143|8.74517489|21.2980143|7.68587053|
|64|−.259155126|20.1689408|20.1689408|15.8475458|13.9152978|
|256|−.253218065|15.2902896|15.1373065|15.2902896|14.7593793|

The exact extracted rows, four-a maxima and sampled-band selections are in [summary.json](validation/hyperboloidal-Q-sigma3-local-screen-experiments-20261009/summary.json). Across all seven forms / 1960 rows, 1570 raw22, 1369 intrinsic20 and 1364 values-only RJB20 matrices have at least one primitive root above1e−8. These counts include the preserved baselines and are not physical-constraint growth counts.

## What the three matrices mean

Raw22 independently seeds all primitive components, including determinant/trace algebraic normal directions. Intrinsic20 seeds the 20 free components and completes the determinant-one metric and tracefree A constraints in all consumed jets before dual differentiation. It therefore retains spatial derivatives of the coefficient lift B(x). The third comparator, R J22 B(x0), uses only the value lift and is not the intrinsic constrained jet restriction on a nonflat reference. Its omitted differentiated-lift normal residual reaches 331.681712823 under the recorded jet scaling; its matrix differs from intrinsic20 by up to 1109.86660394. The three root sets are reported separately rather than interchanged.

The phase is the unscaled Cartesian coordinate exp(+ik n·x), with consistent value/first/second perturbation jets and no Ω amplitude weighting. Coordinate wavelength is2π/k. Metadata also record k_bar=k sqrt(χgtilde^ij n_i n_j) and physical spatial k=Ωk_bar. The sampled bands below scalar coordinate π/h for N16/span2.2 and N24/36/48/span2.1 are only frequency selections. They are not FD accuracy claims; oblique componentwise Nyquist conditions differ. No native FD symbol, KO, upwind, boundary, global coefficient coupling or subsidiary generator appears in this calculation.

Consequently, positive frozen primitive roots do not identify constraint growth, prove continuum instability, or predict a native growing mode. Conversely, the preceding conditional cubic null/curvature tangency result does not establish lower-order finite-k stability. The finite-Q and temporal-corner counterexamples and the absence of a closed nonlinear/radiative hierarchy remain.

## Actual source/control gates and provenance

All 1120 old intrinsic20 matrices and 280 prior early-W σ5 matrices reproduce bitwise. The independent reference-coefficient σ3−σ5 rank-one beta-source reconstruction agrees to 1.1368683772161603e−13 in all three conventions. Geometric rows are identical across forms; k0 real/direction identities, outer equal-weight identities and W_gauge=0 inner-gauge branch identities at r=.45 are exact. This last point lies inside the geometric transition; the exactly Cauchy core ends at r=.05. Beta pole is assembled once, with error 1.1102230246251565e−16.

The 2352 actual double/dual phase finite-difference cases converge in worst relative error from 2.805032499383881e−6 to 2.8050655674741615e−8 to 2.8644111428413417e−10 at ε1e−3,1e−4,1e−5. The consumed intrinsic determinant/trace input normals are≤8.864254524596229e−14 after 1+k² scaling, and stationary-reference output normals≤4.549219351980979e−15 after RHS scaling. Release and Debug ASan/UBSan actual matrix bytes and metadata are identical.

The original C++ batch has 381 pinned inputs and five commands, 118.147606209seconds. It uses C++17 Release -O3 -DNDEBUG and Debug -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer, with all exact Kokkos include flags in its receipt and the extracted summary. The actual run commands were:

```
/Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/fourier22-release /Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/matrices-release.bin
/Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/fourier22-debug /Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/matrices-debug.bin
```

C++ source SHA is 4a59aefa71044ba47197f0a1be31863d8efae62b586c1d676879b771a237eee6. Release executable SHA is aa7802b394176cb49f8bbdf2b606955a89399ad9fd611eaffe5cda8056f47db9; Debug executable SHA is c868dfc500482c8d34e00ba24d844b6992e9e29446418a028a59ceb656e5b792. The helpers remain byte-identical to frozen Q f3f3acfe…, factored-base 216bb80c…, early 562d01b4…. No compiled scientific source was changed in the correction below.

Exploration/pilot launch HEAD was a7260897b92d12d5fc1ceb2f3dc6a886676a663f. The final C++ runner and additive reanalysis launched at 85419e2fa74aa5cc0c357584bdb24e2604897739. Compiled production remains 27c19d20696ea6dd4704032c51dfd026218f64f2. These identities are distinct in the receipts; no production or prior frozen source bytes changed.

## Preserved v1 warning and additive correction

The original 46-file freeze97eca136515520bb7a551b32013ce4faa7bafde0d3a079cee2620f8367e95aa3 is preserved unchanged, including its incorrect REPORT.md claim that all stderr files were empty. Its four C++ commands have empty stderr; the fifth NumPy checker emitted 2433 bytes of RuntimeWarnings during complex-by-real matrix products. The exact warning command was:

```
/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python /Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/check_fourier22.py
```

The accepted additive 13-file index943db2465bb361802fdd237f7658f11f66f142ecff6d147920d0c8e9393a4077 uses unoptimized einsum, np.seterr(all='raise') and warnings.simplefilter('error'). Its one command exits 0 with empty stderr in .9631315seconds, reusing the unchanged C++ matrices without a rerun or tolerance change:

```
/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python /Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier-deterministic/check_fourier22.py
```

All 46 v1 files and 381 original inputs verify unchanged; the additive receipt has 385 inputs. Raw22/intrinsic20 root arrays are identical. Values-only RJB20 root ordering changes with arithmetic reduction order; nearest-root-set distance is≤7.162510439644634e−12 and max-Re change≤1.5918377727075494e−12. Positive-root counts are identical at every parameter row. The comparison is numerical, not an exact multiplicity or conditioning theorem. The additive ERRATUM records this correction without rewriting the original source, receipt, warnings or prose.

This is a local negative comparison, with no new adoption. It provides no uniform energy estimate, full Einstein-germ/radiative-Weyl admission, exact-scri closure, or finite-Q amplitude blowup theorem. A later single BH must still survive the inner wormhole-to-trumpet transition while retaining the Minkowski hyperboloidal reference throughout; no BH integration or compatibility conclusion is supplied here.

The original frozen DERIVATION.md incorrectly says that only beta value columns change. The change is in beta output rows, depending on alpha/chi/metric/beta input values through delta N; the source and formula already implement this correctly. The separate [notation addendum](validation/hyperboloidal-Q-sigma3-local-screen-experiments-20261009/ROOT-NOTATION-ADDENDUM.md) preserves that correction.

The [compact evidence archive](validation/hyperboloidal-Q-sigma3-local-screen-experiments-20261009/README.md) preserves both original frozen indexes, all command records and warnings, the accepted erratum, exact summary and independent reviews. Executables, objects, binary arrays and payloads larger than1MiB remain metadata only. Production source is unchanged.

Read-only catalog verification passes for63 preserved files,3,162,379 bytes
and14 finite JSON files. The catalog SHA256 is
`d0f017a8af652e3dc204a5c33b8ad331c1c41b3db8bac714432ef6952eeda47a`.
The reviewed draft remains byte-exact; the public copy incorporates the two
prose clarifications above.
