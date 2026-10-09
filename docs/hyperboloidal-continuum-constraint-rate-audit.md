# Coefficient-aware continuum constraint-rate validation

The independent finite-rb pointwise gates pass for the unchanged C0/spatial-norm
Minkowski reference control. They validate the selected stationary-reference
constraint rates before radial projection; they establish no discrete Bianchi
identity, constraint energy bound, CPBC or stability theorem.

The exact flat-core operator proof gives all 160 zero entries in
`C(d)L(d)T=B(d)C(d)T` on the complete algebraic tangent chart. In physical Cartesian
ordering H,Mxyz,Zxyz,Theta, the rates include
`M_t=-grad(H)/2+Delta(Z)-grad(div Z)+2κgrad(Theta)`,
`Z_t=M+grad(Theta)-κZ` and `Theta_t=H/2+div Z-2κTheta`.
The core gauge is alpha_t=-3P and beta_t=3Lambda/8; Kphysical=P+2Theta.
The raw22 extension is formal, not an assertion about off-normal constraints.
Literature independently reviewed these signs without a correction.

| Gate, rb=.98 | Cases | Largest final scaled L2 error |
| --- | ---: | ---: |
| Exact polynomial core actual source | 4,431 | 2.201e-16 |
| Exact polynomial core actual constraint rate | 4,431 | 9.443e-16 |
| Gauge-only actual zero rate | 714 | 1.136e-7 absolute L2 |
| Shell actual rate versus subsidiary | 2,751 | 1.979e-8 |

Parameters remain S=1, a=.5, geometric layer .05–.95, physical-P/spatial-norm
gauge xi=2, κinput=10 and κ2=0. All J0/1/2 nonnegative m and nonzero real/imaginary
phases use the frozen fields. The exact core comparison evaluates polynomial
jets directly at the origin/.025/.049. Transition/collar comparisons separately
differentiate complete actual source F(x) and physical constraint q(x) at the
prescribed points and five Cartesian spacings, without invented higher analytic
reference jets. No Theta/Z falloff is imposed.

The gauge constraints and subsidiary rates start exactly zero. The largest
extrapolated actual gauge rate is 1.190e-7. Of 714 actual gauge sequences, 512
show the required reduction and 202 pass the fixed absolute gate with order
unclassified. Shell extrapolated error is 2.107e-8; actual/subsidiary classified
sequence counts are 2,100/2,021, with 651/730 explicitly unclassified. Every h,
increment, observed order and extrapolation is retained. No tolerance changed.
Component RMS/peaks are sample statistics, not integrated physical energies.

All 7,896 cases pass in 546.61 s. All 23,688 API calls have empty stderr and all
14 recorded source/executable pins remain unchanged. The fresh additive driver
batches Cartesian samples and appends receipts; the original core driver is
preserved. Release/ASan outputs match bit-for-bit in all three API modes at three
representative points. The first Debug byte-copy lacked execute permission; its
failed attempt is preserved and the unchanged working binary succeeds.

The immutable local index has 117 files, 300,712,718 bytes and hash
`3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f`.
Saved-data readback verifies every hash and independently recomputes final
comparisons/increments without a kernel rerun. Launch HEAD is 2e0aa3b0;
freeze HEAD is 5ef1c97d; the public runtime implementation remains 27c19d20.
Core/Debug launch HEAD is reconstructed in additive provenance notes; both FD
stages record it directly. rb=.995 is untested here.

The next gate measures radial projection and SAT constraint production on the
same polynomial interpolant, separately from the continuum source tested here.
No eigenvalues, propagation or production changes are admitted by these results.
The later single-BH wormhole-to-trumpet goal still retains the Minkowski
hyperboloidal reference and has not been evolved in this stage.

The actual API builds compile in1.985850458s Release and1.109157334s Debug ASan/UBSan, with empty compile/dependency stderr. They pin1062/1064 compiler dependencies and four link archives each,1068 distinct inputs in their union. Root freshly rehashed those inputs, checked compiled production headers against27c19d20696ea6dd4704032c51dfd026218f64f2, and matched both retained binaries to the frozen copies. Release executable SHA256 is `2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15`; Debug is `75193a3b023eb0004f285fabb1a532efd24881db4cb660b3fa96e084a6e95507`. The API header SHA256 is `d60ea8266abebf5263eb78b81997b60af472dc7b1fe102bd6f2c21baca1a6017`. These are later API binaries; the earlier configuration-checkpoint binary preservation limitation is unchanged.

The [byte-preserved compact evidence archive](validation/hyperboloidal-continuum-constraint-rate-experiments-20261009/README.md)
contains181 cataloged blobs,1,769,788 bytes and42 finite JSON files. Its catalog
SHA256 is `aa60b346c7e8c25a0736f582643a2b68a82196588ab4107da11be5104c5b872f`.
The original117-file index remains verbatim. Its13 tagged large payloads
(full cases/call logs,tarballs and executable copies) remain locally retained
and are metadata only in Git; two duplicated build-attempt executable paths
are likewise metadata only. Catalog verification and the explicitly limited
compact-bundle verifier pass without scientific reruns. Full local saved-case
readback was independently repeated before collection; the compact archive
cannot recompute its absent large numerical case logs.
