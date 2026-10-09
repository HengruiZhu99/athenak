# Certified growth of the rounded finite matrices

The [negative finite-matrix control](hyperboloidal-limited-finite-growth-audit.md) now has an exact arithmetic certificate: each rounded J0 matrix at N8, N12 and N16 has two eigenvalues with positive real part. Floating-point eigenpair residuals alone did not establish this. The new certificate applies to the explicitly encoded rounded matrices, and does not establish a continuum eigenmode or an eigenvalue of the unrounded quadrature operator.

The exact target is the elementwise binary64 round-to-nearest/ties-even sum J=Jbulk+Jsat. Saved energy eigenvectors propose V=L^-T Q; an ordinary complex calculation proposes W. Neither proposal is trusted as an eigenbasis or inverse. Exact integer dyadic arithmetic forms N=WV and R=WJV−ND, where D contains the saved approximate eigenvalues as exact binary64 centers. Exact rational upper bounds give

```
nu >= ||I-N||_infinity < 1,
rho >= ||R||_infinity,
epsilon = rho/(1-nu).
V^-1 J V = D+N^-1 R,
||N^-1 R||_infinity <= epsilon.
```

The complex row bounds use |Re z|+|Im z|, which bounds the usual complex modulus without a square-root approximation. The Neumann bound proves N, V and W nonsingular. Gershgorin discs centered at D with common radius epsilon enclose every eigenvalue. Touching or overlapping closed discs are joined into components. The homotopy D+s N^-1 R, 0<=s<=1, stays inside their fixed union; each separated component contains its number of centers with algebraic multiplicity. Exact strict half-plane tests count positive and negative roots. No diagonalizability assumption or individual eigenpair forward-error theorem is used.

| N | Dimension | Outward upper nu | Outward upper epsilon | Positive / negative eigenvalues | Outward lower real part of each positive root |
|---|---:|---:|---:|---:|---:|
|8|64|9.329e-13|1.397e-10|2 / 62|2.0493330858359444|
|12|96|4.635e-11|1.818e-9|2 / 94|1.8456513955764864|
|16|128|1.184e-10|7.033e-10|2 / 126|1.9128322211424449|

The displayed bounds enclose exact rational values; all proof decisions use the rationals. Each positive root lies in an isolated singleton disc. Thirteen synthetic tests pass, including nonunitary complex similarity, exact Fraction product comparisons, repeated centers, overlapping/touching discs, wrong centers, defective Jordan proposals, a failed Neumann condition, extreme dyadic scales, normal/subnormal rounding and 256 fixed rational interval tests. Actual certificate runtimes are .303/.889/1.962 seconds, with empty stderr and unchanged inputs. An independent root readback changes product association from W(JV) to (WJ)V and exactly reproduces every nu, rho and epsilon, checks outward bounds and separated sign counts in 2.618 seconds.

The [compact evidence archive](validation/hyperboloidal-exact-finite-certificate-experiments-20261009/README.md) contains 50 cataloged blobs totaling 639,507 bytes, catalog SHA256 `baeadc6b87ed7d15e6741ee7c08d935a574d33960a97f6ca4730b1f2e1a6ee1f`. It preserves source, proof, tests, exact admissions, full rational certificates, runtime logs, prior-failure context and original frozen index `4087edf8fbe887fec4a64dddfe8e98158c2f2c376ca8d5452625888e9cd88208`. Complete exact hex-input JSON stays local with metadata, including the N8 file below 1MiB. Both catalog and compact capsule verification pass; the latter explicitly skips the three omitted-input proof replays.

No assembly, eigensolve, propagation, PDE query or production change occurs here. The general nongauge continuum comparator remains unresolved. The result does not identify the growth mechanism, establish constraint propagation, settle a boundary closure or validate the nonlinear pulse or later black-hole transition. Production remains at `27c19d20696ea6dd4704032c51dfd026218f64f2`.

When repeating the archive commands, use `python3 -B` in place of `python3` to prevent the capsule verifier's import from creating an uncataloged bytecode cache. The original collected README remains byte-preserved; this is an additive verification note.
