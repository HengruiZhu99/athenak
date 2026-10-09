Primary algorithm attribution

- Nicholas J. Higham (2005), The Scaling and Squaring Method for the Matrix
  Exponential Revisited, SIAM Journal on Matrix Analysis and Applications
  26(4), 1179–1193, DOI https://doi.org/10.1137/04061101X.
  Author's primary metadata: https://eprints.maths.manchester.ac.uk/634/.
  The algorithm was checked through the following later primary author paper;
  the original 2005 author-PDF fetch timed out in this session.
- Awad H. Al-Mohy and Nicholas J. Higham (2009), A New Scaling and Squaring
  Algorithm for the Matrix Exponential, author manuscript dated 19 January
  2009: https://eprints.maths.manchester.ac.uk/1217/1/paper9.pdf.
  Section 3, equations (3.5)–(3.6), printed page 9, supplies the factored
  degree-13 rational evaluation and solve.  Table 3.1 and Algorithm 3.1,
  printed pages 9–10, supply the one-norm scaling threshold
  5.371920351148152.  Table 3.1 separately assigns 4.25 to Algorithm 5.1;
  that adaptive algorithm is not implemented here.  The paper also discusses
  overscaling and squaring roundoff, which this simple fixed-degree candidate
  does not remove.
- Official SciPy 1.13.1 source was used as a secondary implementation-level
  coefficient cross-check, not as the source of a new mathematical bound:
  https://raw.githubusercontent.com/scipy/scipy/v1.13.1/scipy/linalg/_matfuncs_expm.pyx.in.
  The 14 integer coefficients in its degree-13 routine agree with the
  independent exact factorial/Taylor checks in check_synthetic.py.  The
  candidate Python helper was written independently from the mathematical
  evaluation, rather than copying the SciPy implementation.  It does not
  implement SciPy's adaptive matrix-power selection.

These primary sources were accessed on 2026-10-09.  The web text readback
explicitly returned Table 3.1 rows for both thresholds and equations 3.5–3.6.
No general forward-error certificate for an arbitrary nonnormal matrix is
claimed from the source or from the synthetic gate.
