#!/usr/bin/env python3
"""HELD exact-certificate synthetic/rounding tests; never reads actual matrices."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,io,json,math,random,sys,time,unittest
import dyadic_certificate as dc
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def I(n):return [[float(i==j) for j in range(n)] for i in range(n)]
def diag(entries):return [[entries[i] if i==j else 0. for j in range(len(entries))] for i in range(len(entries))]

class ExactCertificateTests(unittest.TestCase):
    def test_positive_and_negative_diagonal(self):
        a=dc.certify(diag([-1.,2.]),I(2),I(2),[-1.,2.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],1)
        self.assertEqual(a['certified_negative_eigenvalues_at_least'],1)
        a=dc.certify(diag([-1.,-2.]),I(2),I(2),[-1.,-2.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],0)
        self.assertEqual(a['certified_negative_eigenvalues_at_least'],2)

    def test_repeated_positive_centers_count_multiplicity(self):
        a=dc.certify(diag([2.,2.,-2.]),I(3),I(3),[2.,2.,-2.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],2)
        self.assertIn([0,1],[g['center_indices'] for g in a['clusters']])

    def test_complex_conjugate_pair(self):
        V=[[1.,1.],[-1j,1j]];W=[[.5,.5j],[.5,-.5j]]
        a=dc.certify([[1.,-2.],[2.,1.]],V,W,[1+2j,1-2j])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],2)
        self.assertEqual(a['common_radius_upper']['numerator'],'0')

    def test_dense_nonunitary_complex_diagonal_similarity(self):
        # V*W=I, J=V*diag(1.5,-.25)*W; all entries exact dyadic.
        V=[[1+1j,1.],[1.,1-1j]];W=[[1-1j,-1.],[-1.,1+1j]]
        J=[[3.25,-1.75-1.75j],[1.75-1.75j,-2.]]
        a=dc.certify(J,V,W,[1.5,-.25])
        self.assertTrue(a['invertibility_proved'])
        self.assertEqual(a['nu_upper']['numerator'],'0')
        self.assertEqual(a['residual_upper']['numerator'],'0')
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],1)
        self.assertEqual(a['certified_negative_eigenvalues_at_least'],1)

    def test_overlapping_and_wrong_centers_do_not_false_pass(self):
        a=dc.certify(diag([1.,-1.]),I(2),I(2),[0.,0.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],0)
        self.assertEqual(len(a['clusters']),1)

    def test_touching_discs_are_merged(self):
        a=dc.certify(diag([-1.,3.]),I(2),I(2),[0.,2.])
        self.assertEqual(a['common_radius_upper']['numerator'],'1')
        self.assertEqual(a['common_radius_upper']['denominator'],'1')
        self.assertEqual(len(a['clusters']),1)
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],0)
        separated=dc.disc_components([(F(0),F(0)),(F(2)+F(1,2**100),F(0))],F(1))
        self.assertEqual(separated,[[0],[1]])

    def test_nearly_defective_jordan_no_diagonalizability_assumption(self):
        tiny=2.**-40;large=2.**40;V=[[1.,1.],[0.,tiny]];W=[[1.,-large],[0.,large]]
        a=dc.certify([[1.,1.],[0.,1.]],V,W,[1.,1.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],2)
        a=dc.certify([[1.,1.],[0.,1.]],V,W,[-1.,-1.])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],0)

    def test_neumann_boundary_and_singular_basis_reject(self):
        for W in ([[0.,0.],[0.,0.]],[[2.,0.],[0.,2.]]):
            a=dc.certify(diag([1.,2.]),I(2),W,[1.,2.])
            self.assertFalse(a['invertibility_proved'])
            self.assertEqual(a['certified_positive_eigenvalues_at_least'],0)

    def test_extreme_binary64_scale(self):
        small=2.**-1000;large=2.**1000
        a=dc.certify(diag([-small,large]),I(2),I(2),[-small,large])
        self.assertEqual(a['certified_positive_eigenvalues_at_least'],1)
        self.assertEqual(a['certified_negative_eigenvalues_at_least'],1)

    def test_integer_complex_products_against_independent_fractions(self):
        A=[[.1+1j,2.**-1074],[2.**500,-.3j]];B=[[.2j,3.],[2.**-500,1.+.2j]]
        a=dc.DyadicMatrix.from_binary64(A);b=dc.DyadicMatrix.from_binary64(B);c=a.multiply(b)
        for i in range(2):
            for j in range(2):
                real=F(0);imag=F(0)
                for k in range(2):
                    x,y=complex(A[i][k]),complex(B[k][j]);xr,xi=F.from_float(x.real),F.from_float(x.imag);yr,yi=F.from_float(y.real),F.from_float(y.imag)
                    real+=xr*yr-xi*yi;imag+=xr*yi+xi*yr
                self.assertEqual(c.entry(i,j),(real,imag))
        self.assertEqual(dc.DyadicMatrix.from_binary64([[1+1j]]).upper_infinity_norm(),F(2))

    def assert_encloses(self,value):
        lo,hi=dc.outward_binary64(value)
        if math.isfinite(lo):self.assertLessEqual(F.from_float(lo),value)
        else:self.assertEqual(lo,-math.inf)
        if math.isfinite(hi):self.assertGreaterEqual(F.from_float(hi),value)
        else:self.assertEqual(hi,math.inf)
        if math.isfinite(lo) and math.isfinite(hi) and lo!=hi:
            # Adjacent magnitude patterns; signed zero is harmless.
            self.assertEqual(abs(dc.bits(lo)-dc.bits(hi)),1)

    def test_outward_binary64_exact_and_halfway(self):
        for value in [F(0),F(1),F(-1),F(1)+F(1,2**53),F(1,2**1075),-F(1,2**1075),F(1,2**2000),F(2)**1100,-F(2)**1100,F.from_float(dc.MAX_FINITE)]:self.assert_encloses(value)
        self.assertEqual(dc.nearest_binary64(F(1)+F(1,2**53)),1.)
        self.assertEqual(dc.nearest_binary64(F(1,2**1075)),0.)
        self.assertEqual(dc.nearest_binary64(F(1)+3*F(1,2**53)),dc.from_bits(dc.bits(1.)+2))

    def test_random_outward_fraction_bounds(self):
        generator=random.Random(61009)
        for _ in range(256):
            v=F(generator.randrange(-2**80,2**80),generator.randrange(1,2**80))
            e=generator.randrange(-1200,1100);v=v*2**e if e>=0 else v/F(2**(-e));self.assert_encloses(v)

    def test_invalid_inputs_reject(self):
        with self.assertRaises(ValueError):dc.certify([[math.nan]],[[1.]],[[1.]],[1.])
        with self.assertRaises(ValueError):dc.certify([[1.]],[[1.,0.]],[[1.]],[1.])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if not a.execute:print('HELD source-only synthetic tests; no actual matrix access.');return 0
    if a.output is None:p.error('--output required')
    a.output.mkdir(parents=True,exist_ok=False);paths=[Path(__file__),HERE/'dyadic_certificate.py',HERE/'PLAN.md']
    before={str(p.resolve()):sha(p) for p in paths};begin=time.monotonic();stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExactCertificateTests);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (a.output/'tests.log').write_text(stream.getvalue());after={str(p.resolve()):sha(p) for p in paths}
    receipt={'command':[sys.executable,*sys.argv],'inputs_before':before,'inputs_after':after,'sources_unchanged':before==after,
      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-begin,
      'no_actual_matrix_access':True,'passed_synthetic_exact_certificate_tests':result.wasSuccessful() and before==after}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2));return 0 if receipt['passed_synthetic_exact_certificate_tests'] else 1
if __name__=='__main__':sys.exit(main())
