"""Scientific qualification must survive bounded, single-order diagnoses."""
import unittest
from check_hispid_controls import boost_orders,refinement_qualified


def successful_import_row(rms,passed=True):
    return dict(returncode=0,bound_inputs_unchanged=True,
                zero_evolution_verified=True,import_={'passed':True},
                expansion_rms=rms,passed=passed)


class ControlPolicy(unittest.TestCase):
    def row(self,rms,passed=True):
        row=successful_import_row(rms,passed)
        row['import']=row.pop('import_')
        return row

    def test_one_order_requires_explicit_gamma_diagnosis(self):
        self.assertEqual(boost_orders('96',['gamma10'],True),[96])
        for spec,cases,diagnostic in [('96',['gamma10'],False),
            ('96',['boost885'],True),('96',['gamma10','kerr99'],True),
            ('64,96,160',['gamma10'],True),('96,64,160',['gamma10'],False)]:
            with self.assertRaises(ValueError):boost_orders(spec,cases,diagnostic)

    def test_successful_single_order_cannot_qualify_refinement(self):
        self.assertFalse(refinement_qualified([self.row(1e-9)],True))

    def test_three_order_qualification_keeps_every_import_gate(self):
        rows=[self.row(1e-4,False),self.row(1e-5,False),self.row(1e-9)]
        self.assertTrue(refinement_qualified(rows,True))
        rows[0]['import']['passed']=False
        self.assertFalse(refinement_qualified(rows,True))

    def test_timeout_or_iteration_limit_is_not_convergence(self):
        for code in ('timeout',1):
            rows=[self.row(1e-4,False),self.row(1e-5,False),self.row(1e-9)]
            rows[-1]['returncode']=code
            self.assertFalse(refinement_qualified(rows,True))


if __name__=='__main__':unittest.main()
