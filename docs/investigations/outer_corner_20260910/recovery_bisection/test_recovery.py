import unittest,json,tempfile
from pathlib import Path
from criterion import classify,first_early_event
from controller import validate_termination
class Recovery(unittest.TestCase):
    def rows(self,values):return [dict(time=float(i),minLapse=x,cycle=i) for i,x in enumerate(values)]
    def test_recovery_after_dip(self):self.assertEqual(classify(self.rows([1,.09,.81]),200),'disperse')
    def test_strict_thresholds_and_order(self):
        for values in [[1,.1,.81],[1,.09,.8],[1,.81,.09]]:
            with self.subTest(values=values):
                self.assertIsNone(first_early_event(self.rows(values)))
                with self.assertRaises(ValueError):classify(self.rows(values),200)
    def test_first_event_wins(self):
        self.assertEqual(classify(self.rows([1,9e-6,.9]),200),'collapse')
        self.assertEqual(classify(self.rows([1,.09,.9,9e-6]),200),'disperse')
    def test_nonfinite_after_recovery_rejects(self):
        with self.assertRaises(ValueError):classify(self.rows([1,.09,.9,float('nan')]),200)
    def test_native_recovery_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);rows=self.rows([1,.09,.81]);marker=dict(schema_version=3,outcome='dispersal_lapse',minimum_lapse_seen=.089,dip_threshold=.1,recovery_threshold=.8,**rows[-1])
            (d/'stdout.log').write_text('Terminating on user stopping condition: global minimum lapse recovered after dip\n')
            (d/'x.termination.json').write_text(json.dumps(marker))
            self.assertEqual(validate_termination(d,rows),'early_lapse_recovery')
            marker['minimum_lapse_seen']=.1;(d/'x.termination.json').write_text(json.dumps(marker))
            with self.assertRaises(RuntimeError):validate_termination(d,rows)
if __name__=='__main__':unittest.main()
