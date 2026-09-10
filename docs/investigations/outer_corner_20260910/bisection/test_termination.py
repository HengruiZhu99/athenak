from pathlib import Path
import json,tempfile,unittest
from controller import validate_termination

class Termination(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.case=Path(self.tmp.name)
        self.rows=[dict(time=80.,cycle=123,minLapse=9e-6)]
        self.marker=dict(schema_version=3,outcome='collapse_lapse',threshold=1e-5,**self.rows[-1])
        (self.case/'stdout.log').write_text('Terminating on user stopping condition: global minimum lapse below collapse threshold\n')
    def write_marker(self):
        (self.case/'test.termination.json').write_text(json.dumps(self.marker))
    def test_valid_early_stop(self):
        self.write_marker();self.assertEqual(validate_termination(self.case,self.rows),'early_global_lapse')
    def test_crossing_without_marker_rejected(self):
        with self.assertRaises(RuntimeError):validate_termination(self.case,self.rows)
    def test_mismatched_marker_rejected(self):
        for key,value in [('time',81),('minLapse',8e-6),('cycle',124),('threshold',.01),('outcome','dispersion')]:
            with self.subTest(key=key):
                original=self.marker[key];self.marker[key]=value;self.write_marker()
                with self.assertRaises(RuntimeError):validate_termination(self.case,self.rows)
                self.marker[key]=original
    def test_no_crossing_requires_t200(self):
        (self.case/'stdout.log').write_text('Terminating on time limit\n')
        for lapse in [.005,.9]:
            self.assertEqual(validate_termination(self.case,[dict(time=200,minLapse=lapse)]),'final_time_lapse')
        with self.assertRaises(RuntimeError):validate_termination(self.case,[dict(time=199,minLapse=.005)])
    def test_wall_stop_not_classified(self):
        (self.case/'stdout.log').write_text('Terminating on wall clock limit\n')
        with self.assertRaises(RuntimeError):validate_termination(self.case,self.rows)
    def test_authenticated_historical_crossing(self):
        (self.case/'stdout.log').write_text('Terminating on wall clock limit\n')
        self.assertEqual(validate_termination(self.case,self.rows,historical=True),'historical_early_global_lapse')
        with self.assertRaises(RuntimeError):validate_termination(self.case,[dict(time=80,minLapse=.005)],historical=True)
if __name__=='__main__':unittest.main()
