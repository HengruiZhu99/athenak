import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('athena_read', ROOT/'vis/python/athena_read.py')
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

class HistoryTests(unittest.TestCase):
    def test_segments_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'h.hst'
            f.write_text('# Athena++ history data\n# [1]=time [2]=dt [3]=mdot_H1\n0 1 2\n1 1 3\n'
                         '# Athena++ history data\n# [1]=time [2]=dt [3]=mdot_H1 [4]=M0_net_in_moving_H1\n'
                         '# cbd_schema=2 moving_channels=already_Reynolds_corrected\n1 1 4 5\n2 1 6 nan\n')
            data=ar.hst(f)
            np.testing.assert_equal(data['time'], [0,1,2])
            np.testing.assert_equal(data['mdot_H1'], [2,4,6])
            np.testing.assert_equal(data['M0_net_in_moving_H1'], [np.nan,5,np.nan])
            self.assertEqual(len(ar.hst_segments(f)), 2)
            self.assertEqual(len(ar.hst(f,raw=True)['time']),4)
    def test_width_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'h';f.write_text('# Athena++ history data\n# [1]=time [2]=dt\n0 1 3\n')
            with self.assertRaises(RuntimeError): ar.hst(f)
    def test_old_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'h';f.write_text('# Athena++ history data\n# [1]=time [2]=ldot_em_H [3]=ldot_em_H\n0 2 3\n')
            data=ar.hst(f)
            self.assertEqual(data['ldot_em_H__col2'][0],2)
            self.assertEqual(data['ldot_em_H__col3'][0],3)

if __name__=='__main__':unittest.main()
