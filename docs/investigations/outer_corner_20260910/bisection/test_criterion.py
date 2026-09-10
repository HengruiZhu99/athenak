import unittest
from criterion import classify,relative_width
class Criterion(unittest.TestCase):
    def test_final_value_not_earlier_minimum(self):
        self.assertEqual(classify([{'time':50,'minLapse':.001},{'time':200,'minLapse':.2}],200),'disperse')
    def test_strict_boundary(self):
        self.assertEqual(classify([{'time':200,'minLapse':.01}],200),'disperse')
        self.assertEqual(classify([{'time':200,'minLapse':.00999}],200),'collapse')
    def test_incomplete_is_not_dispersal(self):
        with self.assertRaises(ValueError):classify([{'time':50,'minLapse':.5}],200)
    def test_invalid_lapse_is_not_collapse(self):
        for value in [float('nan'),-.01,0]:
            with self.assertRaises(ValueError):classify([{'time':200,'minLapse':value}],200)
    def test_relative_precision(self):
        self.assertGreater(relative_width('-.047','-.05'),.00001)
        self.assertLess(relative_width('-.0499998','-.05'),.00001)
    def test_early_crossing_is_collapse_without_final_time(self):
        self.assertEqual(classify([{'time':79,'minLapse':9e-6}],200),'collapse')
    def test_early_dip_remains_collapse_after_rebound(self):
        self.assertEqual(classify([{'time':79,'minLapse':9e-6},{'time':200,'minLapse':.9}],200),'collapse')
    def test_equality_is_not_an_early_crossing(self):
        with self.assertRaises(ValueError):classify([{'time':79,'minLapse':1e-5}],200)
    def test_nonfinite_after_crossing_is_not_accepted(self):
        with self.assertRaises(ValueError):classify([{'time':79,'minLapse':9e-6},{'time':80,'minLapse':float('nan')}],200)
if __name__=='__main__':unittest.main()
