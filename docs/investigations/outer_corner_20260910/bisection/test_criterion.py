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
if __name__=='__main__':unittest.main()
