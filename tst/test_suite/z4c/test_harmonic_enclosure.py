"""Synthetic ideal-harmonic range controls; no native sampler or horizon solve."""
import math
import unittest
from decimal import Decimal
import numpy as np
from harmonic_enclosure import axial_range,axis_weights,add,square,ZERO,pi_interval,cosine_grid


class HarmonicEnclosureTests(unittest.TestCase):
    def check_range(self,coefficients,minimum,maximum):
        for axis in ('x','y','z'):
            with self.subTest(axis=axis):
                result=axial_range(coefficients,axis,32)
                self.assertLessEqual(result['radius_lower_bound'],minimum)
                self.assertGreaterEqual(result['radius_upper_bound'],maximum)
                self.assertEqual(result['decimal_precision'],100)

    def test_constant_and_linear_known_global_extrema(self):
        self.check_range([4.],4/math.sqrt(4*math.pi),4/math.sqrt(4*math.pi))
        for linear in ((.2,-.3,.1),(-.4,.25,-.1),(1e10,-1e10,1e10)):
            c=np.r_[4.,linear];mean=c[0]/math.sqrt(4*math.pi)
            amplitude=math.sqrt(3/(4*math.pi))*np.linalg.norm(linear)
            self.check_range(c,mean-amplitude,mean+amplitude)

    def test_quadratic_known_extrema_and_tiny_nonaxis_remainder(self):
        for amplitude in (.3,-.3):
            c=np.zeros(9);c[0]=4;c[4]=amplitude
            mean=4/math.sqrt(4*math.pi);a=amplitude*math.sqrt(5/(4*math.pi))
            self.check_range(c,mean+min(a,-a/2),mean+max(a,-a/2))
        # Pure real Y22c is proportional to x^2-y^2; its exact extrema are ±1.
        c=np.zeros(9);c[0]=4;c[7]=.3
        mean=4/math.sqrt(4*math.pi);a=.3*math.sqrt(15/(16*math.pi))
        self.check_range(c,mean-a,mean+a)
        # Add a tiny nonzonal mode to a positive zonal surface. Its uniform
        # contribution is no greater than the addition-theorem bound.
        c=np.zeros(16);c[0]=4;c[4]=.3;c[14]=1e-9
        result=axial_range(c,'z',32)
        self.assertLessEqual(result['radius_lower_bound'],mean-.15*math.sqrt(5/(4*math.pi)))
        self.assertGreater(float(result['orthogonal_remainder_upper']),0)

    def test_all_parities_weight_norm_and_low_degree_phase(self):
        for axis in ('x','y','z'):
            for l in (*range(9),32,160):
                norm=ZERO
                for w in axis_weights(l,axis):norm=add(norm,square(w))
                with self.subTest(axis=axis,l=l):self.assertLessEqual(norm[0],1);self.assertGreaterEqual(norm[1],1)
        for axis,index in (('x',1),('y',2)):
            weights=axis_weights(1,axis)
            self.assertLessEqual(weights[index][0],-Decimal(1))
            self.assertGreaterEqual(weights[index][1],-Decimal(1))
            self.assertTrue(all(weight==ZERO for i,weight in enumerate(weights) if i!=index))
        self.assertEqual(axis_weights(2,'x')[0],(-Decimal('.5'),-Decimal('.5')))

    def test_machin_pi_and_exact_cosine_quadrants(self):
        lo,hi=pi_interval()
        self.assertGreater(lo,Decimal('3.14159265358979323846264338327950288419716939937510'))
        self.assertLess(hi,Decimal('3.14159265358979323846264338327950288419716939937511'))
        values=cosine_grid(32)
        self.assertEqual(values[0],(Decimal(1),Decimal(1)))
        self.assertEqual(values[16],ZERO);self.assertEqual(values[32],(-Decimal(1),-Decimal(1)))
        self.assertLess(abs(float(values[8][0])-math.sqrt(.5)),1e-15)

    def test_malformed_inputs_and_unbounded_requests_reject(self):
        for values in ([],[1.,2.],[float('nan')],[float('inf')],[[1.]]):
            with self.subTest(values=values),self.assertRaises(ValueError):axial_range(values)
        for count in (0,15,17,16386,32.):
            with self.subTest(count=count),self.assertRaises(ValueError):axial_range([1.],intervals=count)
        with self.assertRaises(ValueError):axial_range([1.],axis='diagonal')


if __name__=='__main__':unittest.main()
