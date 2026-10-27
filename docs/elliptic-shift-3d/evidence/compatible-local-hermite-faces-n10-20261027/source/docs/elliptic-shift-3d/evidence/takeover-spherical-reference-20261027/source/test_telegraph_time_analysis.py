#!/usr/bin/env python3
"""Independent analytic error-scaling and rejection checks for time analysis."""
import copy
import unittest
import numpy as np
from analyze_telegraph_time import differences, summarize, controlled_inputs


def profile(n, amplitude):
    r=np.linspace(0,1,n);shape=np.sin(2*r)+.5
    physical=amplitude*shape[:,None]*np.arange(1,8)[None,:]
    q=np.zeros((n,8));q[:,6]=amplitude*shape
    diagnostic=np.zeros((n,12));diagnostic[:,:4]=amplitude*shape[:,None]*np.arange(1,5)[None,:]
    diagnostic[:,10]=1/3+amplitude*shape
    return dict(r=r,grid_jacobian=np.ones(n),q_physical=physical,q=q,beta=-r,gauge_speeds=np.c_[-r,r],
        diagnostic=diagnostic,B=amplitude*shape,
        rays=np.array([[.2+amplitude,amplitude],[.3+amplitude,2*amplitude]]))


class TimeAnalysis(unittest.TestCase):
    def test_fourth_order_error_and_spatial_fraction(self):
        # Exact errors e(dt)=dt^4*g(r); no PDE accuracy is inferred from this fixture.
        dt=.1;a,b,c=[profile(17,(dt/factor)**4) for factor in [1,2,4]]
        fine=profile(33,0.)
        row=dict(t=1,base_half=differences(a,b),half_quarter=differences(b,c),base_quarter=differences(a,c),spatial=differences(a,fine))
        report=summarize([row])
        self.assertAlmostEqual(report['fields_bulk']['observed_time_order'],4.,places=12)
        self.assertAlmostEqual(report['fields_bulk']['raw_time_to_space_fraction'],255/256,places=12)
        self.assertFalse(report['fields_bulk']['below_ten_percent'])
        for name in ['H_rms','M_max','ray_position_max','ray_logJ_max']:
            self.assertAlmostEqual(report[name]['observed_time_order'],4.,places=8)

    def test_identical_zero_and_missing_spatial_reference(self):
        a=profile(17,0.);delta=differences(a,a)
        report=summarize([dict(t=0,base_half=delta,half_quarter=delta,base_quarter=delta)])
        self.assertIsNone(report['fields_bulk']['below_ten_percent'])
        self.assertIsNone(report['fields_bulk']['observed_time_order'])
        report=summarize([dict(t=0,base_half=delta,half_quarter=delta,base_quarter=delta,spatial=delta)])
        self.assertEqual(report['fields_bulk']['raw_time_to_space_fraction'],0.)

    def test_reject_changed_nodes_or_mass_support(self):
        a=profile(17,0.);b=profile(33,0.);b['r'][4]+=.001
        with self.assertRaises(ValueError):differences(a,b)
        b=profile(17,0.);b['diagnostic'][2,10]=np.nan
        with self.assertRaises(ValueError):differences(a,b)

    def test_decimal_mass_polynomial_derivative(self):
        from audit_telegraph_mass import decimal_mass
        from decimal import Decimal
        values=[]
        # The chart logarithm is an exactly representable cubic. Its derivative
        # is exact on either seven-node rational stencil, so refinement must
        # preserve the independently checked geometric mass at r=1/2.
        for n in [17,33]:
            r=np.linspace(0,1,n);q=np.zeros((n,8));q[:,0]=r**3/4;q[:,3]=3
            values.append(decimal_mass(r,q,(n-1)//2))
        self.assertLess(abs(values[0]-values[1]),Decimal('1e-60'))

    def test_controlled_input_and_source_rejection(self):
        base=dict(manifest=dict(arguments=dict(cfl=.05,n=17,kappa=.005,output='a')),
                  binding={'numerical':'sha'},data={0:None,1:None})
        other=copy.deepcopy(base);other['manifest']['arguments'].update(cfl=.025,output='b')
        controlled_inputs(base,other,{'cfl'})
        other['manifest']['arguments']['kappa']=.006
        with self.assertRaises(ValueError):controlled_inputs(base,other,{'cfl'})
        other=copy.deepcopy(base);other['binding']['numerical']='changed'
        with self.assertRaises(ValueError):controlled_inputs(base,other,{'cfl'})
        other=copy.deepcopy(base);del other['data'][1]
        with self.assertRaises(ValueError):controlled_inputs(base,other,{'cfl'})

if __name__=='__main__':unittest.main()
