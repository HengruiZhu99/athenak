from pathlib import Path
import sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from cbd_history import inward_currents,magnetic_normalization,frame_from_metadata
from disk_diagnostics import geometry_identity,geometry_cache_matches,parse_athena_parfile,find_traj_t,superposed_bbh_metric,adm_from_gcov

class Conventions(unittest.TestCase):
    def test_frame_is_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'par';p.write_text('<problem>\nsep=25\nq=1\n')
            with self.assertRaises(ValueError):parse_athena_parfile(p)
            params=parse_athena_parfile(p,historical_frame='inertial')
            a=find_traj_t(10,params)
            params['corotating']=True;params['frame_omega']=params['om']
            b=find_traj_t(10,params);c=find_traj_t(100,params)
            self.assertEqual(b,c);self.assertNotEqual(a['xi1y'],b['xi1y'])
            self.assertNotEqual(b['v1y'],0) # physical boost survives a stationary coordinate center
    def test_metric_rotating_identity(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'par';p.write_text('<problem>\nsep=25\nq=1\ncorotating=false\n')
            par=parse_athena_parfile(p);x,y,z=17.,3.,2.
            gi=superposed_bbh_metric(x,y,z,0,par,None)
            par['corotating']=True;par['frame_omega']=par['om']
            gr=superposed_bbh_metric(x,y,z,100,par,None)
            _,_,ai,bi=adm_from_gcov(gi);_,_,ar,br=adm_from_gcov(gr)
            np.testing.assert_allclose(ar,ai,rtol=1e-14)
            np.testing.assert_allclose(br,bi+np.array([-par['om']*y,par['om']*x,0]),rtol=1e-13)
    def test_no_double_correction(self):
        c={'M0_net_in_moving_H1':np.array([1]),'valid_H1':np.array([1])}
        with self.assertRaises(ValueError):
            inward_currents(c,'H1',mode='moving',metadata='frame=inertial',reynolds={})
        with self.assertRaises(ValueError):frame_from_metadata('')
        with self.assertRaises(ValueError):
            magnetic_normalization([1],1,.5,convention='fixed_total',rate_provenance='BH1_Kerr')
    def test_legacy_signs(self):
        names=['mdot','edot_f','edot_em','phiB','area']
        names += [f'{p}{a}dot_{s}' for p in 'pl' for a in 'xyz' for s in ('f','em')]
        c={f'{k}_H1':np.array([1.]) for k in names}
        h=inward_currents(c,'H1',mode='legacy_fixed',historical_frame='inertial')
        np.testing.assert_equal(h['pin'],[[-2,-2,-2]])
        np.testing.assert_equal(h['jin'],[[-2,-2,-2]])
        np.testing.assert_equal(h['mdot'],[1])

    def test_cache_geometry_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cache.npz';params={'corotating':False,'spin':np.array([0.,0.,0.7])}
            np.savez(p,geometry_identity=np.array(geometry_identity(params)))
            self.assertTrue(geometry_cache_matches(p,params))
            params['corotating']=True
            self.assertFalse(geometry_cache_matches(p,params))
            np.savez(p,old_value=np.array(1))
            self.assertFalse(geometry_cache_matches(p,params))

if __name__=='__main__':unittest.main()
