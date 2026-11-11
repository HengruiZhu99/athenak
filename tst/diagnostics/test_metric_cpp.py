"""Compare Python frame-aware reconstruction with actual prescribed-ADM output."""
from pathlib import Path
import os,sys,subprocess,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'vis/python')]
import athena_read
from disk_diagnostics import parse_athena_parfile,superposed_bbh_metric,adm_from_gcov
EXE=os.environ.get('ATHENA_DIAGNOSTICS_EXE')

@unittest.skipUnless(EXE,'set ATHENA_DIAGNOSTICS_EXE')
class MetricCpp(unittest.TestCase):
    def test_inertial_and_corotating(self):
        for rotating in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                d=Path(temp)
                # Binary Bondi remains the supported corotating initializer. Tilt is
                # exercised in the inertial case only, matching physical restrictions.
                text=(ROOT/'tst/diagnostics/smoke.athinput').read_text()
                text=text.replace('single_black_hole = true','single_black_hole = false')
                text=text.replace('sep = 32','sep = 4').replace('frame_omega = 0.005524271728019903',
                    'frame_omega = '+('0.125' if rotating else '0'))
                text=text.replace('corotating = true','corotating = '+str(rotating).lower())
                text=text.replace('dynamic = false','dynamic = true')
                text=text.replace('flux_modeled_horizons = true','flux_modeled_horizons = false')
                text=text.replace('a1 = 0','a1 = 0.4')
                text=text.replace('<problem>','<problem>\nth_a1 = '+('0' if rotating else '37')+'\nph_a1 = 23')
                text=text.replace('nlim = 3','nlim = 0')
                text+='\n<output3>\nfile_type=tab\nvariable=adm\nslice_x2=2.5\nslice_x3=1.5\ndata_format=%24.16e\ndt=1\n'
                inp=d/'input';inp.write_text(text)
                run=subprocess.run([EXE,'-i',str(inp),'-d',str(d)],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                data=athena_read.tab(d/'tab/matched_bondi.adm.00000.tab')
                par=parse_athena_parfile(inp)
                for index,x in enumerate(data['x1v']):
                    gcov=superposed_bbh_metric(x,2.5,1.5,0,par,None)
                    metric,_,alpha,beta=adm_from_gcov(gcov)
                    for key,value in zip(('adm_gxx','adm_gxy','adm_gxz','adm_gyy','adm_gyz','adm_gzz'),
                        (metric[0,0],metric[0,1],metric[0,2],metric[1,1],metric[1,2],metric[2,2])):
                        np.testing.assert_allclose(data[key][index],value,rtol=1e-12,atol=1e-13)
                    np.testing.assert_allclose(data['adm_alpha'][index],alpha,rtol=1e-12)
                    for key,value in zip(('adm_betax','adm_betay','adm_betaz'),beta):
                        np.testing.assert_allclose(data[key][index],value,rtol=1e-12,atol=1e-13)

    def test_actual_state_and_metric_clocks(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);text=(ROOT/'tst/diagnostics/smoke.athinput').read_text()
            text=text.replace('single_black_hole = true','single_black_hole = false')
            text=text.replace('corotating = true','corotating = false')
            text=text.replace('sep = 32','sep = 4')
            text=text.replace('frame_omega = 0.005524271728019903','frame_omega = 0')
            text=text.replace('dynamic = false','dynamic = true')
            text=text.replace('cfl_number = 0.25','cfl_number = 0.00001')
            text=text.replace('nlim = 3','nlim = 2')
            inp=d/'input';inp.write_text(text)
            run=subprocess.run([EXE,'-i',str(inp),'-d',str(d)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            h=athena_read.hst(d/'matched_bondi.user.hst')
            self.assertGreater(h['diagnostic_state_time'][-1],h['diagnostic_metric_time'][-1])
            np.testing.assert_allclose(h['diagnostic_metric_time'][-1],h['time'][-2],rtol=1e-14)
            np.testing.assert_equal(h['diagnostic_geometry_time'],h['time'])

if __name__=='__main__':unittest.main()
