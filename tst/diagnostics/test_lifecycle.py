"""Actual patched executable lifecycle checks. Set ATHENA_DIAGNOSTICS_EXE to run."""
from pathlib import Path
import os, subprocess, sys, tempfile, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
import athena_read
EXE=os.environ.get('ATHENA_DIAGNOSTICS_EXE')
INPUT=Path(__file__).with_name('smoke.athinput')

@unittest.skipUnless(EXE,'set ATHENA_DIAGNOSTICS_EXE to test patched executable')
class Lifecycle(unittest.TestCase):
    def run_case(self, directory, *args, ranks=1, restart=None, input_file=INPUT):
        directory.mkdir(parents=True)
        cmd=([os.environ.get('MPIEXEC','mpiexec'),'-n',str(ranks)] if ranks>1 else [])
        cmd += [EXE, '-r' if restart else '-i',str(restart or input_file),'-d',str(directory)]
        cmd += ['output1/data_format=%24.16e',*args]
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (directory/'run.log').write_text(result.stdout)
        self.assertEqual(result.returncode,0,result.stdout[-2000:])
        return athena_read.hst(directory/'matched_bondi.user.hst')
    def test_restart_cadence_ranks_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=self.run_case(root/'full','time/nlim=5')
            split=self.run_case(root/'split','time/nlim=2')
            rst=sorted((root/'split/rst').glob('*.rst'))[-1]
            resume=self.run_case(root/'resume','time/nlim=5',restart=rst)
            sparse=self.run_case(root/'sparse','time/nlim=5','output1/dcycle=3')
            key='M0_qualified_cumulative_BH1_Kerr'
            self.assertGreater(full[key][-1],0)
            self.assertGreater(full['M0_covered_duration_BH1_Kerr'][-1],0)
            np.testing.assert_allclose(resume[key][-1],full[key][-1],rtol=2e-12)
            np.testing.assert_allclose(sparse[key][-1],full[key][-1],rtol=2e-12)
            self.assertEqual(full['M0_committed_cycle_BH1_Kerr'][-1],5)
            missing=self.run_case(root/'missing','time/nlim=3','problem/flux_min_cells_per_radius=100')
            self.assertEqual(missing[key][-1],0)
            self.assertEqual(missing['M0_covered_duration_BH1_Kerr'][-1],0)
            self.assertEqual(missing['M0_unavailable_duration_BH1_Kerr'][-1],missing['time'][-1])
            self.assertTrue(np.isnan(missing['M0_net_in_moving_BH1_Kerr']).all())
            if os.environ.get('ATHENA_DIAGNOSTICS_MPI'):
                mpi=self.run_case(root/'mpi','time/nlim=5',ranks=2)
                changed=self.run_case(root/'changed','time/nlim=5',ranks=2,restart=rst)
                np.testing.assert_allclose(mpi[key][-1],full[key][-1],rtol=2e-12)
                np.testing.assert_allclose(changed[key][-1],full[key][-1],rtol=2e-12)

    def test_regional_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('budget_smoke.athinput')
            full=self.run_case(root/'full','time/nlim=5',input_file=inp)
            self.run_case(root/'split','time/nlim=2',input_file=inp)
            rst=sorted((root/'split/rst').glob('*.rst'))[-1]
            resumed=self.run_case(root/'resumed','time/nlim=5',restart=rst)
            for key in full:
                if key.startswith('budget_'):
                    np.testing.assert_allclose(resumed[key][-1],full[key][-1],atol=2e-12,rtol=2e-10)
            self.assertTrue(np.isnan(full['budget_near1_residual_dJ'][-1]))
            self.assertTrue(np.isfinite(full['budget_annulus_residual_dJ'][-1]))

    def test_schema_change_and_old_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            inp=root/'without-epoch-override'
            inp.write_text(INPUT.read_text().replace('flux_new_epoch = false\n',''))
            first=self.run_case(root/'first','time/nlim=2',input_file=inp)
            rst=sorted((root/'first/rst').glob('*.rst'))[-1]
            bad=subprocess.run([EXE,'-r',str(rst),'-d',str(root/'bad'),
                'problem/flux_min_cells_per_radius=2'],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('Changed diagnostic configuration',bad.stdout+bad.stderr)
            fresh=self.run_case(root/'fresh','time/nlim=3',restart=rst,
                *['problem/flux_min_cells_per_radius=2','problem/flux_new_epoch=true'])
            self.assertEqual(fresh['M0_epoch_start_BH1_Kerr'][-1],first['time'][-1])
            old=os.environ.get('ATHENA_DIAGNOSTICS_BASE_EXE')
            if old:
                (root/'old').mkdir()
                run=subprocess.run([old,'-i',str(INPUT),'-d',str(root/'old'),'time/nlim=2'],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                checkpoint=sorted((root/'old/rst').glob('*.rst'))[-1]
                new=self.run_case(root/'new','time/nlim=3',restart=checkpoint)
                self.assertEqual(new['M0_epoch_start_BH1_Kerr'][-1],first['time'][-1])
                self.assertIn('earlier accretion is unavailable',(root/'new/run.log').read_text())

    def test_partial_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            h=self.run_case(Path(temp)/'partial','time/nlim=1',
                'problem/flux_rsurf_inner=9','problem/flux_rsurf_outer=9')
            self.assertTrue(np.isnan(h['mdot_R9']).all())
            self.assertTrue(np.isnan(h['area_R9']).all())
            self.assertTrue((h['ownership_fraction_R9']<1).all())
            self.assertTrue((h['partial_covered_area_R9']>0).all())

    def test_injected_source_stage_weights(self):
        import bin_convert
        def moment(dump):
            data=bin_convert.read_binary(str(dump));total=0.0
            for mb,g in enumerate(data['mb_geometry']):
                axes=[np.linspace(g[2*a],g[2*a+1],9)[:-1]+0.5 for a in range(3)]
                Z,Y,X=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
                for dz in (-0.25,0.25):
                    for dy in (-0.25,0.25):
                        for dx in (-0.25,0.25):
                            x,y,z=X+dx,Y+dy,Z+dz
                            inside=(x-2)**2+y*y+z*z<1.2**2
                            outside=(x-2)**2+y*y/(1-0.25**2)+z*z>=0.4**2
                            total+=np.sum((x*data['mb_data']['dens'][mb])[inside&outside])/8
            return total
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('source_smoke.athinput')
            rk1=self.run_case(root/'rk1',input_file=inp)
            start=moment(root/'rk1/bin/matched_bondi.mhd_w.00000.bin')
            intermediate=moment(root/'rk1/bin/matched_bondi.mhd_w.00001.bin')
            dt=rk1['time'][-1];source=0.2
            np.testing.assert_allclose(rk1['budget_near1_integral_physical_dJ'][-1],
                dt*source*start,rtol=1e-7,atol=1e-12)
            rk2=self.run_case(root/'rk2','time/integrator=rk2',input_file=inp)
            np.testing.assert_allclose(rk2['budget_near1_integral_physical_dJ'][-1],
                0.5*dt*source*(start+intermediate),rtol=1e-7,atol=1e-12)

    def test_amr_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('amr_smoke.athinput')
            h=self.run_case(root/'serial','time/nlim=2',input_file=inp)
            for surface in ('R3.5','R4','H1','BH1_Kerr'):
                np.testing.assert_equal(h['ownership_fraction_'+surface],1)
            if os.environ.get('ATHENA_DIAGNOSTICS_MPI'):
                mpi=self.run_case(root/'mpi','time/nlim=2',input_file=inp,ranks=2)
                np.testing.assert_allclose(mpi['M0_qualified_cumulative_BH1_Kerr'][-1],
                    h['M0_qualified_cumulative_BH1_Kerr'][-1],rtol=2e-11,atol=1e-12)

    def test_zoom_pending_and_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('zoom_smoke.athinput')
            full=self.run_case(root/'full',input_file=inp)
            split=self.run_case(root/'split','time/nlim=2',input_file=inp)
            # The scheduled file precedes AMR; the final file at the same cycle follows it.
            pre=root/'split/rst/matched_bondi.00001.rst'
            post=root/'split/rst/matched_bondi.00002.rst'
            for label,rst in (('pending',pre),('completed',post)):
                resumed=self.run_case(root/label,'time/nlim=12',restart=rst)
                for key in ('M0_qualified_cumulative_BH1_Kerr','M0_covered_duration_BH1_Kerr',
                            'M0_unavailable_duration_BH1_Kerr'):
                    np.testing.assert_allclose(resumed[key][-1],full[key][-1],rtol=3e-10,atol=1e-12)
            self.assertGreater(full['M0_unavailable_duration_BH1_Kerr'][-1],0)
            self.assertTrue((full['reason_BH1_Kerr'].astype(int)&32).any())
            np.testing.assert_allclose(full['M0_covered_duration_BH1_Kerr']+
                full['M0_unavailable_duration_BH1_Kerr'],full['time'],atol=1e-13)

    def test_zoom_budget_jump_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            text=INPUT.with_name('zoom_smoke.athinput').read_text().replace('<problem>',
                '<problem>\nflux_budget=true\nbudget_r_inner=3.5\nbudget_r_outer=4\nbudget_subcells=2',1)
            inp=root/'budget-zoom-input';inp.write_text(text)
            full=self.run_case(root/'full',input_file=inp)
            self.run_case(root/'split','time/nlim=2',input_file=inp)
            for number in (1,2):
                resumed=self.run_case(root/f'resumed{number}','time/nlim=12',
                    restart=root/f'split/rst/matched_bondi.{number:05d}.rst')
                for region in ('near1','cavity','annulus'):
                    for channel in ('cumulative_quantified_remap_jump_dJ','cumulative_quantified_zoom_jump_dJ'):
                        key=f'budget_{region}_{channel}'
                        np.testing.assert_allclose(resumed[key][-1],full[key][-1],rtol=1e-9,atol=1e-10)
                    # A genuine restart in the frozen zone performs one additional
                    # startup recovery measurement. Its unavailable jump is an event,
                    # never an invented finite contribution to the cumulative budget.
                    events=f'budget_{region}_unquantifiable_events'
                    extra=1 if number==2 and region=='near1' else 0
                    self.assertEqual(resumed[events][-1],full[events][-1]+extra)
            self.assertGreater(full['budget_near1_unquantifiable_events'][-1],0)
            self.assertTrue(np.isfinite(full['budget_annulus_cumulative_quantified_remap_jump_dJ'][-1]))

    def test_reset_provenance_survives_repeated_checkpoint(self):
        # Manufacture an unlocalized current reset-provenance flag in an otherwise
        # healthy checkpoint. Exercise the production decoder, sampler, commit and
        # repeated pre-checkpoint export without changing an evolved variable.
        import re
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=self.run_case(root/'first','time/nlim=2')
            original=sorted((root/'first/rst').glob('*.rst'))[-1]
            data=original.read_bytes();end=data.index(b'<par_end>')+len(b'<par_end>\n')
            header,count=re.subn(rb'(source_flags\s*=\s*)[^\s#]+',
                lambda match:match.group(1)+b'0,0,0,16',data[:end])
            self.assertEqual(count,1)
            injected=root/'injected.rst';injected.write_bytes(header+data[end:])
            direct=self.run_case(root/'direct','time/nlim=3',restart=injected)
            stopped=self.run_case(root/'stopped','time/nlim=2',restart=injected)
            checkpoint=sorted((root/'stopped/rst').glob('*.rst'))[-1]
            resumed=self.run_case(root/'resumed','time/nlim=3',restart=checkpoint)
            key='M0_qualified_cumulative_BH1_Kerr'
            self.assertEqual(direct[key][-1],first[key][-1])
            self.assertEqual(resumed[key][-1],first[key][-1])
            self.assertGreater(direct['M0_unavailable_duration_BH1_Kerr'][-1],0)
            self.assertEqual(resumed['M0_interval_reason_BH1_Kerr'][-1],16)
            self.assertEqual(stopped['valid_BH1_Kerr'][-1],0)

if __name__=='__main__':unittest.main()
