"""Controller integration tests with a fake allocator; no simulations are launched."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
from decimal import Decimal
import controller as c

class Workflow(unittest.TestCase):
    def exercise(self, failure=None, adopt=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);base=root/'base';base.mkdir();campaign=root/'campaign';campaign.mkdir()
        (base/'template.athinput').write_text(Path(__file__).with_name('baseline.athinput').read_text())
        for name in ['athena.history_extrema','source-base.txt','source.patch','run_cycle.sh']:(base/name).write_text('test fixture\n')
        (campaign/'BOUNDARY_QUALIFIED.json').write_text(json.dumps({'executable_sha256':c.sha(base/'athena.history_extrema'),'approved_configuration':{'boundary_rhs':'full_constraint_bjorhus','extrap_order':2,'vc_single_rank_device_sync':True}}))
        calls=[]
        class Allocator:
            pid=123
            def __init__(self,cmd,**kwargs):
                self.case=Path(cmd[-1]);calls.append(cmd)
            def wait(self):
                d=self.case;amp=Decimal((d/'amplitude.txt').read_text())
                finaltime=200;value=.005 if amp<=Decimal('-.048') else .9
                if d.name=='cycle_01':
                    if failure=='incomplete':finaltime=199
                    if failure=='nonfinite':value=float('nan')
                    if failure=='crash':return 1
                (d/'run-status').write_text('0\n');(d/'job-id.txt').write_text(str(len(calls)))
                (d/'stdout.log').write_text('Terminating on time limit\n')
                (d/'test.hst').write_text(f'# [1]=time [2]=minLapse\n0 1\n{finaltime} {value}\n')
                (d/'rst').mkdir();(d/'rst/final.rst').write_bytes(b'fixture'*200)
                (d/'initial.coefficients').write_text('fixture')
                return 0
        argv=['controller','--baseline',str(base)]
        if adopt:
            reused=root/'qualified_super';reused.mkdir()
            (reused/'amplitude.txt').write_text('-.05')
            (reused/'input.athinput').write_text(c.campaign_input((base/'template.athinput').read_text(),reused))
            Allocator([str(reused)]).wait();calls.clear()
            (reused/'provenance.json').write_text(json.dumps({'exe_sha256':c.sha(base/'athena.history_extrema'),'input_sha256':c.sha(reused/'input.athinput')}))
            (reused/'inputs.sha256').write_text('\n'.join(c.sha(reused/name)+'  '+name for name in ['input.athinput','initial.coefficients']))
            argv+=['--adopt-super',str(reused)]
        with patch.object(c,'R',campaign),patch('sys.argv',argv),patch.object(c.subprocess,'Popen',Allocator),patch.object(c.subprocess,'run'):
            if failure:
                with self.assertRaises((ValueError,RuntimeError)):c.main()
            else:c.main()
        return json.loads((campaign/'state.json').read_text()),calls

    def test_automatic_successors_and_requested_precision(self):
        state,calls=self.exercise()
        self.assertEqual(state['status'],'COMPLETE')
        self.assertLessEqual(Decimal(state['relative_width']),Decimal('1e-5'))
        self.assertEqual([x['amplitude'] for x in state['completed'][:4]],['-0.047','-0.05','-0.0485','-0.04775'])
        self.assertGreater(len(calls),4)
        self.assertTrue(all('--qos=shared_interactive' in x for x in calls))

    def test_completed_super_is_reused_without_an_allocation(self):
        state,calls=self.exercise(adopt=True)
        self.assertEqual(state['status'],'COMPLETE')
        self.assertTrue(state['completed'][1]['adopted'])
        self.assertEqual(len(state['completed']),len(calls)+1)
        self.assertFalse(any(Path(cmd[-1]).name=='endpoint_super' for cmd in calls))

    def test_failed_midpoints_never_update_bracket_or_submit_successors(self):
        for failure in ['incomplete','nonfinite','crash']:
            with self.subTest(failure=failure):
                state,calls=self.exercise(failure)
                self.assertEqual(state['status'],'FAILED')
                self.assertEqual(len(state['completed']),2)
                self.assertEqual(len(calls),3)
                self.assertEqual(state['sub'],'-0.047');self.assertEqual(state['super'],'-0.05')

if __name__=='__main__':unittest.main()
