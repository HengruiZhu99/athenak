from pathlib import Path
import tempfile,struct,unittest
from unittest.mock import patch
import resume_case as r
class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.case=Path(self.tmp.name)
        for name,value in [('run-status','0'),('job-id.txt','123'),('stdout.log','Terminating on wall clock limit\n'),('amr_history.jsonl','{}'),('input.athinput','fixture'),('initial.coefficients','fixture')]:
            (self.case/name).write_text(value)
        (self.case/'run.hst').write_text('# [1]=time [2]=minLapse [3]=cycle\n0 1 0\n80 .001 3\n')
        (self.case/'inputs.sha256').write_text('\n'.join(r.sha(self.case/name)+'  '+name for name in ['input.athinput','initial.coefficients']))
        payload=bytearray(252);struct.pack_into('<ii',payload,0,1,3);struct.pack_into('<4i',payload,80,4,128,256,1);struct.pack_into('<ddi',payload,232,80,.001,3)
        (self.case/'rst').mkdir();(self.case/'rst/run.rst').write_bytes(b'history_bytes = 2\n<par_end>\n'+payload)
    def test_clean_wall_stop(self):
        _,info=r.validate_checkpoint(self.case);self.assertEqual(info['time'],80)
    def test_crash_is_not_resumed(self):
        (self.case/'run-status').write_text('1')
        with self.assertRaises(RuntimeError):r.validate_checkpoint(self.case)
    def test_history_after_checkpoint_rejected(self):
        (self.case/'run.hst').write_text('# [1]=time [2]=minLapse [3]=cycle\n81 .001 4\n')
        with self.assertRaises(RuntimeError):r.validate_checkpoint(self.case)
    def test_modified_inputs_rejected(self):
        (self.case/'input.athinput').write_text('modified')
        with self.assertRaises(RuntimeError):r.validate_checkpoint(self.case)
    def test_extra_amr_tail_is_not_silently_truncated(self):
        (self.case/'amr_history.jsonl').write_text('{}\n')
        with self.assertRaises(RuntimeError):r.validate_checkpoint(self.case)
    def test_nonfinite_history_rejected(self):
        (self.case/'run.hst').write_text('# [1]=time [2]=minLapse [3]=cycle\n80 nan 3\n')
        with self.assertRaises(ValueError):r.validate_checkpoint(self.case)
    def test_active_allocation_is_not_duplicated(self):
        with patch('sys.argv',['resume_case','--case',str(self.case),'--previous-job','123']),patch.object(r.subprocess,'check_output',return_value='123\n'):
            with self.assertRaisesRegex(RuntimeError,'still active'):r.main()
if __name__=='__main__':unittest.main()
