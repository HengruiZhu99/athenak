"""Controller state/duplicate guards, with all scheduler calls mocked."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import advance

class Guards(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=advance.HERE;advance.HERE=self.root
 def tearDown(self):advance.HERE=self.old;self.temp.cleanup()
 def state(self,**kw):
  s={'phase':'prepared','current_job':None,'jobs':[],'source_commit':'fixture'};s.update(kw);advance.save(s);return s
 def test_running_no_duplicate(self):
  self.state(phase='submitted',current_job='1',jobs=['1'])
  with patch.object(advance,'verify_package'),patch.object(advance,'scheduler',return_value={'state':'R','exit':None}),patch.object(advance.subprocess,'run')as run:
   advance.main(True);run.assert_not_called()
 def test_uncertain_no_retry(self):
  self.state(phase='submission_uncertain')
  with patch.object(advance.subprocess,'run')as run:
   with self.assertRaises(ValueError):advance.main(True)
   run.assert_not_called()
 def test_initial_submission_once(self):
  self.state()
  answers=[SimpleNamespace(returncode=0,stdout='',stderr=''),SimpleNamespace(returncode=0,stdout='123.pbs\n',stderr='')]
  with patch.object(advance,'verify_package'),patch.object(advance.subprocess,'run',side_effect=answers)as run:
   advance.main(True);self.assertEqual(run.call_count,2)
  s=json.loads((self.root/'state.json').read_text());self.assertEqual(s['current_job'],'123');self.assertEqual(s['jobs'],['123'])
 def test_uncertain_qsub_retained(self):
  self.state();answers=[SimpleNamespace(returncode=0,stdout='',stderr=''),SimpleNamespace(returncode=1,stdout='',stderr='connection interrupted')]
  with patch.object(advance,'verify_package'),patch.object(advance.subprocess,'run',side_effect=answers):
   with self.assertRaises(ValueError):advance.main(True)
  self.assertEqual(json.loads((self.root/'state.json').read_text())['phase'],'submission_uncertain')
 def test_failed_job_pauses(self):
  self.state(phase='submitted',current_job='1',jobs=['1'])
  with patch.object(advance,'verify_package'),patch.object(advance,'scheduler',return_value={'state':'F','exit':143}),patch.object(advance.subprocess,'run')as run:
   with self.assertRaises(ValueError):advance.main(True)
   run.assert_not_called()
  self.assertEqual(json.loads((self.root/'state.json').read_text())['phase'],'paused_failure')
 def test_other_queue_pending_defers(self):
  self.state()
  with patch.object(advance,'verify_package'),patch.object(advance.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='999.pbs\n',stderr=''))as run:
   advance.main(True);self.assertEqual(run.call_count,1)
  self.assertEqual(json.loads((self.root/'state.json').read_text())['phase'],'prepared')
 def test_missing_manifest_fails(self):
  self.state()
  with self.assertRaises(FileNotFoundError):advance.verify_package()

if __name__=='__main__':unittest.main()
