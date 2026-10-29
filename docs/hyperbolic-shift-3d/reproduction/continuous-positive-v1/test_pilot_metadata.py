#!/usr/bin/env python3
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=json.loads((HERE.parent/'aurora/stability-continuous-positive-puncture-v1/verification.json').read_text())
class AuditTests(unittest.TestCase):
    def run_case(self,data,success):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'v.json').write_text(json.dumps(data))
            r=subprocess.run([sys.executable,str(HERE/'audit_pilot_metadata.py'),'--verification',str(p/'v.json'),'--output',str(p/'out.json')],capture_output=True)
            self.assertEqual(r.returncode==0,success,r.stderr.decode())
            self.assertEqual((p/'out.json').exists(),success)
    def test_known_metadata_only_correction(self):self.run_case(BASE,True)
    def test_extra_numerical_failure_rejected(self):
        x=copy.deepcopy(BASE);x['checks'].append(dict(check=dict(path=['H_L2'],op='le',value=1),observed=2,passed=False));self.run_case(x,False)
    def test_changed_target_rejected(self):
        x=copy.deepcopy(BASE);x['report']['target_time']=1.;self.run_case(x,False)
    def test_floor_intervention_rejected(self):
        x=copy.deepcopy(BASE);x['report']['lapse_floor_node_interventions']=1;self.run_case(x,False)
    def test_unadmitted_checkpoint_rejected(self):
        x=copy.deepcopy(BASE);x['report']['checkpoint_admitted']=False;self.run_case(x,False)
    def test_payload_count_mismatch_rejected(self):
        x=copy.deepcopy(BASE);x['checkpoint']['cells']+=1;self.run_case(x,False)
    def test_unverified_artifacts_rejected(self):
        x=copy.deepcopy(BASE);x['verification_passed']=False;self.run_case(x,False)
if __name__=='__main__':unittest.main()
