import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import control
import worker
import damping_ablation_campaign as c

class DirectCampaignTests(unittest.TestCase):
    def test_checkpoint_rejects_wrong_schema_and_unhealthy_state(self):
        n=16;nc=2;ns=6*n*n*31
        data=bytearray(struct.pack('<16s4i3Q2d',b'HYP3D_MPI_DG1',n,31,2,1,nc,ns,20,.01,.0005)+c.CONFIG)
        data+=struct.pack('<d',1.)*(2*(31*nc+ns))
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'cells',return_value=nc):
            p=Path(tmp)/'input';p.write_bytes(data)
            self.assertEqual(c.inspect_checkpoint(p,n,hashlib.sha256(data).hexdigest())['time'],.01)
            for offset,fmt,value in [(0,'<16s',b'HYP3D_MPI_DO1'),(0,'<16s',b'HYP3D_MPI_IT1'),(0,'<16s',b'HYP3D_MPI_GD1'),(0,'<16s',b'HYP3D_MPI_GW1'),(0,'<16s',b'HYP3D_MPI_AC1'),
                    (28,'<i',0),(56,'<d',0.),(64,'<d',.001),(144,'<i',1),
                    (152,'<d',0.),(168,'<d',-1.),(176,'<d',float('nan'))]:
                bad=bytearray(data);struct.pack_into(fmt,bad,offset,value);p.write_bytes(bad)
                with self.assertRaises(ValueError):c.inspect_checkpoint(p,n,hashlib.sha256(bad).hexdigest())

    def test_fresh_first_then_only_verified_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'initial').write_text('original')
            sha=control.digest(root/'initial')
            for n in [16]:
                with patch.dict(c.INITIAL_SHA,{n:sha}):
                    made=c.make_plan(root,n,1,initial='initial')
                import json
                plan=json.loads((root/made['plan']).read_text())
                self.assertEqual(plan['start_verification']['time'],0.)
                control.validate_plan(root,plan)
                self.assertIsNone(control.validate_admission(root,plan,'',{}))
                for index,value in [(3,'1'),(3,str(20/3)),(4,'.001'),(5,'-'),(6,'old3M'),(12,'1')]:
                    bad=copy.deepcopy(plan);bad['commands'][0]['args'][index]=value
                    with self.assertRaises(ValueError):control.validate_plan(root,bad)
                bad=copy.deepcopy(plan);bad['n']=24
                with self.assertRaises(ValueError):control.validate_plan(root,bad)
                bad=copy.deepcopy(plan);bad['commands'][0]['executable']='build/'+control.SMOOTH_GAUGE_TARGET
                with self.assertRaises(ValueError):control.validate_plan(root,bad)
                with self.assertRaises(ValueError):c.make_plan(root,n,1,restart='old3M',expected_sha=sha)
                with self.assertRaises(ValueError):c.make_plan(root,n,2,initial='initial')
                with patch.object(c,'inspect_checkpoint',return_value=dict(n=n,time=.01,steps=20,dt=.0005*16/n,bytes=1000,sha256=sha)):
                    continuation=c.make_plan(root,n,2,restart='initial',expected_sha=sha)
                p=json.loads((root/continuation['plan']).read_text());control.validate_plan(root,p)
                checks=[dict(check=x,passed=x['path']!=['status']) for x in p['commands'][0]['checks']]
                self.assertEqual(worker.classify(3,False,dict(status='checkpointed'),checks),'checkpointed')
                checks[-1]['passed']=False
                self.assertEqual(worker.classify(3,False,dict(status='checkpointed'),checks),'failed')

if __name__=='__main__':unittest.main()
