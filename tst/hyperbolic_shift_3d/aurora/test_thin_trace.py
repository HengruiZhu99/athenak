"""Offline V12 descriptor/environment checks; no scheduler or remote actions."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import control,worker
from test_control import example
class ThinTraceAdmissionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.plan=example();self.command=self.plan['commands'][0]
  self.command['executable']='build/'+control.THIN_SPARSE_TARGET
  self.command['args'].extend(['.05','.001','6','12','2'])
 def test_defaults_and_oblique(self):
  config=control.normal_derivative_config(self.command)
  self.assertEqual(config['checkpoint_schema'],'HYP3D_MPI_V12');self.assertEqual(config['checkpoint_header_bytes'],184)
  self.assertEqual(config['core_evolved_layers'],1);self.assertEqual(config['trace_kappa'],0)
  self.assertEqual(config['volume_exchange'],'cached_sparse_alltoallv')
  self.command['kind']='oblique';self.command['args'][2]='oblique';control.validate_plan(self.root,self.plan)
  self.command['executable']='build/'+control.SPARSE_POSITIVE_TARGET
  with self.assertRaisesRegex(ValueError,'Oblique'):control.validate_plan(self.root,self.plan)
 def test_reject_invalid_environment(self):
  for key,value in [('HYPERBOLIC_TRACE_LAYERS','1.5'),('HYPERBOLIC_TRACE_LAYERS','0'),('HYPERBOLIC_TRACE_SHIFTED','2'),
                    ('HYPERBOLIC_TRACE_KAPPA','nan'),('HYPERBOLIC_TRACE_KAPPA','-1'),('HYPERBOLIC_TRACE_NORMAL_RATIO','inf'),
                    ('HYPERBOLIC_SPARSE_POISON','yes'),('LD_PRELOAD','evil'),('HYPERBOLIC_TRACE_KAPPA',True)]:
   self.command['environment']={key:value}
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):control.validate_plan(self.root,self.plan)
 def test_admission_binds_exact_penalty_geometry(self):
  self.plan.update(phase='stability',admission='admission.json')
  self.command['environment']={'HYPERBOLIC_TRACE_LAYERS':'2','HYPERBOLIC_TRACE_KAPPA':'1'}
  config=control.normal_derivative_config(self.command);source='a'*64;binary='b'*64
  admission=dict(passed=True,phase='preflight',source_manifest_sha256=source,low_resolution_n=16,
   binaries={self.command['executable']:binary},gpu_cpu_parity=True,restart_equivalence=True,normal_derivative_configs=[config])
  control.atomic(self.root/'admission.json',admission)
  control.validate_admission(self.root,self.plan,source,{self.command['executable']:binary})
  for key,value in [('trace_kappa',0.),('trace_normal_ratio',0.),('core_evolved_layers',1),('core_evolved_layers',2.),
                    ('shifted_overlap_donors',1),('checkpoint_header_bytes',152),('volume_exchange','full_allgatherv'),('physical_lapse_floor',0.)]:
   admission['normal_derivative_configs']=[dict(config,**{key:value})];control.atomic(self.root/'admission.json',admission)
   with self.subTest(key=key),self.assertRaises(ValueError):control.validate_admission(self.root,self.plan,source,{self.command['executable']:binary})
 def test_worker_uses_bound_environment_without_leakage(self):
  log=self.root/'env.log'
  with patch.dict(os.environ,{'HYPERBOLIC_TRACE_KAPPA':'999','HYPERBOLIC_SPARSE_POISON':'1'}):
   result=worker.run_bounded([sys.executable,'-c','import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("HYPERBOLIC_TRACE_") or k=="HYPERBOLIC_SPARSE_POISON"}))'],self.root,log,10,environment=control.command_environment(self.command))
  self.assertEqual(result['returncode'],0)
  self.assertEqual(json.loads(log.read_text()),control.TRACE_ENV)
if __name__=='__main__':unittest.main()
