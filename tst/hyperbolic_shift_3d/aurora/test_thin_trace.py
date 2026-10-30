"""Offline V12 descriptor/environment checks; no scheduler or remote actions."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import control,worker,preflight,struct
from types import SimpleNamespace
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
                    ('HYPERBOLIC_SPARSE_POISON','yes'),('LD_PRELOAD','evil'),('HYPERBOLIC_TRACE_KAPPA',True),
                    ('HYPERBOLIC_TRACE_PROFILES','2'),('HYPERBOLIC_TRACE_PROFILES','nan')]:
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
 def test_profiles_are_explicit_read_only_output_option(self):
  before=control.normal_derivative_config(self.command)
  self.command['environment']={'HYPERBOLIC_TRACE_PROFILES':'1'}
  self.assertEqual(control.command_environment(self.command)['HYPERBOLIC_TRACE_PROFILES'],'1')
  self.assertEqual(control.normal_derivative_config(self.command),before)
class ThinTracePreflightTests(unittest.TestCase):
 def test_gpu_component_contract(self):
  fixture='backend=SYCL\nPASS: CPU/device physical RHS, partition norms, zero coefficient and exact Minkowski; max_error=1e-16 scaled_error=1e-16 norm_error=1e-17\n'
  self.assertTrue(preflight.parse_component(fixture,'paired_trace_device',0)['passed'])
  self.assertFalse(preflight.parse_component(fixture,'paired_trace_device',1)['passed'])
  self.assertFalse(preflight.parse_component(fixture.replace('SYCL','Serial'),'paired_trace_device',0)['passed'])
  for bad in [fixture+fixture,fixture.replace('1e-17','nan'),fixture.replace('1e-17','3e-12'),fixture.replace(' scaled_error=1e-16','')]:
   with self.assertRaises(ValueError):preflight.parse_component(bad,'paired_trace_device',0)
 def test_v12_shape_and_report_binding(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);n=12
   for layers in (1,2,3):
    cells=12*n**3+8*(n+layers)**3+6*(n+3)*(n+6)**2;surface=31*6*n*n
    header=preflight.HEADER.pack(b'HYP3D_MPI_V12'.ljust(16,b'\0'),n,31,3,1,cells,surface,2,.0005,.00025)
    header+=struct.pack('<dii4i4i2ddii4i2d',2.,7,0,3,7,4,3,4,2,3,3,.18,.16,1e-10,1,0,1,layers,1,0,1.,.25)
    for name in ('a','b'):
     with (root/name).open('wb') as stream:stream.write(header);stream.truncate(184+16*(31*cells+surface))
    parsed=preflight.read_checkpoint(root/'a')
    self.assertEqual(parsed['header_bytes'],184);self.assertEqual(parsed['core_evolved_layers'],layers)
    report=dict(parsed,status='completed',backend='SYCL',mpi_size=12,kind='oblique',t=.0005,target_time=.0005,
                checkpoint_schema=parsed['magic'],checkpoint_header_bytes=184,
                algebraic_projection='final_stage_standard',scri_angular_stencil='cross_face_centered5')
    for name in ('a.json','b.json'):(root/name).write_text(json.dumps(report))
    args=SimpleNamespace(reference='a',candidate='b',reference_report='a.json',candidate_report='b.json',atol=0,rtol=0)
    self.assertTrue(preflight.restart(args,0,root,lambda p:root/p)['passed'])
    for key,bad in [('core_evolved_layers',float(layers)),('shifted_overlap_donors',1),('trace_kappa',0.),('trace_normal_ratio',0.),('interface_method','none')]:
     for remove in (False,True):
      changed=dict(report)
      if remove:changed.pop(key)
      else:changed[key]=bad
      (root/'b.json').write_text(json.dumps(changed))
      with self.subTest(key=key,layers=layers),self.assertRaises(ValueError):preflight.restart(args,0,root,lambda p:root/p)
    for offset,fmt,value in [(152,'i',0),(156,'i',0),(156,'i',4),(160,'i',2),(164,'i',1),(168,'d',float('nan')),(168,'d',-1),(176,'d',-1)]:
     with (root/'a').open('r+b') as stream:stream.write(header);stream.seek(offset);stream.write(struct.pack('<'+fmt,value))
     with self.subTest(offset=offset,value=value),self.assertRaises(ValueError):preflight.read_checkpoint(root/'a')
if __name__=='__main__':unittest.main()
