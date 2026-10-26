#!/usr/bin/env python3
"""Bounded local finite-boundary SAT entry. Reuse compatible completed segments."""
from pathlib import Path
import hashlib,json,resource
from run_conformal_spherical import run_case
root=Path(__file__).resolve().parents[2];work=root/'build-goal1-minkowski-20261018/full-boundary-v4';ev=root/'docs/validation/goal1-minkowski-20261018/full-boundary-v4'
resource.setrlimit(resource.RLIMIT_CORE,(0,0));assert json.loads((ev/'point-review.json').read_text())['passed']
binary=work/'athena_boundary_v4';tag=hashlib.sha256(binary.read_bytes()).hexdigest();assert tag==json.loads((ev/'native-build.json').read_text())['binary_sha256']
out=work/'native-admission';out.mkdir(exist_ok=True)
cases=[('v3-control4',.001,False,4,None),('SAT-stationary4',0.,True,4,None),('SAT-pulse4',.001,True,4,None),('SAT-resume8',.001,True,8,out/'SAT-pulse4-segment0/probe_checkpoint.bin')]
for family,amp,sat,steps,parent in cases:
 for segment in range(3):
  name=family+'-segment'+str(segment);case=out/name
  if (case/'receipt.json').exists():r=json.loads((case/'receipt.json').read_text());print('reuse',name,flush=True)
  else:
   extra={'cubed_sphere/outer_radius':.98,'cubed_sphere/radial_stretch':2.8592697712828787,
    'cubed_sphere/conformal_fixed_dt':1e-5,'cubed_sphere/conformal_probe_steps':steps,'cubed_sphere/conformal_radial_sbp':True,
    'cubed_sphere/conformal_canonical_ko':True,'cubed_sphere/conformal_canonical_full':True,'cubed_sphere/conformal_boundary_sat':sat,
    'cubed_sphere/conformal_quadrupole':False,'cubed_sphere/conformal_checkpoint':str(case/'probe_checkpoint.bin'),
    'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':45}
   if parent is not None:
    pr=json.loads((parent.parent/'receipt.json').read_text());assert pr['reference_unchanged'];assert hashlib.sha256(parent.read_bytes()).hexdigest()==pr['checkpoint_payload']['sha256'];extra['cubed_sphere/conformal_restart']=str(parent)
   r=run_case(binary,work/'source',out,name,'0.0',8,48,8,amp,10,.1,.001,5,60,.03,extra)
  if r['outcome']=='complete':break
  if r['outcome']!='checkpointed':raise RuntimeError('finite boundary admission failure retained: '+name)
  parent=case/'probe_checkpoint.bin'
 else:raise RuntimeError('finite boundary entry incomplete after3segments')
