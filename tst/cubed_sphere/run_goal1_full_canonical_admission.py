#!/usr/bin/env python3
"""Local v3 all-direction full-Pi dissipation admission; no acceptance runs."""
from pathlib import Path
import hashlib,json,resource
from run_conformal_spherical import run_case
root=Path(__file__).resolve().parents[2];work=root/'build-goal1-minkowski-20261018/full-canonical-v3';ev=root/'docs/validation/goal1-minkowski-20261018/full-canonical-v3'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
assert json.loads((ev/'point-review.json').read_text())['passed']
assert json.loads((ev/'active-line-review.json').read_text())['passed']
r=json.loads((ev/'review.json').read_text());assert r['full_native_3D_wave_reduction_pass'] and r['v3_algebra_dissipation_pass']
binary=work/'athena_full_v3';tag=hashlib.sha256(binary.read_bytes()).hexdigest();assert tag==json.loads((ev/'native-build.json').read_text())['binary_sha256']
out=work/'native-admission';out.mkdir(exist_ok=True)
cases=[('v2-control4',.001,False,4,None),('full-stationary4',0.,True,4,None),('full-pulse4',.001,True,4,None),('full-resume8',.001,True,8,out/'full-pulse4-segment0/probe_checkpoint.bin')]
for family,amp,full,steps,parent in cases:
 for segment in range(3):
  case=family+'-segment'+str(segment);d=out/case
  if (d/'receipt.json').exists():r=json.loads((d/'receipt.json').read_text());print('reuse',case,flush=True)
  else:
   extra={'cubed_sphere/outer_radius':.98,'cubed_sphere/radial_stretch':2.8592697712828787,
    'cubed_sphere/conformal_fixed_dt':1e-5,'cubed_sphere/conformal_probe_steps':steps,
    'cubed_sphere/conformal_radial_sbp':True,'cubed_sphere/conformal_canonical_ko':True,'cubed_sphere/conformal_canonical_full':full,
    'cubed_sphere/conformal_quadrupole':False,'cubed_sphere/conformal_checkpoint':str(d/'probe_checkpoint.bin'),
    'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':45}
   if parent is not None:
    pr=json.loads((parent.parent/'receipt.json').read_text());assert pr['reference_unchanged'];assert hashlib.sha256(parent.read_bytes()).hexdigest()==pr['checkpoint_payload']['sha256'];extra['cubed_sphere/conformal_restart']=str(parent)
   r=run_case(binary,work/'source',out,case,'0.0',8,48,8,amp,10,.1,.001,5,60,.03,extra)
  if r['outcome']=='complete':break
  if r['outcome']!='checkpointed':raise RuntimeError('full canonical admission failure retained: '+case)
  parent=d/'probe_checkpoint.bin'
 else:raise RuntimeError('bounded full canonical admission incomplete after3segments')
