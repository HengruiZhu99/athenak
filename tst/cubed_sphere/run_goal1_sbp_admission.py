#!/usr/bin/env python3
"""Bounded local Serial admission, retaining/reusing every completed segment."""
import hashlib,json,resource
from pathlib import Path
from run_conformal_spherical import run_case
root=Path(__file__).resolve().parents[2];work=root/'build-goal1-minkowski-20261018/sbp-v1';evidence=root/'docs/validation/goal1-minkowski-20261018/sbp-v1'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
protocol=json.loads((evidence/'protocol-v3.json').read_text());assert protocol['native_admission']['cut']==.98
assert json.loads((evidence/'geometry-preflight-v3.json').read_text())['cut_exceeds_layer_end']
op=json.loads((evidence/'operator-review.json').read_text());assert op['passed']
assert hashlib.sha256((work/'source/src/mesh/cubed_sphere/research_radial_sbp.hpp').read_bytes()).hexdigest()==op['source_sha256']['tst/cubed_sphere/goal1_radial_sbp.hpp']
out=work/'admission-v3';out.mkdir(exist_ok=True)
old=root/'build-goal1-minkowski-20261018/bin/native_control';new=work/'athena_sbp_v1'
for name,binary,amp,flag in [('frozen-pulse',old,.001,False),('control-pulse',new,.001,False),('sbp-stationary',new,0,True),('sbp-pulse',new,.001,True)]:
    parent=None;tag=hashlib.sha256(binary.read_bytes()).hexdigest()
    for segment in range(3):
        case=name+'-segment'+str(segment);d=out/case
        if (d/'receipt.json').exists():
            r=json.loads((d/'receipt.json').read_text());print('reuse',case,flush=True)
        else:
            extra={'cubed_sphere/outer_radius':.98,'cubed_sphere/radial_stretch':2.8592697712828787,
                'cubed_sphere/conformal_fixed_dt':1e-5,'cubed_sphere/conformal_probe_steps':4,
                'cubed_sphere/conformal_radial_sbp':flag,'cubed_sphere/conformal_quadrupole':False,
                'cubed_sphere/conformal_checkpoint':str(d/'probe_checkpoint.bin'),
                'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':45}
            if parent is not None:
                pr=json.loads((parent.parent/'receipt.json').read_text())
                assert hashlib.sha256(parent.read_bytes()).hexdigest()==pr['checkpoint_payload']['sha256'] and pr['reference_unchanged']
                extra['cubed_sphere/conformal_restart']=str(parent)
            r=run_case(binary,work/'source',out,case,'0.0',8,48,8,amp,10,.1,.001,5,60,.03,extra)
        if r['outcome']=='complete':break
        if r['outcome']!='checkpointed':raise RuntimeError('native admission failure preserved: '+case)
        parent=d/'probe_checkpoint.bin'
    else:raise RuntimeError('admission incomplete after three bounded segments')
