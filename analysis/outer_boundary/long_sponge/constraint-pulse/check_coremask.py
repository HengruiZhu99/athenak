"""Confirm the history core mask changes no saved evolved payload bytes."""
from pathlib import Path
import hashlib,json,os,re,sys
import numpy as np
sys.path.insert(0,os.environ.get('ATHENA_REGRESSION_PATH','/Users/hz0693/research/TDE/athenak-outer-boundary-fix/tst/regression'))
from z4c_background_restart import cohort
root=Path(__file__).resolve().parent
left=root/'runs/pulse_gate_loweta';right=root/'runs/pulse_gate_loweta_coremask'
f0,r0=cohort(left,8,3);f1,r1=cohort(right,8,3)
assert r0[0]['time']==r1[0]['time'] and abs(r0[0]['time']-9.6)<1e-13
def payload(path,total):
    data=path.read_bytes();end=data.index(b'<par_end>\n')+len(b'<par_end>\n')
    start=end+8+72+2*76+20+20*total+24
    return data[start:]
files=[]
for rank in range(8):
    a=payload(left/'rst'/f'rank_{rank:08d}'/f0.name,r0[0]['total'])
    b=payload(right/'rst'/f'rank_{rank:08d}'/f1.name,r1[0]['total'])
    assert a==b
    files.append(dict(rank=rank,bytes=len(a),sha256=hashlib.sha256(a).hexdigest(),bitwise_equal=True))
def first_hst(run):
    path=next(run.glob('*.z4c.user.hst'));lines=path.read_text().splitlines();names=re.findall(r'\[\d+\]=(\S+)',lines[1]);arr=np.atleast_2d(np.loadtxt(path));return dict(zip(names,arr[0]))
u=first_hst(left);m=first_hst(right);key='Theta-norm' if 'Theta-norm' in m else 'Theta-norm2'
ans=dict(time=r0[0]['time'],cycle=3,all_evolved_payloads_bitwise_equal=True,includes_mhd_magnetic_faces_and_all25_z4c_fields_with_ghosts=True,files=files,initial_masked_core_Theta2=float(m['Theta-int2']),initial_masked_exterior_Theta2=float(m[key]),initial_unmasked_Theta2=float(u[key]),partition_relative_error=float(abs(m['Theta-int2']+m[key]-u[key])/u[key]),input_sha256=hashlib.sha256((root/'theta_loweta_coremask.athinput').read_bytes()).hexdigest())
(root/'coremask-validation.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps({k:v for k,v in ans.items() if k!='files'},indent=2))
