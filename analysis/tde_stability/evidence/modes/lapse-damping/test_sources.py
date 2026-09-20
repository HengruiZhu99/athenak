from pathlib import Path
import sys,os,subprocess,json,hashlib,re
import numpy as np
root=Path(__file__).resolve().parent
repo=root.parents[2]
sys.path.insert(0,str(repo/'tst/regression'))
from z4c_hamiltonian_balance import BASE,snapshot,active
exe=Path('/Users/hz0693/research/TDE/build-lapse-damping/src/athena')
old=repo/'review/stability-isolation-20260919/athena-before-source-order'
def setv(s,b,k,v):
 m=re.search(rf'(<{b}>\n)(.*?)(?=\n<|\Z)',s,re.S);assert m
 body=m[2];pat=rf'(?m)^{k}\s*=.*$'
 body=re.sub(pat,f'{k} = {v}',body) if re.search(pat,body) else body+f'\n{k} = {v}\n'
 return s[:m.start()]+m[1]+body+s[m.end():]
base=BASE.format(balance='false',lapse_amplitude=0,theta_amplitude=0,zero_matter='true')
for axis in (1,2,3):base=setv(base,'meshblock',f'nx{axis}',16)
base=setv(base,'z4c','shift_Gamma',2);base=setv(base,'time','cfl_number',.15)
a=np.zeros((1,25,24,24,24));q=-2+(np.arange(24)-4+.5)*.25
z,y,x=np.meshgrid(q,q,q,indexing='ij');pulse=1e-6*np.exp(-((x-.75)**2+y*y+z*z)/.5**2)
a[0,17]=pulse;a[0,14]=pulse;a[0,15]=-.7*pulse;a[0,16]=.3*pulse
raw=root/'theta-gamma-input.bin';a.tofile(raw)
def run(name,opts,binary=exe):
 d=root/'regression-v2'/name;d.mkdir(parents=True,exist_ok=False);s=base
 for b,k,v in opts:s=setv(s,b,k,v)
 (d/'input.athinput').write_text(s)
 with (d/'run.log').open('w') as f:
  subprocess.run([str(binary),'-i','input.athinput'],cwd=d,env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
 return d
zero=run('zero_scaled',[('z4c','damp_lapse_scaled','true')])
for st in (1,2,3):
 for op in ('pre_rhs_state','volume_rhs','post_recast'):
  m,v=snapshot(zero,op,st);assert np.count_nonzero(v)==0,(op,st,np.max(abs(v)))
opts=[('problem','mode_analysis','true'),('problem','mode_input_state',raw)]
off=run('mixed_off',opts+[('z4c','damp_lapse_scaled','false')])
on=run('mixed_on',opts+[('z4c','damp_lapse_scaled','true')])
m,vf=snapshot(off,'rhs_full_vs_bg');m,r0=snapshot(off,'volume_rhs');_,r1=snapshot(on,'volume_rhs');vf=active(m,vf);delta=active(m,r1-r0)
expected=np.zeros_like(delta);factor=.1*(1-vf[:,18]);expected[:,7]=factor*vf[:,17];expected[:,17]=-2*factor*vf[:,17]
for j in (14,15,16):expected[:,j]=-2*factor*vf[:,j]
# Field index7 is Khat; gtilde is unchanged identity, so Gamma_metric=0.
errors={str(j):float(np.max(abs(delta[:,j]-expected[:,j]))) for j in (7,14,15,16,17)}
assert max(errors.values())<2e-15,errors
changed=np.array([7,14,15,16,17]);others=np.setdiff1d(np.arange(25),changed)
assert np.array_equal(delta[:,others],np.zeros_like(delta[:,others]))
assert np.array_equal(snapshot(off,'pre_rhs_state')[1],snapshot(on,'pre_rhs_state')[1])
# Default-off must preserve the prior executable's actual all-stage arithmetic.
seed=[('problem','vacuum_gauge_pulse_amplitude',1e-8)]
legacy=run('legacy',seed,old);default=run('default',seed)
for st in (1,2,3):
 for op in ('pre_rhs_state','volume_rhs','post_recast'):
  assert np.array_equal(snapshot(legacy,op,st)[1].view('u8'),snapshot(default,op,st)[1].view('u8')),(op,st)
# An initially constraint-satisfying atmosphere's matter RHS must be untouched.
mat=[('mhd','zero_tmunu_feedback','false'),('problem','zero_tmunu','false')]
mo=run('matter_off',mat+[('z4c','damp_lapse_scaled','false')]);mn=run('matter_on',mat+[('z4c','damp_lapse_scaled','true')])
assert np.array_equal(snapshot(mo,'volume_rhs')[1],snapshot(mn,'volume_rhs')[1])
res={'passed':True,'source_max_absolute_errors':errors,'zero_all_3_stages_including_ghosts':True,'default_all_stage_bitwise_legacy':True,'initial_matter_rhs_bitwise_unchanged':True,'unmodified_rhs_fields_bitwise_unchanged':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'notes':'Constraint source-product regression only; no stability claim.'}
(root/'source-regression.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
