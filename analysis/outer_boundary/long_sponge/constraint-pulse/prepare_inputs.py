"""Create fresh independent direct-Theta inputs; no runs or submissions."""
from pathlib import Path
import hashlib,json,re,math
import numpy as np

root=Path(__file__).resolve().parent
source=root.parent/'gpu/current/radial_k0.athinput'
base=source.read_text()
def setval(text,section,key,value):
    pattern=rf'(?ms)(^<{re.escape(section)}>\s*\n)(.*?)(?=^<|\Z)'
    m=re.search(pattern,text);assert m,(section,key)
    body=m.group(2);line=rf'(?m)^{re.escape(key)}\s*=.*$'
    if re.search(line,body):body=re.sub(line,f'{key} = {value}',body)
    else:body=body.rstrip()+f'\n{key} = {value}\n\n'
    return text[:m.start(2)]+body+text[m.end(2):]
def make(name,eta,ld,cfl,amp=1e-6,nlim=-1,tlim=50000,coremask=False):
    s=base
    updates={'job':{'basename':name},'time':{'cfl_number':cfl,'nlim':nlim,'tlim':tlim,'ndiag':20},
      'z4c':{'shift_eta':eta,'residual_lapse_damping':ld,'damp_kappa1':0,'damp_kappa2':0},
      'problem':{'vacuum_gauge_pulse_amplitude':0,'characteristic_test_amplitude':0,'characteristic_test_family':'none','outer_sponge_test_theta_pulse_amplitude':amp,'outer_sponge_test_theta_pulse_radius':0,'outer_sponge_test_theta_pulse_width':384,'outer_sponge_test_theta_pulse_dipole_axis':0},
      'output2':{'dt':1000}}
    if coremask:updates['z4c'].update(history_excise_ks_horizon='true',history_excise_ks_radius=512,history_excise_ks_spin=0)
    for sec,kv in updates.items():
        for k,v in kv.items():s=setval(s,sec,k,v)
    s+='''
<output3>
file_type = bin
variable = z4c_Theta
dt = 64
single_file_per_rank = true

<output4>
file_type = bin
variable = con
dt = 128
single_file_per_rank = true
'''
    return s

cases={
 'theta_standard':dict(eta=2,ld=.1,cfl=.009375),
 'theta_loweta':dict(eta=.02,ld=.01,cfl=.05),
 'zero_standard':dict(eta=2,ld=.1,cfl=.009375,amp=0,nlim=3),
 'zero_loweta':dict(eta=.02,ld=.01,cfl=.05,amp=0,nlim=3),
 'pulse_gate_standard':dict(eta=2,ld=.1,cfl=.009375,nlim=3),
 'pulse_gate_loweta':dict(eta=.02,ld=.01,cfl=.05,nlim=3),
 'theta_loweta_early_dt3p2':dict(eta=.02,ld=.01,cfl=.05,tlim=320),
 'theta_loweta_early_dt1p6':dict(eta=.02,ld=.01,cfl=.025,tlim=320),
 'theta_loweta_lapse01':dict(eta=.02,ld=.1,cfl=.05),
 'zero_loweta_lapse01':dict(eta=.02,ld=.1,cfl=.05,amp=0,nlim=3),
 'pulse_gate_loweta_lapse01':dict(eta=.02,ld=.1,cfl=.05,nlim=3),
 'theta_loweta_coremask':dict(eta=.02,ld=.01,cfl=.05,coremask=True),
 'theta_standard_coremask':dict(eta=2,ld=.1,cfl=.009375,coremask=True),
 'theta_loweta_lapse01_coremask':dict(eta=.02,ld=.1,cfl=.05,coremask=True),
 'pulse_gate_loweta_coremask':dict(eta=.02,ld=.01,cfl=.05,nlim=3,coremask=True),
 'zero_loweta_coremask':dict(eta=.02,ld=.01,cfl=.05,amp=0,nlim=3,coremask=True),
 'zero_loweta_lapse01_coremask':dict(eta=.02,ld=.1,cfl=.05,amp=0,nlim=3,coremask=True),
}
manifest={'base':str(source),'base_sha256':hashlib.sha256(base.encode()).hexdigest(),'cases':{}}
for name,kw in cases.items():
    p=root/(name+'.athinput');content=make(name,**kw)
    if p.exists():assert p.read_text()==content,f'Preserving different existing {p}'
    else:p.write_text(content)
    manifest['cases'][name]=dict(parameters=kw,input_sha256=hashlib.sha256(content.encode()).hexdigest())
(root/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Evaluate the exact active-grid seed and its initial regional support.
a=(np.arange(64)+.5)*64-2048;x,y,z=np.meshgrid(a,a,a,indexing='ij');r=np.sqrt(x*x+y*y+z*z)
t=1e-6*np.exp(-.5*(r/384)**2);w=t*t;bins=[0,512,1792,float('inf')]
regions={name:dict(theta_sq_fraction=float(w[(r>=lo)&(r<hi)].sum()/w.sum()),max_abs_theta=float(t[(r>=lo)&(r<hi)].max()),cells=int(((r>=lo)&(r<hi)).sum())) for name,lo,hi in zip(['protected_core','ramp','outer_plateau'],bins[:-1],bins[1:])}
result=dict(amplitude=1e-6,radius=0,width=384,dx=64,sigma_cells=6,FWHM_diameter=2*math.sqrt(2*math.log(2))*384,active_peak=float(t.max()),tail_at_radius={str(q):float(1e-6*math.exp(-.5*(q/384)**2)) for q in [0,512,1792,2048]},regions=regions,source_sigma_max=.001,source_dt_cap=1000,standard_eta_dt=1.2,loweta_eta_dt=.064)
(root/'seed-design.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
