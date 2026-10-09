from pathlib import Path
import json,hashlib,subprocess,sys,time,numpy as np
p=Path(__file__).resolve().parent;repo=p.parents[2];sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
input_path=repo/'build-layer-research/spatial-norm-native-controls/N36-t0.2/finite-angular-long-N36/layer.athinput'
text=input_path.read_text();scope='';mesh={}
for line in text.splitlines():
 line=line.split('#')[0].strip()
 if line.startswith('<'):scope=line.strip('<>')
 elif scope=='mesh' and '=' in line:
  k,v=map(str.strip,line.split('=',1));mesh[k]=v
span=float(mesh['x1max'])-float(mesh['x1min']);assert span==2.1
files=[repo/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=repo,text=True).splitlines()]+[p/f for f in ['damping_profile.hpp','profile_helpers.hpp','native_nyquist.cpp','check_native_nyquist.py']]+[p.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',input_path]
before={str(f.relative_to(repo)):sha(f) for f in files};results=[]
flags=json.loads((p/'receipt.json').read_text())['commands'][0]['command'][:-3]
commands=[flags+[str(p/'native_nyquist.cpp'),'-o',str(p/'native_nyquist')],[str(p/'native_nyquist')]]
for i,cmd in enumerate(commands):
 r=subprocess.run(cmd,cwd=repo,text=True,capture_output=True);results.append({'command':cmd,'returncode':r.returncode,'stderr':r.stderr});r.check_returncode();assert not r.stderr
 if i==1:(p/'native-Nyquist.json').write_text(r.stdout);results[-1]['stdout_sha256']=sha(p/'native-Nyquist.json')
 else:results[-1]['stdout']=r.stdout
rows=json.loads((p/'native-Nyquist.json').read_text());assert len(rows)==12;cases=[]
for x in rows:
 N=x['N'];h=span/N;axis=-1.05+(np.arange(N)+.5)*h;rr=axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2;omega=1-rr[rr<1].max();assert abs(omega-x['Omega'])<2e-14
 assert abs(x['k']-np.pi/h)<1e-13 and abs(x['dt']-.03*x['Omega'])<1e-14
 v=np.asarray(x['L']);A=v[:,:,0]+1j*v[:,:,1];weights=np.array([1.]*12+[1/x['k']]*8);A=np.einsum('i,ij,j->ij',weights,A,1/weights)
 ev=np.linalg.eigvals(A);neg=ev[ev.real<=0];z=x['dt']*neg;excess=float(max(0.,max(abs(1+z+z*z/2+z*z*z/6))-1));assert excess<1e-12
 cases.append({k:x[k] for k in ['N','span','k','Omega','dt','profile','oblique']}|{'max_nonpositive_root_RK3_excess':excess,'max_real_primitive':float(ev.real.max()),'nonpositive_root_count':len(neg)})
after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
out={'passed_actual_native_span_Nyquist_supplement':True,'global_native_or_scri_stability_accepted':False,'scope':'Supplement corrects v2 span2.2 native-Nyquist label. Those previous frequencies are valid samples; native mesh span is2.1. Actual unchanged helper/source; no native evolution or recompile required.','authoritative_input':str(input_path.relative_to(repo)),'authoritative_input_sha256':sha(input_path),'source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,'cases':cases,'binary_sha256':sha(p/'native_nyquist'),'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
(p/'native-Nyquist-receipt.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('PASS actual native span2.1 Nyquist supplement,12 cases,excess0',sha(p/'native-Nyquist-receipt.json'))
