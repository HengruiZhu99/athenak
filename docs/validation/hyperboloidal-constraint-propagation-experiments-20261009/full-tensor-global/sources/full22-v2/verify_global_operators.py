"""Native RHS and native nonlinear norm diagnostics on propagated tangent directions."""
from pathlib import Path
import json,struct,subprocess,time,hashlib
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();res={'semantics':'All directions are differentiated at the stationary reference, not used as finite nonlinear states. Native norms from ProjectAlgebraic+Prepare+Diagnose are averaged for plus/minus states after division byepsilon.','operator_checks':{},'diagnostic_checks':{}}
for g in ['production','spatialnorm']:
 data=np.load(w/f'{g}-projected-expm-t2.0.npz');J=load_npz(w/f'{g}-projected-J20.npz');stderr=(w/f'{g}-global-operator-check.stderr').open('w');proc=subprocess.Popen([str(w.parent/f'server-{g}'),'16','2.2','0.0001'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr);meta=json.loads(proc.stdout.readline());assert meta['points']==1640;dim=meta['dimension'];rows=[]
 for name,it,col in [('gauge_t1',40,0),('gauge_t2',80,0),('shell_t2',80,1)]:
  v=data['values'][it,:,col];expected=J@v;row={'direction':name,'amplitude_sweep':[]}
  for eps in [1e-3,1e-4,1e-5,3e-6]:
   proc.stdin.write(b'f'+struct.pack('d',eps)+v.tobytes());proc.stdin.flush();b=bytearray()
   while len(b)<8*dim:
    q=proc.stdout.read(8*dim-len(b))
    if not q:raise RuntimeError('native RHS check failed')
    b.extend(q)
   actual=np.frombuffer(b,dtype=np.float64);row['amplitude_sweep'].append({'eps':eps,'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),'linf':float(abs(actual-expected).max())})
  rows.append(row)
 proc.stdin.close();proc.wait();stderr.close();res['operator_checks'][g]={'actual_native_RHS_executable_sha256':sha(w.parent/f'server-{g}'),'metadata_grid_matches':meta['n']==16 and meta['span']==2.2,'rows':rows,'exit_status':proc.returncode};print(g,'RHS',rows,flush=True)
err=(w/'direct-native-norm-check.stderr').open('w');proc=subprocess.Popen([str(w/'diagnostic-constraint-norms')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);meta=json.loads(proc.stdout.readline());res['native_diagnostic_reference']=meta;res['native_diagnostic_executable_sha256']=sha(w/'diagnostic-constraint-norms');res['native_diagnostic_source_sha256']=sha(w/'diagnostic_constraint_norms.cpp')
for g in ['production','spatialnorm']:
 data=np.load(w/f'{g}-projected-expm-t2.0.npz');diag=json.loads((w/f'{g}-projected-expm-analysis.json').read_text());rows=[]
 for name,it,col in [('gauge_t0',0,0),('gauge_t1',40,0),('gauge_t2',80,0),('shell_t2',80,1)]:
  v=data['values'][it,:,col];expected=np.array(diag['histories'][col]['history'][it]['native_H_M_Z_rms']);row={'direction':name,'signed_linear_constraint_RMS':expected.tolist(),'amplitude_sweep':[]}
  for eps in [1e-3,1e-4,1e-5]:
   proc.stdin.write(struct.pack('d',eps)+v.tobytes());proc.stdin.flush();b=bytearray()
   while len(b)<48:
    q=proc.stdout.read(48-len(b))
    if not q:raise RuntimeError('native norm check failed')
    b.extend(q)
   both=np.frombuffer(b,dtype=np.float64).reshape(2,3);actual=both.mean(axis=0);row['amplitude_sweep'].append({'eps':eps,'plus_minus_averaged_native_RMS_div_epsilon':actual.tolist(),'relative_error_H_M_Z':(abs(actual-expected)/np.maximum(abs(expected),1e-300)).tolist() if np.linalg.norm(expected)>0 else [0,0,0],'absolute_error_H_M_Z':abs(actual-expected).tolist()})
  rows.append(row)
 res['diagnostic_checks'][g]=rows;print(g,'native diagnose',rows,flush=True)
proc.stdin.close();proc.wait();err.close();res['direct_norm_server_exit']=proc.returncode
# Identity of new per-field native derivatives with existing total native component H1 callback.
res['field_derivative_identity']={}
for g in ['production','spatialnorm']:
 a=json.loads((w/f'{g}-projected-expm-analysis.json').read_text());f=json.loads((w/f'{g}-projected-krylov-field-analysis.json').read_text());errs=[]
 for ha,hf in zip(a['histories'],f['histories']):
  for x,y in zip(ha['history'],hf['history']):
   total=np.sqrt(sum(z['value_l2']**2+z['gradient_l2']**2 for z in y['groups'].values()));errs.append(abs(total-x['reference_volume_component_H1'])/x['reference_volume_component_H1'])
 res['field_derivative_identity'][g]={'maxrelative_total_native_H1_vs_sum_field_value_gradient_norms':max(errs)}
(w/'global-native-operator-diagnostic-verification.json').write_text(json.dumps(res,indent=2)+'\n');print('done field identity',res['field_derivative_identity'],flush=True)
