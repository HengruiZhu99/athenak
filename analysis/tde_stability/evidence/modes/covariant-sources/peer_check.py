"""Read-only checks of saved source-validation artifacts; no new evolution."""
from pathlib import Path
import json,hashlib,re,numpy as np
root=Path(__file__).parent
results={}
# Validate full/background equality at every saved zero-control RK RHS input.
rows=[]
for p in sorted((root/'validation/zero').glob('z4c_snapshot_rhs_full_vs_bg*.json')):
 meta=json.loads(p.read_text());full=p.with_suffix('.bin');bg=p.with_suffix('.background.bin')
 a=np.fromfile(full,dtype='<f8');b=np.fromfile(bg,dtype='<f8')
 assert a.size==np.prod(meta['shape'])==b.size
 rows.append({'cycle':meta['cycle'],'stage':meta['stage'],'full_background_bytes_equal':full.read_bytes()==bg.read_bytes(),'finite':bool(np.isfinite(a).all() and np.isfinite(b).all())})
results['zero_full_inputs']=rows
# Independently check actual serialized arrays, including all ghost cells.
all_equal=[]
for f in sorted((root/'validation/off_new').glob('*.bin')):
 q=root/'validation/off_old'/f.name;all_equal.append(f.read_bytes()==q.read_bytes())
results['default_off_serialized_arrays']={'count':len(all_equal),'all_equal':all(all_equal)}
a=root/'validation/matter_on_low/z4c_snapshot_volume_rhs_rank0_cycle0_stage1.bin';b=root/'validation/matter_off_low/z4c_snapshot_volume_rhs_rank0_cycle0_stage1.bin'
results['matter_initial_rhs_bytes_equal']=a.read_bytes()==b.read_bytes()
# RHS-term logs expose a scalar signed sum for Khat and Theta at saved locations.
# Their decimal precision is only ~6 digits, so compare at printed precision.
for case in ('zero','late_on','late_off'):
 errs={'Khat':[],'Theta':[]}
 for line in (root/'validation'/case/'run.log').read_text().splitlines():
  if not line.startswith('Z4C_RHS_TERM_LOC '):continue
  kv=dict(re.findall(r'(\w+)=([^\s]+)',line))
  for name,terms in [('Khat',('Khat_dda','Khat_alg','Khat_adv','Khat_damp','Khat_mat','Khat_covariant')),('Theta',('Theta_adv','Theta_Ht','Theta_damp','Theta_mat','Theta_covariant'))]:
   expected=sum(float(kv[x]) for x in terms);actual=float(kv['rhs_'+name]);scale=sum(abs(float(kv[x])) for x in terms)
   errs[name].append((abs(expected-actual),abs(expected-actual)/max(scale,1e-300)))
 results[case+'_forensic_signed_sum']={name:{'records':len(v),'max_absolute_difference':max(a for a,b in v),'max_relative_to_term_magnitudes':max(b for a,b in v)} for name,v in errs.items()}
results['forensic_sum_scope']='Local diagnostic scalar sums omit the separate KO contribution while rhs values include it. Nonzero late full-sum differences are not source defects; compare matched on/off increments below. Zero residual tiny differences reflect existing independently recomputed algebraic diagnostics.'
logs={}
for case in ('late_on','late_off'):
 logs[case]={}
 for line in (root/'validation'/case/'run.log').read_text().splitlines():
  if line.startswith('Z4C_RHS_TERM_LOC '):
   kv=dict(re.findall(r'(\w+)=([^\s]+)',line));logs[case][kv['field']]=kv
increments=[]
for field,a in logs['late_on'].items():
 b=logs['late_off'][field]
 if any(a[k]!=b[k] for k in ('x','y','z','gid','cycle','stage')):continue
 for name in ('Khat','Theta'):
  actual=float(a['rhs_'+name])-float(b['rhs_'+name]);expected=float(a[name+'_covariant'])-float(b[name+'_covariant'])
  increments.append({'field':field,'component':name,'actual_increment':actual,'printed_source_increment':expected,'absolute_error':abs(actual-expected)})
results['forensic_covariant_on_off_increments']=increments
results['binary_sha256']=hashlib.sha256((root/'athena-covariant-sources').read_bytes()).hexdigest()
assert all(q['full_background_bytes_equal'] and q['finite'] for q in rows)
assert results['default_off_serialized_arrays']['all_equal'] and results['matter_initial_rhs_bytes_equal']
(root/'peer-check-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
