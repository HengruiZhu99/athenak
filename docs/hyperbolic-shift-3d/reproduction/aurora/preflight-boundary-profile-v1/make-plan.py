from pathlib import Path
import json,sys
root=Path.cwd();stage=root/'output/hyperbolic-core-shell/aurora-staging'
sys.path.insert(0,str(root/'tst/hyperbolic_shift_3d/aurora'));import control
old='preflight-sparse-stage-v1';name='preflight-boundary-profile-v1'
p=json.loads((root/'docs/hyperbolic-shift-3d/reproduction/aurora'/old/'plan.json').read_text().replace(old,name).replace('v21','v23'))
for c in p['commands']:
 if c['name']=='puncture-control':
  c['executable']='build-sycl-mpi-v21/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_sparse'
  c['inputs']=[x.replace('source-v20','source-v21').replace('build-logs/v20','build-logs/v21') for x in c['inputs']]
  for check in c['checks']:
   if check['path']==['boundary_execution']:check['value']='cached_global_projection_angular'
   if check['path']==['boundary_cache_hits']:check.update(op='ge',value=1)
  c['checks'].append(dict(path=['volume_exchange'],op='eq',value='cached_sparse_alltoallv'))
p['notes']=['Fine boundary and all-error-synchronization profiling only; fixed mesh and unchanged V11 equations and equal-cell ownership.', 'Same-node n16 puncture pair: frozen V21 sparse control versus instrumented V23 sparse. Same timestep, input, diagnostics and scientific gates; whole checkpoint equality required.', 'Lifecycle, device parity, Minkowski and pulse restart precede profiling. Boundary subphases subdivide closure time; all-error-sync overlaps other buckets and must not be added to exclusive phases.', 'No asynchronous overlap or AMR evolution claim. Use results to distinguish actual boundary work from waiting, and assess whether communication overlap is worthwhile.']
control.validate_plan(root,p)
(stage/(name+'-plan.json')).write_text(json.dumps(p,indent=2)+'\n')
print(len(p['commands']),sum(len(c['checks']) for c in p['commands']))
