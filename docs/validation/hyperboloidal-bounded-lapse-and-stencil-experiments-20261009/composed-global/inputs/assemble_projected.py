"""Check candidate full22 consistency, then project the unchanged raw CSR exactly."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.sparse import block_diag,load_npz,save_npz
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);a=p.parse_args();g=a.gauge
read=lambda q:json.loads(q.read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
v=read(w/f'{g}-cache0.0001-validation.json');N=v['metadata']['points'];audits=[json.loads(x) for x in (w/f'{g}-cache0.0001-validation.stderr').read_text().splitlines() if x.startswith('{')];ref=audits[0]
best=max(min(x['relative_l2'] for x in r['native_eps_sweep']) for r in v['raw_Jv'])
step=max(min(x['relative_input_l2_vs_J22_step'] for x in t['native_step_amplitude_sweep']) for r in v['one_step'] for t in r['dt_comparison'] if t['native_step_amplitude_sweep'])
sparse=max(x['sparse_vs_cached_relative_l2'] for x in v['raw_Jv'])
checks=[{'name':'reference/Prepare/strict donors','pass':ref['prepared_reference_change']==0 and ref['projected_reference_rhs_max']<1e-10 and max(ref['projected_reference_H'],ref['projected_reference_M'],ref['projected_reference_Z'])<1e-11 and ref['ghost_weight_sum_error']<1e-13 and ref['strict_interior_donor_references']==88992,'observed':ref},
{'name':'raw CSR equals actual cached stencil matvec','pass':sparse<2e-14,'observed':sparse,'threshold':2e-14},
{'name':'raw native22 RHS amplitude sweep best pervector','pass':best<2e-8,'observed':best,'threshold':2e-8},
{'name':'native final-only one-step amplitude sweep best pervector anddt','pass':step<1e-8,'observed':step,'threshold':1e-8},
{'name':'algebraic PL identity','pass':v['lift_project_identity_max']<1e-13,'observed':v['lift_project_identity_max'],'threshold':1e-13},
{'name':'native server exit','pass':v['server_exit_status']==0,'observed':v['server_exit_status']}]
receipt={'variant':'original C0 spatialnorm N20 finiteΩ strict interior','gauge':g,'validation_sha256':sha(w/f'{g}-cache0.0001-validation.json'),'checks':checks,'all_consistency_checks_pass':all(x['pass'] for x in checks),'threshold_semantics':'implementation consistency thresholds matching previous baseline stage gate; not physical acceptance'}
(w/f'{g}-pre-pilot-gate.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
assert receipt['all_consistency_checks_pass'],'do not propagate candidate with failed consistency gate'
A=load_npz(w/f'{g}-cache0.0001-J22.npz');L=np.fromfile(w/f'{g}-cache0.0001-lift.bin',dtype='<f8').reshape(N,22,20);P=np.fromfile(w/f'{g}-cache0.0001-restrict.bin',dtype='<f8').reshape(N,20,22);Ls=block_diag(L,format='csr');Ps=block_diag(P,format='csr');Ls.eliminate_zeros();Ps.eliminate_zeros();J=Ps@A@Ls;J.eliminate_zeros();J.sort_indices();save_npz(w/f'{g}-projected-J20.npz',J)
receipt['matrix']={'shape':list(J.shape),'nnz':int(J.nnz),'sha256':sha(w/f'{g}-projected-J20.npz'),'trace':float(J.diagonal().sum()),'semantics':'P_ref J22 Lift, continuous projected generator, no coefficient dropping'}
(w/f'{g}-pre-pilot-gate.json').write_text(json.dumps(receipt,indent=2)+'\n');print('ASSEMBLED',g,J.shape,J.nnz,flush=True)
