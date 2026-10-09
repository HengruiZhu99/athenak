"""Complete finite-array equivalence for contraction-only performance change."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001';b=P/'J0-N8-rb.98-segmentedQ64-a12x24-BLAS-equivalence001'
plan=json.loads((P/'BLAS-equivalence-plan.json').read_text());reports=[json.loads((d/'report.json').read_text()) for d in (a,b)]
assert reports[0]['source_sha256']==plan['old_source_sha256'];assert reports[1]['source_sha256']==plan['new_source_sha256']
assert all(sha(d/'operator.npz')==r['operator_sha256'] for d,r in zip((a,b),reports));assert all(r['passed_single_quadrature_algebra'] for r in reports)
with np.load(a/'operator.npz') as z:x={k:z[k] for k in z.files}
with np.load(b/'operator.npz') as z:y={k:z[k] for k in z.files}
assert x.keys()==y.keys();rows={}
for k in x:
 assert x[k].dtype==y[k].dtype==np.float64 and np.isfinite(x[k]).all() and np.isfinite(y[k]).all(),k
 assert x[k].shape==y[k].shape,k
 delta=x[k]-y[k];rows[k]={'scaled':float(np.linalg.norm(delta)/max(1.,np.linalg.norm(x[k]),np.linalg.norm(y[k]))),'absolute':float(np.linalg.norm(delta)),'maxabs':float(np.max(np.abs(delta)))}
report={'passed':all(v['scaled']<=2e-9 for v in rows.values()),'finite_complete_arrays':True,'input_matrix_sha256':[r['operator_sha256'] for r in reports],'exact_shapes_dtypes_keys':True,'rank_equal':reports[0]['observed_incoming_rank']==reports[1]['observed_incoming_rank']==4,'threshold':2e-9,'rows':rows,'source_sha256':sha(__file__),'scope':'Contraction-only BLAS numerical equivalence, no physics change or generator spectrum/evolution'}
(P/'BLAS-equivalence-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));print('maximum_scaled',max(v['scaled'] for v in rows.values()));assert report['passed'] and report['rank_equal']
