"""Saved quadrature/angular matrix readback, no scientific kernel call."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',required=True,choices=['radial_quadrature','angular']);a=p.parse_args();out=Path(a.output);assert not out.exists()
inputs=[Path(a.a).resolve(),Path(a.b).resolve()];reports=[json.loads((d/'report.json').read_text()) for d in inputs]
assert all(sha(d/'operator.npz')==r['operator_sha256'] for d,r in zip(inputs,reports))
assert all(reports[0][k]==reports[1][k] for k in ('J','N','rb','coefficient_source_sha256','canceled_source_sha256','executable_sha256'))
proof=json.loads((Path(__file__).resolve().parent/'BLAS-equivalence-report.json').read_text());assert proof['passed'] and proof['finite_complete_arrays']
assert {reports[0]['source_sha256'],reports[1]['source_sha256']}=={'d28f652a641293f525c4f6a091dfd17edb97831bdc72d066f8368c78bfd99e0d','a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7'}
keys=['E','Kweak','Kstrong','Gvolume','Fboundary','SATload','Jbulk','Jsat','manufactured_load','manufactured_boundary_load']
with np.load(inputs[0]/'operator.npz') as z:x={k:z[k] for k in keys}
with np.load(inputs[1]/'operator.npz') as z:y={k:z[k] for k in keys}
rows={k:{'scaled':float(np.linalg.norm(x[k]-y[k])/max(1.,np.linalg.norm(x[k]),np.linalg.norm(y[k]))),'absolute':float(np.linalg.norm(x[k]-y[k])),'maxabs':float(np.max(np.abs(x[k]-y[k])))} for k in keys}
r={'kind':a.kind,'input_paths':[str(d) for d in inputs],'input_matrix_sha256':[r['operator_sha256'] for r in reports],'source_sha256':sha(__file__),'BLAS_equivalence_sha256':sha(Path(__file__).resolve().parent/'BLAS-equivalence-report.json'),'threshold':2e-8,'rows':rows,'passed':all(v['scaled']<=2e-8 for v in rows.values()),'scope':'Same finite trial/equations/SAT; integration readback only'}
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':r['passed'],'maximum_scaled':max(v['scaled'] for v in rows.values()),'output_sha256':sha(out)},indent=2));assert r['passed']
