"""Add pinned principal-sector readback arrays without changing saved operator arrays."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
PR=ROOT/'build-layer-research/continuum/harmonic-principal-constraint-sectors/immutable-harmonic-normal-principal-sectors-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();inp=Path(a.input).resolve();out=Path(a.output).resolve();assert not out.exists();out.mkdir()
assert sha(PR/'index.json')=='05d4d7308477efe26d26ec7256fd9ba824040849363ed7f723031e5760adc0fc'
r=json.loads((inp/'report.json').read_text());assert sha(inp/'operator.npz')==r['operator_sha256']
with np.load(inp/'operator.npz') as z: arrays={k:z[k] for k in z.files}
assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in arrays.values())
sectors=json.loads((PR/'report.json').read_text())['sectors']['1'];rows=[]
for name in ('constraint','gauge','TT'):
 x=np.array([[float(Fraction(x)) for x in row] for row in sectors[name+'_left_rows']],dtype=np.float64);arrays[name+'_left']=x;rows.append(x)
arrays['incoming_left']=np.vstack(rows)
np.savez_compressed(out/'operator.npz',**arrays)
with np.load(out/'operator.npz') as z:
 for k,v in arrays.items():assert np.array_equal(v,z[k]),k
pins=[inp/'operator.npz',inp/'report.json',inp/('assemble_blas.py' if (inp/'assemble_blas.py').exists() else ('assemble_segmented.py' if 'Q_per_panel' in r else 'assemble.py')),inp/'energy_coefficients.py',inp/'canceled_basis_complex.py',inp/'source/receipt.json',inp/'input/receipt.json',PR/'index.json',PR/'report.json',P/'constraint-rate-api-authorization.json',Path(__file__)]
meta={'matrix_sha256':sha(out/'operator.npz'),'J':r['J'],'N':r['N'],'rb':r['rb'],'degree_order':'channel-major modal degree','nodal_map':'X_nodal=T X_modal','boundary_measure':'B includes rb*sqrt(w_angle)','k_in':r['kin'],'k_out':r['kout'],'expected_incoming_rank':r['expected_incoming_rank'],'rank_rtol':1e-10,'rank_atol':1e-11,'thresholds':{'mass_solve_SAT_forcing':2e-9,'weak_strong_volume':2e-8,'point_normal_adapter':5e-11,'modal_condition':1e12},'input_pins':[rec(x) for x in pins],'scope':'Saved algebraic readback decoration: all input arrays preserved bitwise; principal fixed-Ω sector maps only, not physical constraint boundary conditions.'}
(out/'matrix-metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
(out/'receipt.json').write_text(json.dumps({'input_matrix':rec(inp/'operator.npz'),'output_matrix':rec(out/'operator.npz'),'metadata':rec(out/'matrix-metadata.json'),'preserved_all_input_arrays_bitwise':True,'added_keys':['constraint_left','gauge_left','TT_left','incoming_left']},indent=2)+'\n')
print(json.dumps({'npz':rec(out/'operator.npz'),'metadata':rec(out/'matrix-metadata.json')},indent=2))
