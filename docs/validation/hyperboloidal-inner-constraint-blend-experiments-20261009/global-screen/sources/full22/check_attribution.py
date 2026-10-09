"""Actual sparse-row support and initial-gauge equality, no spectral inference."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;old=w.parents[1]/'full-tensor-propagator/full22-v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();res={}
for g in ['production','spatialnorm']:
 m=json.loads((w/f'{g}-cache0.0001-metadata.json').read_text());xyz=np.asarray(m['xyz_omega_volume_ginv_chi'])[:,:3];r=np.linalg.norm(xyz,axis=1);N=len(r);A=load_npz(w/f'{g}-projected-J20.npz');B=load_npz(old/f'{g}-projected-J20.npz');D=(A-B).tocsr();D.eliminate_zeros();out=np.repeat(r>=.85,20);unchanged=np.tile(np.isin(np.arange(20),[0,1,2,3,4,5,16,17,18,19]),N)
 def maxabs(q):return float(abs(q.data).max()) if q.nnz else 0.
 v=np.load(w.parent/f'{g}-validation-vectors.npz')['gauge_pulse'];delta=D@v;baseline=B@v
 row={'candidate_J20_sha256':sha(w/f'{g}-projected-J20.npz'),'frozen_C0_J20_sha256':sha(old/f'{g}-projected-J20.npz'),'outer_nodes_r_ge_085':int(np.sum(r>=.85)),'outer_rows_maxabs_difference':maxabs(D[out]),'chi_metric_alpha_beta_rows_maxabs_difference':maxabs(D[unchanged]),'all_rows_maxabs_difference':maxabs(D),'initial_gauge_J_difference_relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(baseline)),'initial_gauge_J_difference_linf':float(abs(delta).max()),'interpretation':'support of actual semidiscrete difference; no global energy/stability/closure conclusion'}
 row['checks_pass']=row['outer_rows_maxabs_difference']<1e-8 and row['chi_metric_alpha_beta_rows_maxabs_difference']<1e-8 and row['initial_gauge_J_difference_relative_l2']<1e-10
 assert row['checks_pass'];res[g]=row
(w/'actual-matrix-support-attribution.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
