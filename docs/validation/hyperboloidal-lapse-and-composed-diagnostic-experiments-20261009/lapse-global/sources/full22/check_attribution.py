"""Actual matrix/reference identity for the predicted local alpha-only change."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import coo_matrix,load_npz
w=Path(__file__).resolve().parent;old=w.parents[1]/'full-tensor-propagator/full22-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();maximum=lambda A:float(abs(A.data).max()) if A.nnz else 0.
g='spatialnorm';meta=json.loads((w/f'{g}-cache0.0001-metadata.json').read_text());n=meta['points']
q=np.asarray(json.loads((w/'reference-coefficients.json').read_text()));assert q.shape==(n,13)
assert np.array_equal(q[:,:4],np.asarray(meta['xyz_omega_volume_ginv_chi'])[:,:4])
alpha,beta,da,c=q[:,5],q[:,6:9],q[:,9:12],q[:,12]
expected=np.column_stack([-c*np.sum(beta*da,axis=1)/alpha,-c[:,None]*da])
row=np.repeat(20*np.arange(n)+16,4);col=np.ravel(20*np.arange(n)[:,None]+np.array([16,17,18,19]))
f=w/f'{g}-projected-J20.npz';f0=old/f'{g}-projected-J20.npz';J,J0=load_npz(f),load_npz(f0)
E=coo_matrix((expected.ravel(),(row,col)),shape=J.shape).tocsr();E.eliminate_zeros();D=J-J0;err=D-E
outside=D.copy().tolil();outside[row,col]=0.;outside=outside.tocsr();outside.eliminate_zeros()
outer=np.flatnonzero(q[:,4]>=.85);outerrows=(20*outer[:,None]+np.arange(20)).ravel()
nonalpha=(20*np.arange(n)[:,None]+np.array([f for f in range(20) if f!=16])).ravel()
seed=np.load(w.parent/'spatialnorm-validation-vectors.npz')['gauge_pulse'];act=D@seed
r={'formula':'Delta alpha_t=−c*(delta beta·grad alpha_ref+(beta_ref·grad alpha_ref)*delta alpha/alpha_ref)',
 'semantics':'local value-only alpha row; no derivative/geometry/pole/gauge-shift change; no stability theorem',
 'candidate_matrix_sha256':sha(f),'C0_matrix_sha256':sha(f0),'reference_coefficients_sha256':sha(w/'reference-coefficients.json'),
 'expected_nnz':int(E.nnz),'max_expected_entry':float(abs(expected).max()),'max_changed_entry':maximum(D),
 'max_absolute_error_vs_expected':maximum(err),'max_absolute_unexpected_entry':maximum(outside),
 'outer_nodes':len(outer),'outer_all20_rows_max_change':maximum(D[outerrows]),'all_nonalpha_rows_max_change':maximum(D[nonalpha]),
 'gauge_seed_action_change_l2':float(np.linalg.norm(act)),'gauge_seed_action_change_linf':float(abs(act).max()),
 'changed_gauge_action_nonalpha_linf':float(abs(act.reshape(n,20)[:,[f for f in range(20) if f!=16]]).max()),
 'core_nodes_r_le_005':int(np.count_nonzero(q[:,4]<=.05)),'core_scope':'this N16 cell-centred grid has no core<=.05 nodes; exact-core source identity is provided by the independent frozen local gate'}
f22=w/f'{g}-cache0.0001-J22.npz';f022=old/f'{g}-cache0.0001-J22.npz'
A,A0=load_npz(f22),load_npz(f022)
row22=np.repeat(22*np.arange(n)+18,4);col22=np.ravel(22*np.arange(n)[:,None]+np.array([18,19,20,21]))
E22=coo_matrix((expected.ravel(),(row22,col22)),shape=A.shape).tocsr();E22.eliminate_zeros();D22=A-A0
nonalpha22=(22*np.arange(n)[:,None]+np.array([f for f in range(22) if f!=18])).ravel()
r['full22']={'matrix_sha256':sha(f22),'C0_matrix_sha256':sha(f022),'max_absolute_error_vs_expected':maximum(D22-E22),'all_nonalpha_rows_max_change':maximum(D22[nonalpha22]),'outer_all22_rows_max_change':maximum(D22[(22*outer[:,None]+np.arange(22)).ravel()])}
r['passed_attribution_1e-8']=r['max_absolute_error_vs_expected']<1e-8 and r['max_absolute_unexpected_entry']==r['outer_all20_rows_max_change']==r['all_nonalpha_rows_max_change']==0 and r['full22']['max_absolute_error_vs_expected']<1e-8 and r['full22']['all_nonalpha_rows_max_change']==r['full22']['outer_all22_rows_max_change']==0
assert r['passed_attribution_1e-8'],r
(w/'actual-matrix-lapse-attribution.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
