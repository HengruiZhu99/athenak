"""Authorized warning-free replay of two previously screened reduced pairs.

Only the fixed rank32 and independent rank16 dense Ritz export is replayed.
No global eigensolve, new search, propagation or matrix generation.
"""
from pathlib import Path
import hashlib,json,subprocess,time,warnings
import numpy as np
from scipy.linalg import svd
from scipy.sparse import load_npz
W=Path(__file__).resolve().parent;R=W.parents[2]
B=R/'build-layer-research/boundary/full-tensor-C0-N20-20261009';O=B/'full22'
M=R/'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={B/'immutable-C0-N20-default-span-v2-20261009/index.json':'8ec4c4dd84898d19b055831696de6e3608542b1fd10a88e04f087b1f689f99aa',
 M/'index.json':'396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69',
 M/'check_and_freeze.py':'6e839bf9c0d91338d47b03110c1e5dc1f65c97a9954baf00d43971207d4829b2',
 O/'spatialnorm-projected-J20.npz':'bfc114a9495b49d01b9275f4efd8a1fe49d167273f8d849631d25cddf9c2a103',
 O/'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz':'2e777543e187896882e7a97f4f23a63fee057145a8c4f034932a782ebe0154c7',
 O/'spatialnorm-cache0.0001-metadata.json':'cd72ca5486a65e90744ed04f00a4da58f7246e2f63e5743147fba19209babb86'}
for p,h in pins.items():assert sha(p)==h,p
for index in [M/'index.json',B/'immutable-C0-N20-default-span-v2-20261009/index.json']:
 for name,row in json.loads(index.read_text())['files'].items():assert sha(index.parent/name)==row['sha256'],name
warnings.simplefilter('error',RuntimeWarning)
started=time.monotonic();J=load_npz(O/'spatialnorm-projected-J20.npz');states=np.load(O/'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz')
vectors={};rows=[]
for key,name,rank in [('candidate0','n20-late2-gauge-s2',32),('candidate1','n20-late4-gauge-s1',16)]:
 report=json.loads((M/(name+'.json')).read_text());pins[M/(name+'.json')]=sha(M/(name+'.json'))
 lo,hi=report['window'];ids=np.flatnonzero((states['times']>=lo-1e-12)&(states['times']<=hi+1e-12))[::report['stride']]
 X=states['values'][ids[:-1],:,0].T;X=X/np.sqrt(np.sum(X*X,axis=0))
 u,s,vh=svd(X,full_matrices=False,lapack_driver='gesdd');U=u[:,:rank];JU=J@U
 small=np.einsum('ki,kj->ij',U,JU,optimize=False);eig,eigvec=np.linalg.eig(small)
 k=int(np.argmin(abs(eig-(1.9182803+6.8055088j))))
 v=np.einsum('ij,j->i',U,eigvec[:,k],optimize=False);v/=np.sqrt(np.sum(abs(v)**2));v*=np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
 residual=float(np.sqrt(np.sum(abs(J@v-eig[k]*v)**2)))
 old=next(q for q in report['results'] if q['rank']==rank and q['method']=='rayleigh-ritz')
 oldk=int(np.argmin(abs(np.array([complex(*z) for z in old['eigenvalues']])-eig[k])))
 assert abs(residual-old['actual_J_residual_generator_units'][oldk])<1e-9
 assert abs(eig[k]-complex(*old['eigenvalues'][oldk]))<1e-8
 assert residual<1e-6 and s[rank-1]/s[0]>1e-10
 assert np.isfinite(v).all()
 vectors[key]=v;rows.append({'key':key,'case':name,'rank':rank,'method':'rayleigh-ritz','window':report['window'],'stride':report['stride'],
  'lambda':[float(eig[k].real),float(eig[k].imag)],'actual_J_residual_generator_units':residual,
  'singular_value_fraction_at_rank':float(s[rank-1]/s[0]),'difference_from_prior_lambda':float(abs(eig[k]-complex(*old['eigenvalues'][oldk]))),
  'difference_from_prior_residual':abs(residual-old['actual_J_residual_generator_units'][oldk])})
agreement=float(np.sqrt(np.sum(abs(vectors['candidate1']-vectors['candidate0'])**2)))
assert agreement<1e-5
np.savez_compressed(W/'candidate-vectors.npz',**vectors)
out={'status':'WARNING_FREE_REPLAY_OF_PREVIOUSLY_SCREENED_FIXED_REDUCED_EXPORTS',
 'scope':'Approximate/pseudospectral vectors only; singular fraction1e-10 is a screening heuristic, not eigenvalue certification. No global eigensolve/search/evolution/newmatrix.',
 'coordinates':'Complex unit Euclidean free20 point-major active k/j/i; largest entry phased real positive; no field rescaling.',
 'candidates':rows,'phase_aligned_candidate_distance':agreement,'seconds':time.monotonic()-started,
 'candidate_vectors_sha256':sha(W/'candidate-vectors.npz'),'source_sha256':sha(__file__),
 'input_sha256':{str(p):h for p,h in pins.items()},'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
 'runtime_warnings_promoted_to_errors':True}
(W/'candidate-metadata.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('EXPORTED',rows,agreement,out['seconds'],flush=True)
