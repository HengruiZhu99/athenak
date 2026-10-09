"""Boundary-only support, cubic symmetry, matching-time temporal and refinement checks."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.io import mmread
w=Path(__file__).resolve().parent;results=[]
for N,span in [(16,2.2),(20,2.2),(24,2.2),(24,2.1)]:
 stems=[w/f'N{N}-span{span}-{c}-ko0.1' for c in ['ray','mls']];As=[mmread(str(p)+'.mtx').tocsr() for p in stems];points=[np.genfromtxt(str(p)+'-points.csv',delimiter=',',names=True) for p in stems];q=points[0];m=len(q);assert np.array_equal(points[0],points[1]);boundary=q['boundary_row'].astype(bool);D=As[1]-As[0];interior=np.r_[~boundary,~boundary];outside=D[interior];err=float(abs(outside.data).max()) if outside.nnz else 0.;assert err==0
 changed=np.diff(D.indptr)>0;result={'n':N,'span':span,'unknowns':2*m,'boundary_point_rows':int(boundary.sum()),'changed_coupled_rows':int(changed.sum()),'interior_coupled_rows':int(interior.sum()),'interior_row_difference_exact_zero':err==0,'symmetry':[]};nx=N+6;ids=q['index'].astype(int);lookup={s:i for i,s in enumerate(ids)}
 for name,transform in [('x_reflection',lambda i,j,k:(nx-1-i,j,k)),('xy_swap',lambda i,j,k:(j,i,k))]:
  perm=[]
  for s in ids:
   i,j,k=transform(s%nx,s//nx%nx,s//(nx*nx));perm.append(lookup[i+nx*(j+nx*k)])
  p=np.r_[perm,np.array(perm)+m]
  for c,A in zip(['ray','mls'],As):
   difference=A[p][:,p]-A;absmax=float(abs(difference.data).max()) if difference.nnz else 0.;relative=absmax/abs(A.data).max();assert relative<2e-12;result['symmetry'].append({'closure':c,'transform':name,'max_entry_commutator':absmax,'relative_to_max_matrix_entry':relative})
 results.append(result)
temporal=[]
for c in ['ray','mls']:
 a=np.load(w/f'N24-span2.1-{c}-ko0.1-fixed-dt0.1.npz');b=np.load(w/f'N24-span2.1-{c}-ko0.1-fixed-dt0.05.npz');assert np.array_equal(a['times'],b['times']);temporal.append({'closure':c,'max_state_difference_at_identical_output_times':float(abs(a['states']-b['states']).max()),'per_time':[{'time':float(t),'linf_difference':float(abs(x-y).max())} for t,x,y in zip(a['times'],a['states'],b['states'])]})
refinement=[]
for c in ['ray','mls']:
 for N in [16,20,24]:
  p=w/f'N{N}-span2.2-{c}-ko0.1-fixed-dt0.1.json';r=json.loads(p.read_text());refinement.append({'closure':c,'n':N,'spacing':r['native_export']['spacing'],'error_rms_at_exact_times':{str(x['time']):x['error_rms'] for x in r['history']},'error_linf_at_exact_times':{str(x['time']):x['error_linf'] for x in r['history']}})
r={'support_and_symmetry':results,'timestep_halving':temporal,'same_span_refinement':refinement,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(w/'closure-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
