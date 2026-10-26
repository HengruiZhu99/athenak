#!/usr/bin/env python3
"""Review synthetic frozen-coefficient spectra, constrained to trace-free jets.
Live inverse metrics define the algebraic tangent projection.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--evolved',action='store_true');a=p.parse_args()
lines=iter(a.input.read_text().splitlines());records=[]
C=np.zeros((2,22));C[0,[1,4,6]]=1;C[1,[11,14,16]]=1
_,_,v=np.linalg.svd(C,full_matrices=True);B=v[2:].T
for line in lines:
 if not line.startswith('case '):continue
 _,family,r,perturb,omega,lo,hi=line.split();matrices=[]
 metric_line=next(lines).split();assert metric_line[0]=='inverse'
 inverse=np.array(list(map(float,metric_line[1:]))).reshape(3,3)
 trace=np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
 C=np.zeros((2,22));C[0,1:7]=trace;C[1,11:17]=trace
 _,_,v=np.linalg.svd(C,full_matrices=True);B=v[2:].T
 normal_line=next(lines).split();assert normal_line[0]=="normal"
 normal=np.array(list(map(float,normal_line[1:])))
 assert next(lines)=="constraints"
 constraints=np.array([[float(x) for x in next(lines).split()] for _ in range(8)])
 for step in (1.,.5):
  assert next(lines)==f'matrix {step:g}'
  matrices.append(np.array([[float(x) for x in next(lines).split()] for _ in range(22)]))
 A=np.einsum("ia,ij,jb->ab",B,matrices[0],B); eigen,V=np.linalg.eig(-A)
 residual=np.linalg.norm(np.einsum("ij,jk->ik",-A,V)-V*eigen)/max(1,np.linalg.norm(A)*np.linalg.norm(V))
 record=dict(family='physical_trace_layer' if int(family) else 'reference_wave_map',radius=float(r),perturbation=('checkpoint_sample_'+perturb if a.evolved else ['reference','lapse_times_1.2','outward_shift_plus_0.02'][int(perturb)]),omega=float(omega),physical_speeds=[float(lo),float(hi)],speeds_real=sorted(eigen.real.tolist()),max_imaginary=float(max(abs(eigen.imag))),eigen_residual=float(residual),eigenvector_condition=float(np.linalg.cond(V)),extraction_step_relative_difference=float(np.linalg.norm(matrices[1]-matrices[0])/max(1,np.linalg.norm(matrices[0]))),algebraic_subspace_leak=float(np.linalg.norm(np.einsum("ai,ij,jb->ab",C,matrices[0],B))/max(1,np.linalg.norm(matrices[0]))))
 assert np.isfinite(A).all() and residual<1e-12
 assert record["extraction_step_relative_difference"]<1e-11
 assert record["algebraic_subspace_leak"]<1e-11
 D=np.einsum('ai,ib->ab',constraints,B)
 # Row normalization makes the kernel rank test insensitive to powers of Omega.
 D/=np.linalg.norm(D,axis=1)[:,None]
 _,singular,right=np.linalg.svd(D,full_matrices=True)
 rank=int(np.count_nonzero(singular>1e-9));kernel=right[rank:].T;quotient=right[:rank].T
 leak=np.linalg.norm(np.einsum('ai,ij,jb->ab',D,A,kernel))/max(1,np.linalg.norm(A))
 reduced=np.einsum('ia,ij,jb->ab',quotient,A,quotient)
 ce=np.linalg.eigvals(-reduced)
 record['constraint_rank']=rank;record['constraint_kernel_leak']=float(leak)
 record['constraint_quotient_speeds_real']=sorted(ce.real.tolist())
 record['constraint_quotient_max_imaginary']=float(max(abs(ce.imag)))
 record['constraint_kernel_dimension']=int(kernel.shape[1])
 assert rank==8 and leak<1e-10,(rank,leak)
 # Transverse trace-free metric/A amplitudes vanish for principal pure gauge.
 _,_,nv=np.linalg.svd(normal[None,:],full_matrices=True);E=nv[1:].T
 metric=np.linalg.inv(inverse)
 gram=np.einsum('ia,ij,jb->ab',E,metric,E)
 E=np.einsum('ia,ab->ib',E,np.linalg.inv(np.linalg.cholesky(gram).T))
 e,f=E[:,0],E[:,1];tensors=[np.outer(e,e)-np.outer(f,f),np.outer(e,f)+np.outer(f,e)]
 TT=np.zeros((4,22))
 for j,tensor in enumerate(tensors):
  weights=[tensor[0,0],2*tensor[0,1],2*tensor[0,2],tensor[1,1],2*tensor[1,2],tensor[2,2]]
  TT[j,1:7]=weights;TT[2+j,11:17]=weights
 P=np.einsum('ai,ib,bc->ac',TT,B,kernel)
 _,ss,rr=np.linalg.svd(P,full_matrices=True);assert np.count_nonzero(ss>1e-9)==4
 gauge=np.einsum('ij,jk->ik',kernel,rr[4:].T)
 # Gauge space must be invariant within the constraint-free space.
 GA=np.einsum('ia,ij,jb->ab',gauge,A,gauge)
 gauge_leak=np.linalg.norm(np.einsum('ij,jb->ib',A,gauge)-np.einsum('ia,ab->ib',gauge,GA))/max(1,np.linalg.norm(A))
 ge=np.linalg.eigvals(-GA)
 physical=np.einsum('ij,jk->ik',kernel,rr[:4].T)
 pe=np.linalg.eigvals(-np.einsum('ia,ij,jb->ab',physical,A,physical))
 record['gauge_kernel_leak']=float(gauge_leak)
 record['gauge_speeds_real']=sorted(ge.real.tolist());record['gauge_max_imaginary']=float(max(abs(ge.imag)))
 record['physical_quotient_speeds_real']=sorted(pe.real.tolist())
 record['physical_quotient_max_imaginary']=float(max(abs(pe.imag)))
 assert gauge_leak<1e-10,gauge_leak
 # Two real characteristic branches permit basis-independent projectors.
 speed_matrix=-A;low,high=float(lo),float(hi);eye=np.eye(20)
 minus=(high*eye-speed_matrix)/(high-low);plus=(speed_matrix-low*eye)/(high-low)
 defect=np.einsum('ij,jk->ik',speed_matrix-low*eye,speed_matrix-high*eye)
 scale=max(1,np.linalg.norm(speed_matrix)**2)
 symmetrizer=np.einsum('ji,jk->ik',minus,minus)+np.einsum('ji,jk->ik',plus,plus)
 symmetric_flux=np.einsum('ij,jk->ik',symmetrizer,speed_matrix)
 symmetry=np.linalg.norm(symmetric_flux-symmetric_flux.T)/max(1,np.linalg.norm(symmetric_flux))
 spectrum=np.linalg.eigvalsh(symmetrizer)
 record['two_branch_polynomial_relative_residual']=float(np.linalg.norm(defect)/scale)
 record['symmetrizer_min_eigenvalue']=float(spectrum.min());record['symmetrizer_condition']=float(spectrum.max()/spectrum.min())
 record['symmetrized_flux_relative_asymmetry']=float(symmetry)
 assert record['two_branch_polynomial_relative_residual']<1e-12
 assert spectrum.min()>0 and symmetry<1e-10
 records.append(record)
 print(record['family'],r,record['perturbation'],'light',lo,hi,'spectrum',min(eigen.real),max(eigen.real),'imag',record['max_imaginary'],'cond',record['eigenvector_condition'])
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(dict(status='diagnostic spectra; full mode classification pending',reduction='q=(chi,g6,alpha,beta3), p=(A6,P,Theta,Lambda3); radial derivative of q, p. Trace-free metric/A tangent subspace.',records=records),indent=2)+'\n')
