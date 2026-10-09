"""Saved source-normal and exact-polynomial mass readbacks; no generator spectrum."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.special import roots_jacobi
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from assemble_degree import modal,common_quadrature,layout,mm
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(a,b):d=a-b;return {'scaled':float(np.linalg.norm(d)/max(1.,np.linalg.norm(a),np.linalg.norm(b))),'absolute':float(np.linalg.norm(d)),'maxabs':float(np.max(np.abs(d)))}
p=argparse.ArgumentParser();p.add_argument('--matrix',required=True);a=p.parse_args();d=Path(a.matrix).resolve();dest=d/'auxiliary-readback';assert not dest.exists();dest.mkdir()
r=json.loads((d/'report.json').read_text());assert sha(d/'operator.npz')==r['operator_sha256'];N=r['N'];rb=r['rb'];B=rb*rb;nodes,_=common_quadrature(N,rb)
exactpath=P/f'polynomial-mass-N{N}/exact.json';exact=json.loads(exactpath.read_text());assert exact['passed_exact'] and exact['N']==N;mass=[]
for L in range(5):
 x,w=roots_jacobi(2*N+4,0,L+.5);rho=(x+1)*B/2;weights=w*(B/2)**(L+1.5)/2
 M,_,_=modal(rho,N,L,B);T,_,_=modal(nodes,N,L,B);iv=np.linalg.inv(T);modal_mass=np.einsum('qi,q,qj->ij',M,weights,M,optimize=False)
 nodal=np.einsum('qi,ij->qj',M,iv,optimize=False);Mn=np.einsum('qi,q,qj->ij',nodal,weights,nodal,optimize=False)
 target=mm(iv.T,iv);mass.append({'L':L,'modal_I':error(modal_mass,np.eye(N)),'nodal_exact_congruence':error(Mn,target),'back_congruence':error(mm(T.T,mm(Mn,T)),np.eye(N)),'nodal_condition':float(np.linalg.cond(Mn))})
source=np.loadtxt(d/'source/output.txt');assert source.shape[1]==150
inp=np.linalg.norm(source[:,33:55],axis=1);out=np.linalg.norm(source[:,:22],axis=1)
ni=source[:,146:148];no=source[:,148:150]
normals={'rows':len(source),'input_scaled':float(np.max(np.abs(ni)/np.maximum(1.,inp[:,None]))),'output_scaled':float(np.max(np.abs(no)/np.maximum(1.,out[:,None]))),'input_absolute':float(np.max(np.abs(ni))),'output_absolute':float(np.max(np.abs(no)))}
proof={'origin_frame_evaluated':False,'statement':'Every solid-CG component is r^L W(r^2) times a bounded angular polynomial. In the exact core normalization coefficients are constant; Ds of L0 is 2r W_rho, and Ds of L>=1 has leading L r^(L-1) W. Thus all q/V trial fields remain bounded at r=0, H/Kn remain bounded, and r^2 y_i^T H Kn y_j tends to zero. No origin trace or cross-L condition is imposed.','dependencies':['Frozen regular basis and exact polynomial core local gate']}
passed=all(z[k]['scaled']<=5e-11 for z in mass for k in ('modal_I','nodal_exact_congruence','back_congruence')) and max(normals['input_scaled'],normals['output_scaled'])<=5e-11
report={'passed':passed,'matrix_sha256':r['operator_sha256'],'source_output_sha256':sha(d/'source/output.txt'),'checker_sha256':sha(__file__),'polynomial_exact_sha256':sha(exactpath),'mass':mass,'normals':normals,'origin_flux_proof':proof,'tolerances':{'exact_polynomial_and_normals_scaled':5e-11},'scope':'Source normals and independent polynomial dense mass; coupled E checks are separate'}
(dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':passed,'normals':normals,'mass_max_scaled':max(z[k]['scaled'] for z in mass for k in ('modal_I','nodal_exact_congruence','back_congruence'))},indent=2));assert passed
