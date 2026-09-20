"""Quick reproducibility check of demonstrated roots, not a stability pass."""
import argparse,json
import scipy
from volume import *
from boundary import assess_mode
from completions import assess_complete
from diagnostics import physical,coordinate_mode

def mode(lam,k,cfg,which):
 result,(U,D,T,B,R,vh,norms)=assess_mode(lam,k,cfg,which,True)
 co=np.linalg.solve(R,vh[-1].conj());co/=np.max(abs(U@co));u=U@co;du=D@co;ddu=U@T@T@co
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0;rhs=V0@u+V1@du+V2@ddu
 result.update(lambda_value=[float(np.real(lam)),float(np.imag(lam))],k=k,boundary_scaled=float(max(abs(B@co/norms))),bulk_relative=float(np.linalg.norm(lam*u-rhs)/max(np.linalg.norm(rhs),1e-300)),constraints=physical(u,du,ddu,k,cfg.chi))
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
 cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295);k=.1
 # A separate normal-principal reduction must match the production characteristic rows.
 nd=Config(alpha=cfg.alpha,chi=cfg.chi,G=1,kappa=0,eta=0,lapse_damping=0)
 v0,_=volume(0,0,nd);vp,_=volume(1,0,nd);vm,_=volume(-1,0,nd);v1=(vp-vm)/2;v2=(vp+vm)/2-v0
 p=[6,7,8,13];q=[0,1,16,17];mat=np.block([[v1[np.ix_(p,p)],v2[np.ix_(p,q)]],[v0[np.ix_(q,p)],v1[np.ix_(q,q)]]]);err=float(max(abs(mat-char.scalar_matrix(cfg.alpha,cfg.chi,2*cfg.alpha,1)).ravel()));assert err<1e-13
 roots=[mode(z,k,cfg,'zero_rate')for z in [.024553768000726287,.015380823902007103]]
 for m in roots:assert m['sigma_min']<1e-11 and m['boundary_scaled']<1e-11 and m['bulk_relative']<1e-10
 resolved_k=np.pi/(8*32)
 resolved=[mode(z,resolved_k,cfg,'zero_rate')for z in [.0033580199984648913,.005537958992185146,.007257794403914399]]
 for m in resolved:assert m['sigma_min']<1e-11 and m['boundary_scaled']<1e-11 and m['bulk_relative']<1e-10
 assert resolved[0]['constraints']['Theta']>1e-4 and resolved[2]['constraints']['Theta']>1e-4
 assert resolved[1]['constraints']['Theta']<1e-10 and resolved[1]['constraints']['Q_max']>1e-4
 lam=cfg.beta_n*k/np.sqrt(2)+1j*cfg.alpha*np.sqrt(cfg.chi)*k/np.sqrt(2);surface=mode(lam,k,cfg,'physical_radiation_coupled')
 assert surface['sigma_min']<1e-11 and surface['constraints']['Q_max']<1e-10 and surface['constraints']['electric_Weyl']>1e-4
 coord_lam=cfg.beta_n*k;coord_symbol=assess_complete(coord_lam,k,cfg,'mixed_normal_beta_weyl');assert coord_symbol['sigma_min']<1e-11
 u,du,ddu=coordinate_mode(cfg,k);V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);rhs=V0@u+(Vp-Vm)/2@du+((Vp+Vm)/2-V0)@ddu;coord=physical(u,du,ddu,k,cfg.chi);coord['bulk_max_abs']=float(max(abs(coord_lam*u-rhs)));assert max(coord[x]for x in ['Theta','Q_max','H','M_max','electric_Weyl','magnetic_Weyl','bulk_max_abs'])<1e-12
 # Ensure the all-q controls are actually all-q rows (no substring dispatch contamination).
 for treatment in ['q_dirichlet','q_wave']:
  _,(U,D,T,B,R,vh,norms)=assess_complete(.02,k,cfg,treatment,True);expected=U[QP]if treatment=='q_dirichlet'else .02*U[QP]+(cfg.alpha*np.sqrt(cfg.chi)-cfg.beta_n)*D[QP];assert np.array_equal(B,expected)
 result={'status':'verified negative evidence; no boundary stability gate passed','numpy':np.__version__,'scipy':scipy.__version__,'config':asdict(cfg),'normal_principal_max_error':err,'original_zero_rate_roots':roots,'tangentially_resolved_original_roots':{'dx_M':32,'cells_per_tangential_wavelength':16,'k_M_inverse':resolved_k,'scope':'Frozen continuum roots; tangential resolution alone does not establish normal-profile or production-mode resolution.','roots':resolved},'physical_constraint_surface_mode':surface,'mixed_normal_beta_coordinate_mode':{'lambda_real':coord_lam,'symbol':coord_symbol,'analytic_mode':coord}}
 if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
