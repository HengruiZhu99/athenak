#!/usr/bin/env python3
"""Independent DOP853/beta-function audit of shared compact initial geometry."""
import argparse,hashlib,io,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import betainc,betaln
from kernel import Kernel
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/hyperbolic-shift-3d/reproduction/refined-convergence-v1'))
from study_compact_slicing import Profile,solve_shape,A,K_SCALE,M

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--sampler',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel(a.library,separated=True,compact_end=.97);anchor=Kernel(a.library,separated=True)
 r=np.unique(np.r_[np.linspace(A+1e-6,.999,401),A,A-1e-6,.97,.97-1e-6,.97+1e-6])
 def sample(tol):
  raw=subprocess.check_output([a.sampler,str(tol),'.97'],input='\n'.join(map(str,r)),text=True)
  lines=raw.splitlines();return np.fromstring(lines[0],sep=' '),np.loadtxt(io.StringIO('\n'.join(lines[1:])))
 cfg,c=sample(2e-13);tightcfg,t=sample(3e-14);p=solve_shape(.97);ref=Profile(.97,p,tolerance=3e-14);f=ref.fields(r)
 # DOP853 integrates both map and the native FREE shift integral independently.
 def rhs(x,u):
  z=np.clip((x-A)/(.97-A),0,1);S=betainc(p,7,z)
  Sp=np.exp((p-1)*np.log(z)+6*np.log1p(-z)-betaln(p,7))/(.97-A) if 0<z<1 else 0
  y=u[0];F=np.sqrt(y*y-2*M*y**3+S*S);R=1/y
  rho=(R-M+np.sqrt(R*(R-2*M)))/2;psi=1+M/(2*rho)
  alpha=(1-S)/psi**2+S*np.sqrt(1-2*M/R+R*R)
  return [-F/x,-alpha*Sp*y/F]
 sol=solve_ivp(rhs,(.97,A-1e-6),[ref.sol.sol(.97)[0],0],method='DOP853',rtol=3e-14,atol=3e-16,max_step=.0005,dense_output=True)
 assert sol.success
 o=(1-r*r)/2;R=f['R'];rho=(R-M+np.sqrt(R*(R-2*M)))/2;psi=1+M/(2*rho)
 free_alpha=o*((1-f['S'])/psi**2+f['S']*np.sqrt(1-2*M/R+R*R))
 Ar=2*(f['lr']-f['lt'])/(3*o)
 J=np.where(r<.97,sol.sol(np.minimum(r,.97))[1],0)
 expected=np.column_stack((r,f['chi'],Ar,f['trace'],free_alpha,J,R))
 errors=np.max(abs(c-expected)/(1+abs(expected)),axis=0)
 refined=np.max(abs(c-t)/(1+abs(t)),axis=0)
 assert max(errors)<3e-10,errors
 assert max(refined)<3e-10,refined
 assert np.all(np.diff(c[:,6])>0)
 assert abs(cfg[0]-A)<1e-12 and abs(cfg[1]-K_SCALE)<1e-12 and abs(cfg[2]-p)<2e-8
 assert np.array_equal(k.config[2:12],anchor.config[2:12])
 # Check actual initializer after conformal-frame transform and free-lapse blend.
 q,src,omega=k.initial(r);O=omega[:,0];factor=O/o
 blend=np.clip((r-k.config[4])/(k.config[7]-k.config[4]),0,1)
 blend=126*blend**5-420*blend**6+540*blend**7-315*blend**8+70*blend**9
 alpha=free_alpha*factor*(1-blend)+blend*f['alpha']*factor
 expected_api=np.column_stack((f['chi']/factor**2,Ar/factor,f['trace'],alpha,f['beta']))
 api_errors=np.max(abs(q[:,[0,3,5,8,9]]-expected_api)/(1+abs(expected_api)),axis=0)
 # The source is intentionally inactive within the flat telegrapher core.
 mask=r>k.config[4]
 source_errors=[float(np.max(abs(src[mask,0]-f['alpha'][mask]*factor[mask]))),float(np.max(abs(src[mask,1]-(1-f['S'][mask]))))]
 assert max(api_errors)<3e-10 and max(source_errors)<3e-10,(api_errors,source_errors)
 # Finite differences of the C++ fields test derivatives, independently of C++ RHS.
 h=2e-5;rd=np.linspace(A+.002,.995,201)
 vals=[]
 for off in [-2,-1,1,2]:vals.append(k.initial(rd+off*h)[0])
 d=(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h);fd=ref.fields(rd)
 outer=rd>k.config[7]
 derivative_error=float(np.max(abs(d[outer,8]-fd['alpha_prime'][outer])/(1+abs(fd['alpha_prime'][outer]))))
 assert derivative_error<2e-6,derivative_error
 # All reference source components must cancel the stationary OUTER gauge RHS.
 # Construct these jets from the independent map, not from C++ reference fields.
 from evolution import weights
 rs=np.linspace(k.config[8]+.001,.995,65)
 def stationary(x):
  v=ref.fields(x);om=(1-x*x)/2;Q=np.zeros((len(x),12))
  Q[:,0]=v['chi'];Q[:,1:3]=1;Q[:,3]=2*(v['lr']-v['lt'])/(3*om)
  Q[:,4]=-Q[:,3]/2;Q[:,5]=v['trace'];Q[:,8]=v['alpha'];Q[:,9]=v['beta'];return Q
 Q=stationary(rs);d1=[];d2=[]
 for x in rs:
  nodes=x+np.arange(-3,4)*.0002;values=stationary(nodes)-stationary(np.array([x]))
  d1.append(weights(nodes,x,1)@values);d2.append(weights(nodes,x,2)@values)
 _,sources,_=k.initial(rs);reg,pole,_,_=k.parts(rs,Q,np.array(d1),np.array(d2),sources)
 stationary_rate=np.max(abs(reg+pole/((1-rs*rs)/2)[:,None]),axis=0)
 assert max(stationary_rate[[8,9]])<2e-8 and max(stationary_rate)<3e-6,stationary_rate
 # The physical geometry is constrained, but the actual central gauge is not stationary.
 from evolution import ChartEvolution
 e=ChartEvolution(k,64,normal_policy='analytic',reference_jets=True,reference_step=.000125,reference_tail='scri')
 rate=e.rhs(e.initial.copy());central_rate=float(np.max(abs(rate[e.r<.4])))
 assert central_rate>1e-3
 rejected=[]
 for end in [-.1,.8,1.,float('nan')]:
  try:Kernel(a.library,separated=True,compact_end=end)
  except ValueError:rejected.append(str(end))
 assert len(rejected)==4
 out=dict(p_CPP=float(cfg[2]),p_independent=float(p),anchor_a=float(cfg[0]),anchor_k=float(cfg[1]),join_relative_error=float(cfg[3]),end_R_M=float(k.compact_config[2]),native_scaled_errors=errors.tolist(),tolerance_refinement_scaled_errors=refined.tolist(),actual_initializer_scaled_errors=api_errors.tolist(),source_absolute_errors=source_errors,lapse_derivative_scaled_error=derivative_error,stationary_outer_RHS_max_by_field=stationary_rate.tolist(),independent_analytic_H_max=float(max(abs(f['H_exact']))),central_actual_RHS_max=central_rate,rejected_inputs=rejected,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Spherical initial geometry and shared off-constraint equation checks only; no evolution stability, boosted, or 3D claim.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output/'profiles.npz',r=r,cpp=c,tight=t,independent=expected,q=q,source=src)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for axis,col,label in zip(ax.flat,[1,3,4,6],['native chi','K','free lapse','R']):axis.plot(r,c[:,col],label='shared C++');axis.plot(r,expected[:,col],'--',label='independent');axis.set(xlabel='compact r',ylabel=label)
 ax.flat[0].legend();fig.savefig(a.output/'geometry.png',dpi=140);plt.close(fig)
 print(json.dumps(out));k.close();anchor.close()
if __name__=='__main__':main()
