#!/usr/bin/env python3
"""Independent TT Penrose pullback factorization; native dynamics remain untested."""
from pathlib import Path
import mpmath as m
import json,hashlib,ast,math,argparse
from goal1_global_clock_oracle import Clock
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);m.mp.dps=90;R=Path(__file__).resolve().parents[2];old=R/'tst/cubed_sphere/check_goal1_tensor_oracle.py';cache=json.loads((R/'docs/validation/goal1-minkowski-20261018/native-global-clock-v1/reused-height-cache.json').read_text())
protocol={'scope':'Two-polarization compact retarded TT Penrose metric/ADM/chi/gtilde transformation only. No native A/P/Lambda jets, native point source or propagation admission.','precision':90,'compact_profile':'f(u)=[1-((u+.35)/.15)^2]^16 for |u+.35|<.15, zero otherwise, C15','r_times':[['.3','0'],['.7','.3'],['.95','.4'],['.99999','.4']],'directions':[[1,0,0],[.8,-.48,.36],[1,1,1]],'polarizations':['plus','cross'],'budget':'1e-50 scaled direct-vs-factored finite pullback; exact endpoint factor limit checked from finite approach separately','negative_control':'Omit radial compactification Jacobian stretch','new_evolved_fields':0};(a.output/'protocol-v1.json').write_text(json.dumps(protocol,indent=2)+'\n')
def profile(u,j):
 s=(u+m.mpf('.35'))/m.mpf('.15')
 if abs(s)>=1:return m.mpf(0)
 return sum((-1)**k*m.binomial(16,k)*math.factorial(2*k)/math.factorial(2*k-j)*s**(2*k-j)for k in range((j+1)//2,17))/m.mpf('.15')**j
# Reuse the independently verified Cartesian radial tensor expression verbatim,
# substituting the explicitly versioned compact profile. No old experiment rerun.
module=ast.parse(old.read_text());keep=[x for x in module.body if isinstance(x,ast.FunctionDef)and x.name in ['radial','tensor']];namespace={'mp':m,'math':math,'profile':profile};exec(compile(ast.Module(body=keep,type_ignores=[]),str(old),'exec'),namespace);tensor=namespace['tensor'];rows=[];endpoint=[];norm=lambda M:max(abs(z)for z in M)
for sector,E in [('plus',m.diag([1,-1,0])),('cross',m.matrix([[0,1,0],[1,0,0],[0,0,0]]))]:
 for rv,tv in protocol['r_times']:
  r=m.mpf(rv);t=m.mpf(tv);clock=Clock(r,cache);o,L,A,b=clock.reference(r);U=t+clock.uref(r);rho=r/o;T=U+rho;z=o/r;f=[profile(U,k)for k in range(5)]
  pp=z*(f[4]+z*(2*f[3]+z*(3*f[2]+z*(3*f[1]+3*z*f[0]))));ss=(pp-6*z**3*(f[2]+3*z*f[1]+3*z*z*f[0]))/2
  bb=-2*L*o/r**2*(f[3]+3*z*f[2]+6*z*z*f[1]+6*z**3*f[0]);aa=6*L*L*o/r**3*(f[2]+3*z*f[1]+3*z*z*f[0]);G=(L/A)**2;beta=-b*A/L;chi=G**(-m.mpf(1)/3)
  for direction in protocol['directions']:
   n=m.matrix([m.mpf(str(v))for v in direction]);n/=m.sqrt((n.T*n)[0]);I=m.eye(3);NN=n*n.T;P=I-NN;v=E*n;e=(n.T*v)[0];vp=v-e*n;Ep=P*E*P;cross=vp*n.T+n*vp.T
   H=tensor(T,list(rho*n),E);J=(I+(L/o-1)*NN)/o;direct=o*o*J.T*H*J;factored=pp*Ep+ss*e*P+bb*cross+aa*e*NN
   gamma=P+G*NN;inverse=P+NN/G;B=beta*n;dBeta=-inverse*direct*B;dAlpha=-(B.T*direct*B)[0]/(2*A);dChi=-chi*sum((inverse*direct)[i,i]for i in range(3))/3;dGtilde=chi*direct+dChi*gamma
   dBeta2=-beta*(bb*vp+aa/G*e*n);dAlpha2=-beta*beta*aa*e/(2*A);dChi2=-chi*e*(-pp+2*ss+aa/G)/3;dGtilde2=chi*factored+dChi2*gamma
   errors={'Penrose_metric':norm(direct-factored)/max(1,norm(direct)),'native_shift':norm(dBeta-dBeta2)/max(1,norm(dBeta)),'native_lapse':abs(dAlpha-dAlpha2)/max(1,abs(dAlpha)),'native_chi':abs(dChi-dChi2)/max(1,abs(dChi)),'native_gtilde':norm(dGtilde-dGtilde2)/max(1,norm(dGtilde)),'linear_unit_determinant':abs(sum(((inverse/chi)*dGtilde2)[i,i]for i in range(3)))/max(1,norm(dGtilde2))};negative=norm(direct-H)/max(1,norm(direct));rows.append({'sector':sector,'r':rv,'t':tv,'direction':direction,'errors':{k:float(v)for k,v in errors.items()},'omitted_radial_stretch_negative':float(negative)})
 # Exact endpoint is a factor limit, not a boundary condition or dynamics test.
 U=m.mpf('.4')+m.mpf(cache['outer_constant']);f=[profile(U,k)for k in range(5)];limits=[f[4],f[4]/2,-4*f[3],24*f[2]];approach=[]
 for delta in ['1e-4','1e-6','1e-8']:
  r=1-m.mpf(delta);clock=Clock(r,cache);o,L,A,b=clock.reference(r);u=m.mpf('.4')+clock.uref(r);z=o/r;ff=[profile(u,k)for k in range(5)];pp=(ff[4]+z*(2*ff[3]+z*(3*ff[2]+z*(3*ff[1]+3*z*ff[0]))))/r;ss=(pp-6*z*z/r*(ff[2]+3*z*ff[1]+3*z*z*ff[0]))/2;bb=-2*L/r**2*(ff[3]+3*z*ff[2]+6*z*z*ff[1]+6*z**3*ff[0]);aa=6*L*L/r**3*(ff[2]+3*z*ff[1]+3*z*z*ff[0]);error=max(abs(x-y)/max(1,abs(y))for x,y in zip([pp,ss,bb,aa],limits));approach.append({'delta':delta,'scaled_factor_limit_error':float(error)})
 endpoint.append({'sector':sector,'delta_gamma_divided_by_Omega_coefficient_limits':list(map(str,limits)),'finite_approach':approach})
maxerr=max(v for x in rows for v in x['errors'].values());review={'scope':protocol['scope'],'finite_cases':len(rows),'max_scaled_error':maxerr,'finite_pullback_passed':maxerr<1e-50,'negative_max':max(x['omitted_radial_stretch_negative']for x in rows),'endpoint_factor_limits':endpoint,'records':rows,'original_physical_tensor_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_evolved_fields':0,'native_curvature_trace_connection_source_or_propagation_admitted':False};(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items()if k not in ['records','endpoint_factor_limits']},indent=2));assert review['finite_pullback_passed']and review['negative_max']>.01
