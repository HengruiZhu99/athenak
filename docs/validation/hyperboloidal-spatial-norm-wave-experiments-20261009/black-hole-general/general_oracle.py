"""100-digit independent general-parameter null-rate/linear-response oracle."""
import json
from pathlib import Path
import mpmath as mp
import frozen_outer_oracle as f
mp.mp.dps=100

CONFIGS=[('original','1','.5','.5','1.5','1.5','1'),
 ('light','1','.5','.2','1','1.5','1'),
 ('a075','1','.75','.35','1.5','1.5','1'),
 ('rho25','1','1','.5','2.5','1.5','1'),
 ('scale_default','2','1','1','1.5','1.5','1'),
 ('large_a','2','2','.3','2','1.5','1'),
 ('small_S','.8','.8','.16','1.25','1.5','1'),
 ('scale_covariant','2','1','1','1.5','.75','.5')]

def configure(S,a,M,rho,nu,etaR):
 f.S,f.a,f.M,f.nu,f.regular_eta=map(mp.mpf,(S,a,M,nu,etaR))
 return mp.mpf(rho)*f.S/f.a**2,f.S/f.a*(1-1/mp.mpf(rho))

def evaluate(O,eta,C,dB=mp.mpf(0)):
 v=f.evaluate(O,xi_rate=1/f.a,db2=dB)
 r=mp.sqrt(f.S*f.S-2*f.a*f.S*O);op=-r/(f.a*f.S);L=f.S/f.a-O
 m=f.M*O/(2*r);psi=1+m
 alpha=(1-m)/psi*L;beta=-r/f.a*(1-m)/psi**3+dB*O*O
 wn=-beta*op/alpha
 chdiv=-f.M/(2*r)*(4+6*m+4*m*m+m**3)/psi**4
 extra=-eta*(v['beta_deviation_div']+C*chdiv)
 v['beta_dot']+=extra;v['Ndot']+=2*wn*op/alpha*extra
 v['shift_manifold_dot']=v['beta_dot']+C*v['geometry']/(op*op)
 return v

def formula(S,a,M,rho,nu,etaR):
 return M*(M*(rho-8)+4*a*a*(2*etaR-nu)+8*a*(4-rho))/(4*S*a*(4-rho))

if __name__=='__main__':
 rows=[];summary=[]
 for name,*values in CONFIGS:
  S,a,M,rho,nu,etaR=map(mp.mpf,values);eta,C=configure(*values)
  v=evaluate(mp.mpf(0),eta,C)
  for key in ['alpha_dot','beta_dot','geometry','Ndot','shift_manifold_dot']:assert abs(v[key])<mp.mpf('1e-75'),(name,key,v[key])
  n1=mp.diff(lambda x:evaluate(x,eta,C)['Ndot'],mp.mpf(0))
  bc=mp.diff(lambda x:evaluate(x,eta,C,mp.mpf(1))['Ndot'],mp.mpf(0))-n1
  dB=-n1/bc;expected=formula(S,a,M,rho,nu,etaR)
  assert abs(dB-expected)<mp.mpf('1e-75')
  assert abs(bc-2*(4-rho)/a**3)<mp.mpf('1e-75')
  residual=mp.diff(lambda x:evaluate(x,eta,C,dB)['Ndot'],mp.mpf(0))
  assert abs(residual)<mp.mpf('1e-75')
  n2=mp.diff(lambda x:evaluate(x,eta,C,dB)['Ndot'],mp.mpf(0),2)/2
  initialN2=mp.diff(lambda x:evaluate(x,eta,C,dB)['N_raw'],mp.mpf(0),2)/2
  assert abs(initialN2-(1/S**2+2*dB/(a*S)))<mp.mpf('1e-75')
  summary.append({'name':name,'S':str(S),'a':str(a),'M':str(M),'rho':str(rho),'nu':str(nu),'eta_regular':str(etaR),'eta':str(eta),'C':str(C),'original_Ndot_Omega':str(n1),'delta_b2_coefficient':str(bc),'delta_b2':str(dB),'corrected_Ndot_Omega':str(residual),'corrected_Ndot_Omega2':str(n2),'corrected_Nraw_Omega2':str(initialN2)})
  for D in [mp.mpf(0),dB]:
   for O in [mp.mpf('1e-4'),mp.mpf('1e-12'),mp.mpf('1e-40'),mp.mpf(0)]:
    val=evaluate(O,eta,C,D);rows.append({'name':name,'delta_b2':str(D),'Omega':str(O),**{k:str(x) for k,x in val.items()}})
 # A complete obstruction and a degenerate nonunique control at rho=4.
 controls=[]
 for M in [mp.mpf('.5'),mp.mpf('.125')]:
  eta,C=configure(1,'.5',M,4,'1.5',1)
  n1=mp.diff(lambda x:evaluate(x,eta,C)['Ndot'],mp.mpf(0))
  bc=mp.diff(lambda x:evaluate(x,eta,C,mp.mpf(1))['Ndot'],mp.mpf(0))-n1
  assert abs(bc)<mp.mpf('1e-75')
  expected=2*M*(M-mp.mpf('.5')**2*mp.mpf('.5'))/mp.mpf('.5')**4
  assert abs(n1-expected)<mp.mpf('1e-75')
  controls.append({'M':str(M),'rho':4,'original_Ndot_Omega':str(n1),'delta_b2_coefficient':str(bc),'obstruction':M==mp.mpf('.5')})
 assert abs(mp.mpf(summary[0]['delta_b2'])-mp.mpf(summary[7]['delta_b2']))<mp.mpf('1e-75')
 Path(__file__).with_name('oracle.json').write_text(json.dumps({'dps':mp.mp.dps,'summary':summary,'rows':rows,'degenerate_controls':controls},indent=2)+'\n')
 print('PASS independent general oracle: 8 parameter sets, dimensional rescaling, rho4 obstruction/nonunique controls')
 for row in summary:print(row['name'],'delta_b2=',mp.nstr(mp.mpf(row['delta_b2']),18),'Ndot_Omega2=',mp.nstr(mp.mpf(row['corrected_Ndot_Omega2']),18))
