"""Optional bounded all-q comparator scan; a search is not a stability proof."""
from completions import *
import argparse

def scan_case(cfg,k,completion):
 assess=lambda z:assess_complete(z,k,cfg,completion)
 real=np.geomspace(1e-7,max(.5,2*k),280);ys=[assess(complex(x))['sigma_min']for x in real];minima=[]
 for i in range(1,len(real)-1):
  if ys[i]<ys[i-1] and ys[i]<ys[i+1]:
   fit=optimize.minimize_scalar(lambda z:assess(complex(z))['sigma_min'],bracket=real[i-1:i+2],method='brent',options={'xtol':1e-13,'maxiter':160});minima.append(dict(real=float(fit.x),imag=0.,**assess(complex(fit.x))))
 lo=1e-6;hi=max(.15,1.5*k);imhi=max(.15,2*k);xx=np.geomspace(lo,hi,34);yy=np.linspace(0,imhi,50);ss=np.array([[assess(complex(x,y))['sigma_min']for y in yy]for x in xx]);seeds=[]
 for i in range(len(xx)):
  for j in range(len(yy)):
   slab=ss[max(0,i-1):min(len(xx),i+2),max(0,j-1):min(len(yy),j+2)]
   if ss[i,j]<=np.min(slab):seeds.append((float(ss[i,j]),i,j))
 seeds=sorted(seeds)[:12];complexmins=[]
 for val,i,j in seeds:
  fit=optimize.minimize(lambda z:assess(complex(*z))['sigma_min'],[xx[i],yy[j]],method='Nelder-Mead',bounds=[(lo,hi),(0,imhi)],options={'xatol':1e-11,'fatol':1e-11,'maxiter':250});complexmins.append(dict(real=float(fit.x[0]),imag=float(fit.x[1]),**assess(complex(*fit.x))))
 return dict(config=asdict(cfg),k=k,real_minima=minima,complex_minima=complexmins,bounds={'real':[lo,hi],'imag':[0,imhi]})

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 cases={'weak_G1':Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295),'strong_G1':Config(alpha=.6531769730886202,chi=.4266401581732121,beta_n=.2255366474924469)};out=[]
 for completion in ['q_dirichlet','q_wave']:
  for name,cfg in cases.items():
   for k in ([.01,.05,.1]if name=='weak_G1'else [.1,1.,2*np.pi]):
    q=dict(completion=completion,case=name,**scan_case(cfg,k,completion));out.append(q)
    print(completion,name,k,'minimum',min(p['sigma_min']for p in q['complex_minima']),flush=True)
 args.output.write_text(json.dumps({'scope':__doc__,'data':out},indent=2)+'\n')
