"""Independent comparison to origin/main's boosted AF geometry and lapse recipe."""
import sys,subprocess,json,hashlib,tempfile
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
sys.path.insert(0,str(OUT.parent/'boosted-puncture-v1'))
from embedding_probe import BoostedEmbedding
b=BoostedEmbedding();m=b.m;v=b.v;G=b.G;k=b.base.k
# Transcribed directly from src/pgen/tests/z4c_boosted_puncture.cpp on origin/main.
def af(X):
 rho=np.sqrt((G*X[0])**2+X[1]**2+X[2]**2);psi=1+m/(2*rho);a=(1-m/(2*rho))/psi
 B=np.sqrt(G*G*(1-v*v*a*a/psi**4));metric=psi**4*np.diag([B*B,1,1]);ap=4*m/(m+2*rho)**2
 second=(4*v*v*(m-2*rho)**2/(m+2*rho)**3+4*v*v*(m-2*rho)/(m+2*rho)**2-m*(m+2*rho)**3/(4*rho**5))/(psi**4-v*v*(m-2*rho)**2/(m+2*rho)**2)
 K=np.zeros((3,3));K[0,0]=G*G*B*X[0]*v/rho*(2*ap-a/2*second)
 K[1,1]=K[2,2]=2*G*G*X[0]*v*a*(-m/(2*rho*rho))/(psi*B*rho)
 K[0,1]=K[1,0]=B*X[1]*v/rho*(ap-a/2*second);K[0,2]=K[2,0]=B*X[2]*v/rho*(ap-a/2*second)
 return metric,K,psi**-2*B**(-1/3)
points=np.array([[.12,.04,.03],[-.16,.06,.02],[.04,.08,.13],[.07,-.02,.01]])
with tempfile.TemporaryDirectory() as t:
 exe=str(Path(t)/'point');subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT),str(ROOT/'tst/hyperbolic_shift_3d/test_native_boosted_wormhole.cpp'),'-o',exe],check=True)
 def query(xs,free):
  p=subprocess.run([exe,'.1']+(['free'] if free else []),input=''.join(' '.join(map(str,x))+'\n' for x in xs),capture_output=True,text=True,check=True)
  return np.array([list(map(float,line.split())) for line in p.stdout.splitlines()])
 rows=[];new=query(k*points,True);old=query(k*points,False)
 for X,q in zip(points,new):
  gamma,K,alpha=af(X);g=b.geometry(k*X);om=(1-np.dot(k*X,k*X))/2
  rows.append(dict(X=X.tolist(),metric_error=float(np.max(abs(gamma-k*k*g['metric']))),K_error=float(np.max(abs(K-k*k*g['curvature']))),physical_lapse_error=float(abs((q[18]+(1+np.dot(k*X,k*X))/2)/om-alpha)),initial_shift=(q[22:25]-k*X).tolist()))
 far=np.array([r*np.array([.3,.4,np.sqrt(.75)]) for r in [.2,.3,.56,.8,.95,.99,.999]])
 fn=query(far,True);fo=query(far,False);unchanged=np.max(abs(np.delete(fn-fo,[18,22,23,24],axis=1)))
 scri=[]
 for x,q in zip(far,fn):scri.append(dict(r=float(np.linalg.norm(x)),lapse=float(q[18]+(1+np.dot(x,x))/2),beta=(q[22:25]-x).tolist()))
 result=dict(main_commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),AF_source_sha256=hashlib.sha256((ROOT/'src/pgen/tests/z4c_boosted_puncture.cpp').read_bytes()).hexdigest(),rows=rows,geometric_fields_unchanged=float(unchanged),scri_approach=scri,reference_velocity=v,compact_reference_velocity=k*v,initial_tracker_velocity=0,reference_exit_time_over_M=.18/(k*v*m),target_time_over_M=10)
 assert max(r['metric_error'] for r in rows)<1e-10
 assert max(r['K_error'] for r in rows)<1e-10,rows
 assert max(r['physical_lapse_error'] for r in rows)<1e-13
 assert max(abs(t) for r in rows for t in r['initial_shift'])==0 and unchanged==0
 (OUT/'af-gauge-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
