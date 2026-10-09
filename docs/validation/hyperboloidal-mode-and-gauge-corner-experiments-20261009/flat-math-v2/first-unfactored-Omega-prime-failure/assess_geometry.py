"""Derived alternative Minkowski height for a flat Penrose spatial metric; math only."""
import hashlib,json,time
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
HERE=Path(__file__).resolve().parent
S=mp.mpf(1);a=mp.mpf('.5');r0=mp.mpf('.05');r1=mp.mpf('.95')
def omega(r):
 if r<=r0:return mp.mpf(1)
 outer=(S-r)*(S+r)/(2*a*S)
 if r>=r1:return outer
 s=(r-r0)/(r1-r0);t=(r1-r)/(r1-r0);w=1/(1+mp.exp(1/s-1/t))
 return 1-w+w*outer
def vals(r,new):
 O=omega(r);Op=mp.diff(omega,r);L=O-r*Op
 if new:b=mp.sqrt((-r*Op)*(2*O-r*Op))
 else:
  s=(r-r0)/(r1-r0);t=(r1-r)/(r1-r0);w=1/(1+mp.exp(1/s-1/t))
  b=r*w/a
 A=L if new else mp.sqrt(O*O+b*b)
 return O,L,b,A

def Aref(r,new):return vals(r,new)[3]
def boost(r,new):return vals(r,new)[2]
def Kphys(r,new):
 O,L,b,A=vals(r,new);Op=mp.diff(omega,r);bp=mp.diff(lambda z:boost(z,new),r)
 return -(O*(bp+2*b/r)-3*b*Op)/L
started=time.monotonic();rows=[]
for j in range(1,400):
 r=r0+(r1-r0)*mp.mpf(j)/400
 row={'r':float(r)}
 for name,new in [('original',False),('flat',True)]:
  O,L,b,A=vals(r,new);bp=mp.diff(lambda z:boost(z,new),r)
  hR=b/A;g=L*L/(A*A)
  row[name]={'alpha':float(A),'dalpha':float(mp.diff(lambda z:Aref(z,new),r)),
    'ddalpha':float(mp.diff(lambda z:Aref(z,new),r,2)),
    'Kphys':float(Kphys(r,new)),'dKphys':float(mp.diff(lambda z:Kphys(z,new),r)),
    'penrose_radial_metric':float(g),'chi':float((A/L)**(mp.mpf(2)/3)),
    'outgoing':float(A*(A+b)/L),'ingoing':float(-A*O*O/(L*(A+b)))}
  if new:
   assert abs(A*A-b*b-O*O)<mp.mpf('1e-70')
   assert abs(g-1)<mp.mpf('1e-70') and 0<hR<1
 rows.append(row)
summary={}
for name in ['original','flat']:
 summary[name]={k:{'max_abs':max(abs(row[name][k]) for row in rows),'min':min(row[name][k] for row in rows),'max':max(row[name][k] for row in rows)} for k in rows[0][name]}
result={'scope':'80-digit mathematical geometry comparison only; no actual tensor/kernel/native/global gate or stability acceptance',
 'parameters':{'S':1,'a':.5,'r0':.05,'r1':.95},
 'derived_identity':'d=-r Omega_prime>=0; b=sqrt[d(2Omega+d)]; A=L=Omega+d; Penrose spatial metric=I; chi=1; gtilde=I; Lambda=0',
 'height':'h_R=b/L; inner exactly0, outer standardCMC; sqrt of flat exponential preserves smooth endpoint analytically, floating evaluation not yet implemented',
 'sample_count':len(rows),'seconds':time.monotonic()-started,'summary':summary,'rows':rows,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'samples':len(rows),'seconds':result['seconds'],'summary':summary}))
