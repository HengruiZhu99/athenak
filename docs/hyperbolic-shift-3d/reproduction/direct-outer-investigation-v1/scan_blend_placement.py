"""Read-only frozen-state audit of later gauge transitions; no evolution changes."""
from pathlib import Path
import ctypes as C,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d'))
from audit_inner_characteristics import spaces

def smooth(t):
 t=np.clip(t,0,1);x=np.minimum(t,1-t);v=x**5*(126+x*(-420+x*(540+x*(-315+70*x))));return np.where(t>.5,1-v,v)
def coefficients(r,alpha,chi,begin):
 o=(1-r*r)/2;s=smooth((.9-r)/(.9-begin));F=alpha*alpha+4*o*o*alpha
 lapse=chi*((1-s)*F+s*2*o*alpha)+2*s*s*np.maximum(o*o*chi,1e-5)
 shift=s+(1-s)*1.5*chi*F
 return s,lapse,shift

libpath=ROOT/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/libhyperbolic_shift_3d_inner_characteristics.dylib'
lib=C.CDLL(str(libpath));fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.inner_gauge_symbol.argtypes=[C.c_int]+[C.c_double]*3+[fp,fp,fp]
def symbol(r,alpha,chi,g,n,begin):
 P=np.zeros((26,26));assert lib.inner_gauge_symbol(2,r,alpha,chi,np.ascontiguousarray(g),np.ascontiguousarray(n),P)==0
 native=P.copy();o=(1-r*r)/2;s,L,mu=coefficients(r,alpha,chi,begin);F=alpha*alpha+4*o*o*alpha
 P[16:23,:]=0;P[16,11]=-((1-s)*F+s*2*o*alpha)
 P[16,20:23]=s*o*max(o*o*chi,1e-5)*(np.linalg.inv(g)@n)
 for i in range(3):P[17+i,13+i]=s;P[17+i,23+i]=(1-s)*1.5*chi*F
 if s: P[20:23,16]=s*2/o*n
 if begin==.2:assert np.max(np.abs(P-native))<2e-9
 q=n@np.linalg.inv(g)@n;cp=alpha*np.sqrt(chi*q);cl=np.sqrt(L*q);cs=np.sqrt(4*mu*q/3);ct=np.sqrt(mu*q)
 expected=np.array([0.]*6+[-cp,cp]*6+[-cl,cl,-cs,cs]+[-ct,ct]*2)
 ev=np.linalg.eigvals(P);error=max(np.max(abs(np.sort(ev.real)-np.sort(expected))),np.max(abs(ev.imag)))
 info=spaces(P,expected);return info['complete'],float(error)

coords=np.genfromtxt(ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv',delimiter=',',names=True);r=coords['r'];N=len(r)
cases=[('IT1','inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01','37665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f'),('DO1','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c')]
report=dict(evolution_changed=False,scope='Frozen 10M states only; not a global hyperbolicity or future stability proof',native_library_sha256=hashlib.sha256(libpath.read_bytes()).hexdigest(),cases={})
for label,folder,campaign,sha in cases:
 p=ROOT/'output/hyperbolic-core-shell'/folder/'aurora/campaigns'/campaign/'output/puncture.checkpoint';assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 a=np.memmap(p,dtype='<f8',mode='r',offset=168,shape=(N,31));alpha=a[:,18]+(1+r*r)/2;chi=a[:,0];g=np.zeros((N,3,3))
 for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=a[:,1+f]+(i==j)
 eig=np.linalg.eigvalsh(g);assert np.min(eig)>0;Qmax=1/eig[:,0]
 ids=np.unique([int(np.argmin(abs(r-x))) for x in np.linspace(.01,.99,40)])
 result={}
 for begin in [.2,.4,.5,.56]:
  s,L,mu=coefficients(r,alpha,chi,begin);gap=4*mu/3-L;k=int(np.argmin(gap));complete=True;error=0
  for i in ids:
   for n in [*np.eye(3),np.ones(3)/np.sqrt(3),np.array([coords[k][i] for k in ['x','y','z']])/r[i]]:
    ok,e=symbol(float(r[i]),float(alpha[i]),float(chi[i]),g[i],n,begin);complete=complete and ok;error=max(error,e)
  assert error<2e-7 and complete
  result[str(begin)]=dict(inner_end=begin,outer_start=.9,maximum_weight_slope=2.4609375/(.9-begin),min_normalized_longitudinal_gap=float(gap[k]),gap_cell=k,gap_radius=float(r[k]),max_lapse_speed=float(np.sqrt(np.max(L*Qmax))),max_longitudinal_shift_speed=float(np.sqrt(np.max(4*mu/3*Qmax))),sampled_symbols=len(ids)*5,complete_sampled_bases=complete,max_spectrum_error=error)
 report['cases'][label]=dict(checkpoint_sha256=sha,cells=N,placements=result)
rplot=np.linspace(0,1,1001);fig,ax=plt.subplots(figsize=(7,4))
for begin in [.2,.4,.5,.56]:ax.plot(rplot,smooth((.9-rplot)/(.9-begin)),label=f'{begin:g} to 0.9')
ax.axvspan(.16,.18,color='k',alpha=.12);ax.set(xlabel='Compactified radius',ylabel='Inner gauge weight',title='Later transitions preserve the same outer endpoint');ax.legend(title='Transition interval');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(HERE/'blend-placement-weights.png',dpi=150)
(HERE/'blend-placement-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
