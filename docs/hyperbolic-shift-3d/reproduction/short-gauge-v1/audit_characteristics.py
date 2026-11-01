"""Read-only SG1 characteristic samples on initial/older states, not an SG1 trajectory.

Reuse the native audited geometric principal block; independently replace the
three gauge rows/blocks using the SG1 interval. The .2--.9 control is checked
against the native direct-outer matrix. This is not a global hyperbolicity proof.
"""
from pathlib import Path
import argparse, ctypes as C, hashlib, json, sys
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-sg1-endpoint',action='store_true',help='Also audit the verified N16 SG1 10M checkpoint, preserving the original audit file.')
args=parser.parse_args()
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d'))
from audit_inner_characteristics import spaces

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def smooth(t):
    t=np.clip(t,0,1);x=np.minimum(t,1-t);v=x**5*(126+x*(-420+x*(540+x*(-315+70*x))))
    return np.where(t>.5,1-v,v)
def coefficients(r,alpha,chi,end):
    o=(1-r*r)/2;s=smooth((end-r)/(end-.2));F=alpha*alpha+4*o*o*alpha
    L=chi*((1-s)*F+s*2*o*alpha)+2*s*s*np.maximum(o*o*chi,1e-5)
    mu=s+(1-s)*1.5*chi*F
    return o,s,F,L,mu
coords_path=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv'
c=np.genfromtxt(coords_path,delimiter=',',names=True);r=c['r']; xyz=np.column_stack([c[k] for k in ['x','y','z']]);active=(c['volume']>0)&(r>.1)&(r<.95)
libpath=ROOT/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/libhyperbolic_shift_3d_inner_characteristics.dylib'
lib=C.CDLL(str(libpath));fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.inner_gauge_symbol.argtypes=[C.c_int]+[C.c_double]*3+[fp,fp,fp]
initial=ROOT/'output/hyperbolic-core-shell/resolution-20m-v1/archive/retired-before-n32-ac1/inputs/puncture-continuous-n16.txt'
cases=[('initial',initial,'35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca')]
for label,variant,trial,sha in [
 ('IT1_10M','inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01','37665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f'),
 ('DO1_10M','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c'),
 ('DG1_10M','damping-ablation-v1','n16-damping-ablation-fresh10m-v1-segment01','b582e457f8a8ba9b143436225bb57318b7e28a241fb495b913eea4787bfa04cf')]:
    cases.append((label,ROOT/'output/hyperbolic-core-shell'/variant/'aurora/campaigns'/trial/'output/puncture.checkpoint',sha))
if args.include_sg1_endpoint:
    cases.append(('SG1_10M',ROOT/'output/hyperbolic-core-shell/short-gauge-v1/aurora/campaigns/n16-short-gauge-fresh10m-v1-segment01/output/puncture.checkpoint','d581c2860a1049c5a79568c90658b56463ebe8487229e06f7df3d34b27f80749'))
scope='SG1 coefficients on original initial data and frozen older 10M states; not an SG1 evolution or global hyperbolicity proof'
if args.include_sg1_endpoint:
    scope='SG1 coefficients on initial, older controls and verified SG1 10M endpoint; sampled directions at 0.1<r<0.95 only. Not a full-trajectory, puncture, scri or global hyperbolicity proof.'
result=dict(scope=scope,interval=[.2,.56],radius_audited=[.1,.95],coordinates_sha256=digest(coords_path),native_library_sha256=digest(libpath),states={})
for label,path,sha in cases:
    assert digest(path)==sha
    a=np.loadtxt(path,skiprows=1) if label=='initial' else np.memmap(path,dtype='<f8',mode='r',offset=168,shape=(len(r),31))
    alpha=a[:,18]+(1+r*r)/2;chi=a[:,0];g=np.zeros((len(r),3,3))
    for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=a[:,1+k]+(i==j)
    assert np.all(np.linalg.eigvalsh(g[active])>0) and np.all(alpha[active]>0) and np.all(chi[active]>0)
    _,_,_,L,mu=coefficients(r,alpha,chi,.56);ids=np.flatnonzero(active)
    gaps={'transverse_shift_minus_lapse':mu-L,'longitudinal_shift_minus_lapse':4*mu/3-L,'lapse_minus_physical':L-alpha*alpha*chi}
    record=dict(state_sha256=sha,active_cells=len(ids),gaps={},sampled_symbols=0,incomplete=[],max_spectrum_error=0.,max_native_control_error=0.,max_balanced_basis_condition=0.)
    for key,gap in gaps.items():
        k=ids[np.argmin(gap[ids])];record['gaps'][key]=dict(minimum=float(gap[k]),minimum_radius=float(r[k]),negative_cells=int(np.sum(gap[ids]<0)),zero_cells=int(np.sum(gap[ids]==0)))
    selected=np.unique([ids[np.argmin(abs(r[ids]-target))] for target in np.linspace(.105,.945,40)])
    for k in selected:
        radius=float(r[k]);al=float(alpha[k]);ch=float(chi[k]);metric=g[k];normals=[*np.eye(3),np.ones(3)/np.sqrt(3),xyz[k]/r[k]]
        for n in normals:
            native=np.zeros((26,26));assert lib.inner_gauge_symbol(2,radius,al,ch,np.ascontiguousarray(metric),np.ascontiguousarray(n),native)==0
            def matrix(end):
                o,s,F,l,mu=coefficients(radius,al,ch,end);P=native.copy();P[16:23,:]=0
                P[16,11]=-((1-s)*F+s*2*o*al);P[16,20:23]=s*o*max(o*o*ch,1e-5)*(np.linalg.inv(metric)@n)
                for i in range(3):P[17+i,13+i]=s;P[17+i,23+i]=(1-s)*1.5*ch*F
                if s:P[20:23,16]=s*2/o*n
                q=n@np.linalg.inv(metric)@n;cp=al*np.sqrt(ch*q);cl=np.sqrt(l*q);cs=np.sqrt(4*mu*q/3);ct=np.sqrt(mu*q)
                return P,np.array([0.]*6+[-cp,cp]*6+[-cl,cl,-cs,cs]+[-ct,ct]*2)
            control,_=matrix(.9);control_error=float(np.max(abs(control-native)));assert control_error<2e-9
            P,expected=matrix(.56);ev=np.linalg.eigvals(P);error=float(max(np.max(abs(np.sort(ev.real)-np.sort(expected))),np.max(abs(ev.imag))));assert error<2e-7
            z=spaces(P,expected);record['sampled_symbols']+=1;record['max_spectrum_error']=max(record['max_spectrum_error'],error);record['max_native_control_error']=max(record['max_native_control_error'],control_error)
            if not z['complete']:record['incomplete'].append(dict(cell=int(k),radius=radius,normal=n.tolist(),audit=z))
            else:record['max_balanced_basis_condition']=max(record['max_balanced_basis_condition'],z['balanced_basis_condition'])
    assert digest(path)==sha;result['states'][label]=record
output=HERE/('characteristic-audit-sg1-endpoint.json' if args.include_sg1_endpoint else 'characteristic-audit.json')
output.write_text(json.dumps(result,indent=2)+'\n')
for label,v in result['states'].items():print(label,{k:v[k] for k in ['sampled_symbols','max_spectrum_error','max_balanced_basis_condition']},'incomplete',len(v['incomplete']),{k:g['minimum'] for k,g in v['gaps'].items()})
