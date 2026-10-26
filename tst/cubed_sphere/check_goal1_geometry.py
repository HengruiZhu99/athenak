#!/usr/bin/env python3
"""Independent 90-digit coordinate-pullback and inverse-map jet audit.

No native source is imported into the oracle. Runs an already built read-only
exporter. This is geometry/continuum evidence, not native evolution acceptance.
"""
import argparse, hashlib, itertools, json, math, subprocess
from pathlib import Path
import mpmath as mp

mp.mp.dps=90
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
protocol=dict(precision_decimal_digits=90,scaled_error_budget=1e-10,
    scale='abs(native-oracle)/max(1,abs(oracle)); exact zero uses absolute error',
    negative_control='Omit map Hessian contraction for exp(.2x+.3y-.1z)',
    scope='Analytic geometry and jets only; no discretized interfaces or native time evolution',
    immutable_reference=True)
protocol_path=a.output/'geometry-protocol-v1.json'
if protocol_path.exists(): assert json.loads(protocol_path.read_text())==protocol
else: protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
requests=[]
for curvature,r0,r1 in [(.5,.05,.95),(1.,.35,.75)]:
    radii={0.,r0,r1,1.,math.nextafter(1.,0.)}
    for edge in [r0,r1]:
        radii.update([math.nextafter(edge,0.),math.nextafter(edge,1.)])
        for power in [2,3,5,8,12]:
            radii.update([edge-(r1-r0)*10**(-power),edge+(r1-r0)*10**(-power)])
    radii.update([r0+(r1-r0)*s for s in [.1,.25,.49,.5,.51,.75,.9]])
    requests += ['layer '+' '.join(map(repr,[curvature,r0,r1,r])) for r in sorted(radii)]
for stretch,patch,s,uv in itertools.product([0.,2.8592697712828787],range(6),[0.,.43,1.],[(.31,-.27),(1.,.4),(1.,1.)]):
    requests.append('map '+' '.join(map(repr,[.15,1.,stretch,patch,s,*uv])))
(a.output/'geometry-input.txt').write_text('\n'.join(requests)+'\n')
run=subprocess.run([str(a.binary.resolve())],input='\n'.join(requests)+'\n',capture_output=True,text=True,check=True)
(a.output/'geometry-native.txt').write_text(run.stdout)
(a.output/'geometry-native.stderr').write_text(run.stderr)
records=[];raw=[];endpoint=[];negative=[]
def M(x): return mp.mpf(float(x))
def compare(kind,identity,native,oracle):
    delta=abs(M(native)-oracle);scaled=delta/max(1,abs(oracle))
    records.append(dict(kind=kind,identity=identity,native=float(native),oracle=str(oracle),
        absolute_error=float(delta),scaled_error=float(scaled),passed=scaled<=mp.mpf('1e-10')))

for line in run.stdout.splitlines():
    words=line.split()
    if not words or words[0] not in ['map','layer']:continue
    if words[0]=='layer':
        cv,r0v,r1v,rv=map(float,words[1:5]);c,r0,r1,r=map(M,[cv,r0v,r1v,rv])
        def W(x):
            if x<=r0:return mp.mpf(0)
            if x>=r1:return mp.mpf(1)
            s=(x-r0)/(r1-r0);t=1-s
            e0=mp.exp(-1/s);e1=mp.exp(-1/t)
            return e0/(e0+e1)
        def O(x):return 1-W(x)+W(x)*(1-x*x)/(2*c)
        def B(x):return x*W(x)/c
        def L(x):return O(x)-x*mp.diff(O,x)
        def A(x):return mp.sqrt(O(x)**2+B(x)**2)
        def gamma(x):return (L(x)/A(x))**2
        def chi(x):return gamma(x)**(-mp.mpf(1)/3)
        def gr(x):return chi(x)*gamma(x)
        def bt(x):return -B(x)*A(x)/L(x)
        def kb(x):return -(mp.diff(B,x)+2*W(x)/c)/L(x)
        def kp(x):return O(x)*kb(x)+3*B(x)*mp.diff(O,x)/L(x)
        oracle=[mp.diff(W,r,d) for d in range(4)]
        oracle += [mp.diff(O,r,d) for d in range(3)]
        oracle += [mp.diff(A,r,d) for d in range(3)]
        oracle += [L(r),B(r),kp(r),kb(r)]
        oracle += [mp.diff(f,r,d) for f in [chi,gr,chi,bt] for d in range(3)]
        kr=-mp.diff(B,r)/L(r);kt=-W(r)/(c*L(r));trace=kr+2*kt
        oracle += [gr(r)*(kr-trace/3),chi(r)*(kt-trace/3),
            A(r)*(A(r)+B(r))/L(r),-A(r)*O(r)**2/(L(r)*(A(r)+B(r)))]
        names=['switch','switch_d','switch_dd','switch_ddd','omega','omega_d','omega_dd',
            'alpha','alpha_d','alpha_dd','L','b','Kphysical','Kbar',
            'chi','chi_d','chi_dd','g_rr','g_rr_d','g_rr_dd','g_tt','g_tt_d','g_tt_dd',
            'beta_r','beta_r_d','beta_r_dd','A_rr','A_tt','outgoing','ingoing']
        native=list(map(float,words[5:]));assert len(oracle)==30 and len(native)==33
        for name,value,expected in zip(names,native,oracle):
            compare('layer',f'a={cv},r={rv},{name}',value,expected)
        raw.append(dict(a=cv,r=rv,omega=native[4],raw_rhs_max=native[30],
            native_interior_assembled=bool(native[31]),gauge_assembled=bool(native[32])))
        if r==1:endpoint.append(raw[-1])
        # Pull back Minkowski via R=r/Omega and T=t+height(r), independently
        # checking the metric, lapse and shift rather than a copied Omega jet.
        if r>0 and r<1:
            rp=mp.diff(lambda x:x/O(x),r)
            hp=L(r)*B(r)/(O(r)**2*A(r))
            pulled=O(r)**2*(rp*rp-hp*hp)
            compare('pullback',f'a={cv},r={rv},gamma_rr',native[17]/native[14],pulled)
            compare('pullback',f'a={cv},r={rv},g_tr',native[23]*native[17]/native[14],-O(r)**2*hp)
    else:
        L,R,k=map(M,words[1:4]);patch=int(words[4]);q=list(map(M,words[5:8]));x=list(map(M,words[8:11]))
        native=list(map(float,words[11:]));assert len(native)==36
        ax=patch//2;b=(ax+1)%3;c=(ax+2)%3;sign=1 if patch%2==0 else -1
        def inverse(X,j):
            d=sign*X[ax];u=X[b]/d;v=X[c]/d
            if j==1:return u
            if j==2:return v
            t=(d-L)/(R/mp.sqrt(1+u*u+v*v)-L)
            return t if k==0 else -mp.log(1-(1-mp.exp(-k))*t)/k
        oracle=[]
        for j in range(3):
            for i in range(3):
                order=[0]*3;order[i]=1
                oracle.append(mp.diff(lambda *X:inverse(X,j),x,tuple(order)))
        for j in range(3):
            for i in range(3):
                for l in range(3):
                    order=[0]*3;order[i]+=1;order[l]+=1
                    oracle.append(mp.diff(lambda *X:inverse(X,j),x,tuple(order)))
        for j,(value,expected) in enumerate(zip(native,oracle)):
            compare('map',f'patch={patch},stretch={float(k)},q={list(map(float,q))},component={j}',value,expected)
        # Hessian chain rule requires the H term even with global Cartesian fields.
        J=mp.matrix(3,3,native[:9]);J=mp.matrix([[native[3*i+j] for j in range(3)] for i in range(3)])
        Ji=J**-1;v=mp.matrix([mp.mpf('.2'),mp.mpf('.3'),mp.mpf('-.1')]);f=mp.exp(sum(v[i]*x[i] for i in range(3)))
        fq=Ji.T*v*f
        omitted=max(abs(sum(M(native[9+9*j+3*i+l])*fq[j] for j in range(3))) for i in range(3) for l in range(3))
        negative.append(float(omitted))
report=dict(protocol=protocol,passed=all(r['passed'] for r in records) and max(negative)>1e-3,
    counts={kind:sum(r['kind']==kind for r in records) for kind in ['layer','pullback','map']},
    maximum_scaled_error=max(r['scaled_error'] for r in records),
    worst=sorted(records,key=lambda r:r['scaled_error'],reverse=True)[:20],
    failures=[r for r in records if not r['passed']],raw_stationary_rhs=raw,
    exact_scri_endpoint_admitted=False,exact_scri_interior_rejection=endpoint,
    omitted_H_negative_control_max=max(negative),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),mpmath_version=mp.__version__)
(a.output/'geometry-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(a.output/'geometry-comparisons.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['passed','counts','maximum_scaled_error','omitted_H_negative_control_max','failures']},indent=2))
