"""Independent exact saved-data readback, with alternate product association."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build-layer-research/continuum/finite-matrix-exact-certificate-20261009'
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def read(p): return json.loads(p.read_text())
def rational(d):
    v=F(int(d['numerator']),int(d['denominator']))
    lo,hi=[F.from_float(float.fromhex(x)) for x in d['outward_binary64_hex']]
    assert lo<=v<=hi
    return v
def integers(encoded):
    rows=[[(F.from_float(float.fromhex(a)),F.from_float(float.fromhex(b))) for a,b in row] for row in encoded]
    den=max(q.denominator for row in rows for z in row for q in z)
    assert den&(den-1)==0
    return ([[int(a*den) for a,b in row] for row in rows],[[int(b*den) for a,b in row] for row in rows],den)
def multiply(a,b):
    ar,ai,ad=a;br,bi,bd=b;bt=list(zip(*br));it=list(zip(*bi));rr=[];ii=[]
    for x,y in zip(ar,ai):
        rr.append([sum(v*w for v,w in zip(x,z))-sum(v*w for v,w in zip(y,q)) for z,q in zip(bt,it)])
        ii.append([sum(v*w for v,w in zip(x,q))+sum(v*w for v,w in zip(y,z)) for z,q in zip(bt,it)])
    return rr,ii,ad*bd

begin=time.monotonic();results=[];pins={str(Path(__file__)):sha(Path(__file__))}
for n in (8,12,16):
    p=BASE/f'N{n}-certificate001';paths=[p/'exact-binary64-input.json',p/'certificate.json',p/'receipt.json']
    pins.update({str(q):sha(q) for q in paths});d=read(paths[0]);c=read(paths[1]);receipt=read(paths[2])
    assert receipt['error'] is None and receipt['sources_unchanged'] and receipt['rounded_sum_binding_checked']
    J,V,W=[integers(d[k]) for k in ('J','V','W')];D=integers([d['saved_lambda']]);count=8*n
    N=multiply(W,V)
    # Alternative association to the producer: (W*J)*V, not W*(J*V).
    M=multiply(multiply(W,J),V)
    nu=F(max(sum(abs(N[0][i][j]-(N[2] if i==j else 0))+abs(N[1][i][j]) for j in range(count)) for i in range(count)),N[2])
    assert nu==rational(c['nu_upper'])<1
    nd=N[2]*D[2];common=max(M[2],nd);assert common%M[2]==common%nd==0
    scaleM=common//M[2];scaleND=common//nd;rowbounds=[]
    for i in range(count):
        total=0
        for j in range(count):
            a,b=N[0][i][j],N[1][i][j];x,y=D[0][0][j],D[1][0][j]
            total+=abs(scaleM*M[0][i][j]-scaleND*(a*x-b*y))
            total+=abs(scaleM*M[1][i][j]-scaleND*(a*y+b*x))
        rowbounds.append(total)
    rho=F(max(rowbounds),common);eps=rho/(1-nu)
    assert rho==rational(c['residual_upper']) and eps==rational(c['common_radius_upper'])
    centers=[(F(x,D[2]),F(y,D[2])) for x,y in zip(D[0][0],D[1][0])]
    groups=[g['center_indices'] for g in c['clusters']];assert sorted(j for g in groups for j in g)==list(range(count))
    pos=neg=0
    for g,record in zip(groups,c['clusters']):
        lower=min(centers[j][0] for j in g)-eps;upper=max(centers[j][0] for j in g)+eps
        assert lower==rational(record['real_lower']) and upper==rational(record['real_upper'])
        assert record['entire_cluster_strictly_positive']==(lower>0)
        assert record['entire_cluster_strictly_negative']==(upper<0)
        assert record['eigenvalues_counted_with_algebraic_multiplicity']==len(g)
        for i in g:
            for j in set(range(count))-set(g):
                assert (centers[i][0]-centers[j][0])**2+(centers[i][1]-centers[j][1])**2>4*eps**2
        if lower>0:pos+=len(g)
        if upper<0:neg+=len(g)
    assert pos==c['certified_positive_eigenvalues_at_least']==2
    assert neg==c['certified_negative_eigenvalues_at_least']==count-2
    results.append({'N':n,'nu_exact_match':True,'rho_exact_match_alternate_association':True,'epsilon_exact_match':True,'outward_bounds_verified':True,'positive_count':pos,'negative_count':neg})
assert all(sha(Path(p))==v for p,v in pins.items())
result={'status':'PASS_exact_saved_certificate_readback','sources_and_inputs':pins,'unchanged':True,'results':results,'seconds':time.monotonic()-begin,
 'scope':'Exact supplied rounded finite J only. Alternate integer products, exact rational bounds, separated clusters and half-plane counts. No eigensolve, propagation or PDE query.'}
(HERE/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
