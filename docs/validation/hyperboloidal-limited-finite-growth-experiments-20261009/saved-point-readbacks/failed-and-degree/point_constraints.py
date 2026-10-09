"""Source-only shared saved-map contractions; no kernel calls or spectra."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
FROZEN=R/'continuum/finite-rb-projection-defect/immutable-J0-projection-constraint-defect-20261009'
MAP=FROZEN/'exact-projected/attempt-1791559237496324000/analytic-center-maps.npz'
RECEIPT=FROZEN/'exact-projected/attempt-1791559237496324000/receipt.json'
PINS={str(FROZEN/'index.json'):'970a117d2790f668fc3e086de7477c4ea94b683a8772c27a751f2625a623ef28',
 str(MAP):'44a6e015931a506e0d328d29f86362647e24cffa5f62d4be9ac52adbd49cface',
 str(RECEIPT):'98f18f39011601946df3b78d242a42e7ba2be3c6dfe37a174591e1d83e1ab78c'}
L=(0,0,0,0,1,1,2,2)
FIELDS=('H','Mx','My','Mz','Zx','Zy','Zz','Theta_physical')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def verify_pins(pins):
    for path,value in pins.items():
        if sha(path)!=value:raise RuntimeError('saved source/data pin changed: '+str(path))
def error(a,b):
    absolute=math.hypot(*(float(v) for v in (a-b).flat))
    return {'absolute_l2':absolute,'scaled_l2':absolute/max(1.,math.hypot(*(float(v) for v in a.flat)),math.hypot(*(float(v) for v in b.flat)))}
def complex_error(a,b):
    absolute=math.hypot(*(float(v) for v in (a-b).real.flat),*(float(v) for v in (a-b).imag.flat))
    scale=max(1.,math.hypot(*(float(v) for v in a.real.flat),*(float(v) for v in a.imag.flat)),
        math.hypot(*(float(v) for v in b.real.flat),*(float(v) for v in b.imag.flat)))
    return {'absolute_l2':absolute,'scaled_l2':absolute/scale}
def load_maps(np):
    verify_pins(PINS);receipt=json.loads(RECEIPT.read_text())
    assert receipt['passed_analytic_projected_constraint_point_gate'] is True
    assert receipt['full_fourteen_witness_projection_defect_gate_passed'] is False
    with np.load(MAP,allow_pickle=False) as data:
        points=data['points'].copy();q=data['constraints'].copy();rhs=data['rhs'].copy()
    assert points.shape==(21,3) and q.shape==(21,8,3,8) and rhs.shape==(21,8,3,22)
    assert np.isfinite(points).all() and np.isfinite(q).all() and np.isfinite(rhs).all()
    return points,q,rhs

def radial_jets(rho,N,rb,np,eval_jacobi):
    if N not in (8,12,16) or rb!=.98:raise RuntimeError('undeclared degree/radius')
    rho=np.asarray(rho);z=2*rho/(rb*rb)-1
    out=np.zeros((len(rho),8,3,N));B=rb*rb
    for c,ell in enumerate(L):
        for k in range(N):
            norm=math.sqrt(2*(2*k+ell+1.5)/B**(ell+1.5))
            out[:,c,0,k]=norm*eval_jacobi(k,0,ell+.5,z)
            if k:out[:,c,1,k]=norm*(k+ell+1.5)/B*eval_jacobi(k-1,1,ell+1.5,z)
            if k>=2:out[:,c,2,k]=norm*(k+ell+1.5)*(k+ell+2.5)/B**2*eval_jacobi(k-2,2,ell+2.5,z)
    assert np.isfinite(out).all();return out

def modal_jets(points,N,rb,np,eval_jacobi):
    rho=[math.fsum(float(v)*float(v) for v in x) for x in points]
    return radial_jets(rho,N,rb,np,eval_jacobi)

def operator_checks(data,N,rb,np,roots_jacobi,eval_jacobi):
    for key in ('nodal_from_modal','Jbulk','Jsat','E','Kweak','SATload'):
        a=data[key]
        if a.shape!=(8*N,8*N) or a.dtype.kind!='f' or not np.isfinite(a).all():
            raise RuntimeError('invalid operator array '+key)
    result={'bulk_Riesz':error(np.einsum('ij,jk->ik',data['E'],data['Jbulk'],optimize=False),data['Kweak']),
        'SAT_Riesz':error(np.einsum('ij,jk->ik',data['E'],data['Jsat'],optimize=False),data['SATload'])}
    assert all(v['scaled_l2']<=2e-9 for v in result.values())
    rho=(roots_jacobi(N,0,.5)[0]+1)*rb*rb/2;modes=radial_jets(rho,N,rb,np,eval_jacobi)
    expected=np.zeros((8*N,8*N))
    for c in range(8):expected[c*N:(c+1)*N,c*N:(c+1)*N]=modes[:,c,0,:]
    result['analytic_T']=error(data['nodal_from_modal'],expected)
    assert result['analytic_T']['scaled_l2']<=5e-11
    if 'energy_cholesky' in data:
        lower=data['energy_cholesky']
        assert lower.shape==(8*N,8*N) and np.isfinite(lower).all()
        assert np.all(np.diag(lower)>0) and np.array_equal(lower,np.tril(lower))
        result['saved_Cholesky']=error(np.einsum('ik,jk->ij',lower,lower,optimize=False),data['E'])
        assert result['saved_Cholesky']['scaled_l2']<=2e-9
    return result,rho

def contract_real(q,modes,coefficients,np):
    # Coefficient columns retain physical amplitudes/energy normalization.
    N=modes.shape[-1];assert coefficients.ndim==2 and coefficients.shape[0]==8*N
    assert np.isfinite(coefficients).all() and not np.iscomplexobj(coefficients)
    w=np.einsum('pcdk,ckv->pcdv',modes,coefficients.reshape(8,N,-1),optimize=False)
    result=np.einsum('pcdq,pcdv->pqv',q,w,optimize=False);assert np.isfinite(result).all();return result

def contract_split(q,modes,coefficients,np):
    # Linear real/imaginary maps; no complex kernel or moving-frame rotation.
    real=contract_real(q,modes,coefficients.real,np)
    imag=contract_real(q,modes,coefficients.imag,np)
    return real+1j*imag

def scalar_contract_real(q,modes,coefficients,np):
    # Independent saved-data check: scalar Jacobi/map loops, no einsum/BLAS.
    N=modes.shape[-1];out=np.zeros((len(q),8,coefficients.shape[-1]))
    for p in range(len(q)):
        for v in range(coefficients.shape[-1]):
            w=[[math.fsum(float(modes[p,c,d,k])*float(coefficients[c*N+k,v])
                for k in range(N)) for d in range(3)] for c in range(8)]
            for field in range(8):
                out[p,field,v]=math.fsum(float(q[p,c,d,field])*w[c][d]
                    for c in range(8) for d in range(3))
    assert np.isfinite(out).all();return out

def action_split(matrix,coefficients,np):
    real=np.einsum('ij,jv->iv',matrix,coefficients.real,optimize=False)
    imag=np.einsum('ij,jv->iv',matrix,coefficients.imag,optimize=False)
    return real+1j*imag

def sample_stats(values):
    # Physical constraint components, coordinate Euclidean sample summaries.
    # This is not an integrated energy or geometric covector contraction.
    out=[]
    for column in range(values.shape[-1]):
        v=values[:,:,column]
        def norm(row):return math.hypot(*(float(x) for x in row.real),*(float(x) for x in row.imag))
        out.append({'sample_l2_peak':max(norm(row) for row in v),
          'sample_l2_RMS':math.sqrt(math.fsum(norm(row)**2 for row in v)/len(v)),
          'H_peak':max(abs(complex(row[0])) for row in v),
          'M_coordinate_l2_peak':max(norm(row[1:4]) for row in v),
          'Z_coordinate_l2_peak':max(norm(row[4:7]) for row in v),
          'Theta_physical_peak':max(abs(complex(row[7])) for row in v)})
    return out
