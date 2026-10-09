#!/usr/bin/env python3
"""HELD exact finite-matrix certificate; actual matrices require separate release."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,math,subprocess,sys,time
import dyadic_certificate as dc
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def solve_upper(lower,Q):
    """Approximate modal eigenvector proposal V=L^-T Q, ordinary binary64 only."""
    n=len(lower);V=[[0j]*n for _ in range(n)]
    for column in range(n):
        for i in range(n-1,-1,-1):
            numerator=Q[i][column]-sum(complex(lower[k][i])*V[k][column] for k in range(i+1,n))
            V[i][column]=numerator/float(lower[i][i])
    return V

def approximate_inverse(V):
    """Ordinary complex binary64 Gauss-Jordan proposal; NOT trusted as an inverse."""
    n=len(V);a=[[complex(v) for v in row]+[complex(i==j) for j in range(n)] for i,row in enumerate(V)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda k:abs(a[k][i]))
        if a[pivot][i]==0:raise RuntimeError('approximate inverse proposal has zero pivot')
        a[i],a[pivot]=a[pivot],a[i];value=a[i][i]
        a[i]=[v/value for v in a[i]]
        for k in range(n):
            if k==i:continue
            multiplier=a[k][i];a[k]=[x-multiplier*y for x,y in zip(a[k],a[i])]
    return [row[n:] for row in a]

def encode_complex(rows):
    out=[]
    for row in rows:
        current=[]
        for z in row:
            z=complex(z)
            if not (math.isfinite(z.real) and math.isfinite(z.imag)):raise RuntimeError('nonfinite binary64 proposal')
            current.append([z.real.hex(),z.imag.hex()])
        out.append(current)
    return out

def execute(args):
    # NumPy is used only to decode the preserved NPZ; certificate uses no BLAS.
    import numpy as np
    if (sys.float_info.radix,sys.float_info.mant_dig)!=(2,53):raise RuntimeError('binary64 input/preparation platform required')
    auth=json.loads(args.authorization.read_text())
    if auth.get('finite_matrix_exact_certificate_admitted') is not True or auth.get('N')!=args.N:
        raise RuntimeError('HELD actual certificate lacks exact admission')
    for key,path in [('driver_sha256',Path(__file__)),('core_sha256',HERE/'dyadic_certificate.py'),('plan_sha256',HERE/'PLAN.md')]:
        if auth.get(key)!=sha(path):raise RuntimeError('authorization pin mismatch '+key)
    named=[('operator_path','operator_sha256'),('payload_path','payload_sha256'),
           ('growth_receipt_path','growth_receipt_sha256'),('synthetic_receipt_path','synthetic_receipt_sha256')]
    pins={str(Path(auth[p]).resolve()):auth[h] for p,h in named}
    for path,value in pins.items():
        if sha(path)!=value:raise RuntimeError('input pin mismatch '+path)
    paths=list(map(Path,pins))+[Path(__file__),HERE/'dyadic_certificate.py',HERE/'PLAN.md',args.authorization]
    before={str(p.resolve()):sha(p) for p in paths};args.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':[sys.executable,*sys.argv],'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'source_before':before,'error':None,'N':args.N,'J':0,'rb':.98,'certificate_computation_completed':False,
      'original_SciPy_expm_attempt_remains_failed':True,'both_ordinary_FD_attempts_remain_failed':True,
      'general_nongauge_continuum_comparator_unresolved':True,
      'scope':'Exact eigenvalue enclosure of the supplied rounded finite matrix only; no continuum/subsidiary/native stability theorem.'}
    write(args.output/'launch.json',receipt);begin=time.monotonic()
    try:
        synthetic=json.loads(Path(auth['synthetic_receipt_path']).read_text())
        assert synthetic['passed_synthetic_exact_certificate_tests'] is True and synthetic['no_actual_matrix_access'] is True
        assert synthetic['inputs_before'][str((HERE/'dyadic_certificate.py').resolve())]==sha(HERE/'dyadic_certificate.py')
        growth=json.loads(Path(auth['growth_receipt_path']).read_text());n=8*args.N
        assert (growth['J'],growth['N'],growth['rb'])==(0,args.N,.98)
        assert growth['passed_finite_ODE_numerical_checks'] is True and growth['error'] is None
        assert growth['payload_sha256']==auth['payload_sha256']
        assert growth['inputs_before'][str(Path(auth['operator_path']).resolve())]==auth['operator_sha256']
        with np.load(auth['operator_path'],allow_pickle=False) as data:
            bulk=data['Jbulk'].copy();sat=data['Jsat'].copy();lower=data['energy_cholesky'].copy()
        with np.load(auth['payload_path'],allow_pickle=False) as data:
            Q=data['energy_eigenvectors'].copy();values=data['eigenvalues'].copy()
        assert all(a.shape==(n,n) and np.isfinite(a).all() for a in (bulk,sat,lower,Q))
        assert values.shape==(n,) and np.isfinite(values).all()
        assert all(a.dtype.kind=='f' for a in (bulk,sat,lower)) and Q.dtype.kind=='c' and values.dtype.kind=='c'
        assert np.array_equal(lower,np.tril(lower)) and np.all(np.diag(lower)>0)
        # Target is elementwise RN-even binary64(Jbulk+Jsat), matching the saved
        # analyzer's binary64 addition. It is NOT their unrounded rational sum.
        J=[[dc.nearest_binary64(Fraction.from_float(float(bulk[i,j]))+Fraction.from_float(float(sat[i,j]))) for j in range(n)] for i in range(n)]
        assert all(J[i][j]==float(bulk[i,j])+float(sat[i,j]) for i in range(n) for j in range(n))
        receipt['rounded_sum_binding_checked']=True
        V=solve_upper(lower.tolist(),Q.tolist());W=approximate_inverse(V);centers=[complex(v) for v in values]
        exact_input={'schema':'Each complex binary64 entry is a pair of float.hex strings; exact dyadic values.',
          'target':'Elementwise exact RN-even binary64(Jbulk+Jsat), then treated as exact matrix.',
          'J':encode_complex(J),'V':encode_complex(V),'W':encode_complex(W),'saved_lambda':encode_complex([centers])[0]}
        write(args.output/'exact-binary64-input.json',exact_input)
        certificate=dc.certify(J,V,W,centers);write(args.output/'certificate.json',certificate)
        receipt.update({'certificate_computation_completed':True,
          'certified_positive_eigenvalues_at_least':certificate['certified_positive_eigenvalues_at_least'],
          'invertibility_proved':certificate['invertibility_proved'],'certificate_status':certificate['status'],
          'outputs':[{'path':'exact-binary64-input.json','sha256':sha(args.output/'exact-binary64-input.json'),'role':'large_payload'},
                     {'path':'certificate.json','sha256':sha(args.output/'certificate.json'),'role':'source_or_receipt'}]})
    except Exception as exc:receipt['error']={'type':type(exc).__name__,'message':str(exc)}
    receipt['source_after']={str(p.resolve()):sha(p) for p in paths};receipt['sources_unchanged']=before==receipt['source_after']
    receipt['seconds']=time.monotonic()-begin
    write(args.output/'receipt.json',receipt);print(json.dumps(receipt,indent=2,allow_nan=False))
    # A completed inconclusive certificate is recorded, without a positive pass.
    return 0 if receipt['certificate_computation_completed'] and receipt['sources_unchanged'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--N',type=int,choices=(8,12,16))
    p.add_argument('--authorization',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if not a.execute:print('HELD exact certificate; actual matrices require separate exact authorization.');return 0
    if a.N is None or a.authorization is None or a.output is None:p.error('--N, --authorization and --output required')
    return execute(a)
if __name__=='__main__':sys.exit(main())
