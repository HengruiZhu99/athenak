"""Saved JSON/byte-pin readback only; stdlib, no oracle import or query."""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import math
import time

P=Path(__file__).resolve().parent
A=P/'values-attempt001'
I=P/'values-invocation001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(path):
    def reject(value):raise ValueError('nonfinite JSON token '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite float')
    if isinstance(value,list):
        for item in value:finite(item)
    if isinstance(value,dict):
        for item in value.values():finite(item)


def main():
    output=P/'saved-readback001.json'
    if output.exists():raise RuntimeError('fresh readback output required')
    begin=time.monotonic()
    child=read(A/'receipt.json');outer=read(I/'receipt.json');checks=read(A/'checks.json')
    assert sha(A/'receipt.json')=='b054132e54cbe1e63b217f2cb1d416ea87e955f5ef1dfa2c945936e127730493'
    assert sha(I/'receipt.json')=='f6e5ee874b40543841c520f398a12c5e9f4609af29520848d72199c576e2f267'
    assert child['passed_scalar_values_only'] and child['sources_unchanged']
    assert outer['passed_outer_process'] and outer['returncode']==0 and outer['sources_unchanged']
    assert len(checks)==282 and child['checks']==282
    assert not child['failed_checks']
    assert (I/'stderr.log').read_bytes()==b''
    for receipt in [child,outer]:
        assert receipt['source_before']==receipt['source_after']
        for path,expected in receipt['source_before'].items():assert sha(path)==expected,path
    for path,expected in child['output_pins'].items():assert sha(path)==expected,path
    groups={}
    for row in checks:
        error,tol=Decimal(row['error']),Decimal(row['tolerance'])
        assert error.is_finite() and tol.is_finite() and error>=0 and tol>=0
        assert row['passed'] and error<=tol
        parts=row['name'].split('/')
        group='/'.join(parts[:2]) if parts[0] in ['coarea_convergence','precision','ray_convergence','ray_coarea','zero_pulse'] else parts[0]
        if group not in groups or error>Decimal(groups[group]['maximum_scaled_error']):
            groups[group]={'maximum_scaled_error':row['error'],'case':row['name'],'tolerance':row['tolerance']}
    arrays={name:read(A/(name+'.json')) for name in ['values','initial-data','controls','height','rays']}
    assert [len(arrays[name]) for name in ['values','initial-data','controls','rays']]==[96,56,72,18]
    for rows in arrays.values():finite(rows)
    displayed=[]
    for row in arrays['values']:
        for field in ['u','phi']:
            assert len(row[field])==4
            assert all(Decimal(v).is_finite() for v in row[field])
        if row['dps']==110 and row['level']=='full128':
            displayed.append({'name':row['name'],'u':row['u'],'phi':row['phi'],'interpretation':row['interpretation']})
    data={'kind':'Saved-only exact-flat integral result/provenance readback','passed_saved_readback':True,'readback_source_sha256':sha(__file__),'child_receipt_sha256':sha(A/'receipt.json'),'outer_receipt_sha256':sha(I/'receipt.json'),'checks_sha256':sha(A/'checks.json'),'child_pins':len(child['source_before']),'outer_pins':len(outer['source_before']),'checks':len(checks),'rows':{name:len(rows) for name,rows in arrays.items()},'groups':groups,'full128_110digit_values':displayed,'seconds':time.monotonic()-begin,'new_oracle_queries':0,'new_arithmetic_scope':'Decimal comparisons of saved errors/tolerances and byte hashes only','scope':'Finite reference-event scalar values/initial data only; no derivative, inverse, native-target coverage, global caustic or native acceptance claim.'}
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(sha(output),flush=True)


if __name__=='__main__':main()
