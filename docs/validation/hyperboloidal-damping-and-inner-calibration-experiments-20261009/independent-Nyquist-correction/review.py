"""Read-only independent correction of native versus global Nyquist labels."""
from pathlib import Path
import hashlib
import json
import numpy as np


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
GATE=ROOT/'build-layer-research/continuum/damping-profile-control/immutable-C0-profile-local-v3-20261009'
INDEX='ee1bb51aa40cf7829e8f0262e4f8c6230458a1480dbecbce46071a037d757e84'
OLD_REVIEW='1be97e8f5b9bda440e38acf3554495ce928917d81bba0a0026adc9fd223d9ab4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(GATE/'index.json')==INDEX
    index=json.loads((GATE/'index.json').read_text())
    for name,digest in index['files'].items():assert sha(GATE/name)==digest
    for name,row in index['large_outputs_outside_snapshot'].items():
        p=GATE.parent/name
        assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    assert sha(GATE/'receipt.json')=='d832017b2091f9f9e618fd9f4b6cfdbce0afcb19972ad43eb8217bd4681bdbaa'
    assert sha(GATE/'damping_profile.hpp')=='64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532'
    original=ROOT/'build-layer-research/continuum/independent-profile-review/frozen-index.json'
    assert sha(original)==OLD_REVIEW
    receipt=json.loads((GATE/'native-Nyquist-receipt.json').read_text())
    assert receipt['passed_actual_native_span_Nyquist_supplement']
    assert not receipt['global_native_or_scri_stability_accepted']
    assert receipt['source_before']==receipt['source_after'] and receipt['sources_unchanged']
    assert len(receipt['commands'])==2 and all(x['returncode']==0 and not x['stderr'] for x in receipt['commands'])
    assert all(sha(ROOT/name)==digest for name,digest in receipt['source_after'].items())
    assert sha(ROOT/receipt['authoritative_input'])==receipt['authoritative_input_sha256']
    settings={};section=''
    for line in (ROOT/receipt['authoritative_input']).read_text().splitlines():
        line=line.split('#')[0].strip()
        if line.startswith('<'):section=line.strip('<>')
        elif '=' in line:
            k,v=map(str.strip,line.split('=',1));settings[section+'/'+k]=v
    span=[float(settings[f'mesh/x{i}max'])-float(settings[f'mesh/x{i}min']) for i in (1,2,3)]
    assert span==[2.1]*3
    rows=json.loads((GATE/'native-Nyquist.json').read_text());assert len(rows)==12
    observed=[]
    for row in rows:
        n=row['N'];h=span[0]/n
        axes=[float(settings[f'mesh/x{i}min'])+(np.arange(n)+.5)*h for i in (1,2,3)]
        r2=axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2
        minimum=float((1-r2[r2<1]).min())
        assert abs(minimum-row['Omega'])<2e-14
        assert abs(row['k']-np.pi/h)<1e-13
        assert abs(row['dt']-.03*row['Omega'])<1e-14
        value=np.asarray(row['L']);L=value[:,:,0]+1j*value[:,:,1]
        assert np.isfinite(L).all()
        weights=np.array([1.]*12+[1/row['k']]*8)
        eig=np.linalg.eigvals(weights[:,None]*L/weights[None,:])
        z=row['dt']*eig[eig.real<=0]
        excess=max(0.,float(abs(1+z+z*z/2+z*z*z/6).max())-1)
        assert excess<1e-12
        observed.append({k:row[k] for k in ('N','span','k','Omega','dt','profile','oblique')}|
                        {'recomputed_grid_Omega_min':minimum,'nonpositive_root_RK3_excess':excess,
                         'positive_primitive_roots_retained':int((eig.real>0).sum())})
    out={'status':'PASS','gate_v3_index_sha256':INDEX,'supplement_receipt_sha256':sha(GATE/'native-Nyquist-receipt.json'),
         'unchanged_original_review_index_sha256':OLD_REVIEW,'source_paths_verified':len(receipt['source_after']),
         'commands_exit0':2,'rows':observed,
         'correction':'Earlier frozen review inherited v2 incorrect actual-native-Nyquist label for pi*N/2.2. Those numbers are valid global-span2.2 Fourier samples. Native span2.1 uses pi*N/2.1, independently checked here. Earlier frozen bytes remain unchanged.',
         'helper_equations_builds_and_original_numeric_receipt_unchanged':True,
         'no_native_global_scri_or_contractivity_acceptance':True,'source_sha256':sha(Path(__file__))}
    (HERE/'receipt.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
