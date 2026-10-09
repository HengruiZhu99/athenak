"""Freeze actual lower-order lapse gate before any separate global/native work."""
from pathlib import Path
import hashlib
import json
import shutil


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'immutable-inner-lapse-advection-local-20261009'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    receipt=json.loads((HERE/'receipt.json').read_text())
    assert receipt['passed_lower_order_lapse_local_gates']
    assert not receipt['native_global_or_scri_stability_accepted']
    assert len(receipt['commands'])==10 and len(receipt['source_before'])==376
    assert receipt['source_before']==receipt['source_after'] and receipt['sources_unchanged']
    assert all(sha(ROOT/name)==digest for name,digest in receipt['source_after'].items())
    for command in receipt['commands']:
        assert command['returncode']==0 and not command['stderr']
        if 'stdout_file' in command:
            assert sha(HERE/command['stdout_file'])==command['stdout_sha256']
    for name,digest in receipt['binary_sha256'].items():assert sha(HERE/name)==digest
    check=json.loads((HERE/'check-report.json').read_text());assert check['status']=='PASS'
    assert check['actual_full20_rows']==1900 and check['negative_root_RK3_cases']==132
    sources=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.hpp','.cpp','.json','.log','.txt') and p.name!='full20.json']
    sources += [p for p in (HERE/'failed-checker-symbol-shadowing').rglob('*') if p.is_file()]
    OUT.mkdir()
    files={}
    for source in sorted(sources):
        name=str(source.relative_to(HERE));target=OUT/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert target.read_bytes()==source.read_bytes()
        files[name]={'sha256':sha(target),'bytes':target.stat().st_size,
                     'source':str(source.relative_to(ROOT))}
    large=[HERE/name for name in ('full20.json','full20','nonlinear','nonlinear_debug','principal')]
    index={'immutable':True,'scope':'Physical-P regular lapse source only; C0 kappa2=0 spatial-norm baseline; finite-Omega local admission',
           'native_global_or_scri_stability_accepted':False,'files':files,
           'large_files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in large},
           'receipt_sha256':sha(HERE/'receipt.json'),'helper_sha256':sha(HERE/'inner_lapse_advection.hpp'),
           'geometry_layer_radii':[.05,.95],'gauge_layer_radii':[.45,.85]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    for name,row in files.items():assert sha(OUT/name)==row['sha256']
    print('Frozen',len(files),'files',sum(row['bytes'] for row in files.values()),'bytes',len(large),'large hashes',sha(OUT/'index.json'))


if __name__=='__main__':main()
