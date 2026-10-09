"""Freeze trace/combined gate, source drafts and complete failed attempts."""
from pathlib import Path
import hashlib,json,shutil

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
DEST=P/'immutable-inner-conformal-trace-local-20261009'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    receipt=json.loads((P/'receipt.json').read_text())
    assert receipt['status']=='PASS' and len(receipt['commands'])==12
    assert all(row['returncode']==0 and row['stderr']=='' for row in receipt['commands'])
    assert receipt['source_before']==receipt['source_after']
    for name,digest in receipt['source_after'].items():assert sha(ROOT/name)==digest
    assert receipt['helper_sha256']=='f2a1011eef65d2860e6d74963be98dcdf4184a65a33600475d00086c30d8922e'
    old=P.parent/'inner-lapse-advection-control/kernel_symbol_copy.cpp'
    s=(P/'kernel_symbol_copy.cpp').read_text()
    s=s.replace('p.domega[i]=.09*(i+1);','').replace('  for(int i=0;i<3;++i)omega.gradient[i]=p.domega[i];\n','')
    start=s.index('    // Nonzero prescribed Omega jets add constant lower-order geometric sources.')
    end=s.index('    hyp::GaugeRHS<double> gl{}, gs{}, gbase{};',start)
    s=s[:start]+s[end:]
    assert s==old.read_text()
    DEST.mkdir()
    binary=set(receipt['binary_sha256'])
    for source in sorted(P.rglob('*')):
        if not source.is_file() or DEST in source.parents or source.name in binary or source.name=='full20.json':continue
        target=DEST/source.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);assert sha(source)==sha(target)
    extras=[P.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',
            ROOT/'tst/hyperboloidal/kernel_symbol.cpp',ROOT/'tst/hyperboloidal/check_kernel_symbol.py',
            P.parent/'inner-lapse-advection-control/immutable-inner-lapse-advection-local-20261009/index.json']
    for source in extras:
        target=DEST/'dependencies'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    files={str(path.relative_to(DEST)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(DEST.rglob('*')) if path.is_file()}
    large={str(path.relative_to(ROOT)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in [P/'full20.json',P.parent/'inner-lapse-advection-control/full20.json']+[P/name for name in binary]}
    index={'scope':'Frozen trace-only and direct-combined inner conformal-trace finiteOmega local gates only; no native/global/stability/scri/BH admission',
      'files':files,'file_count':len(files),'bytes':sum(x['bytes'] for x in files.values()),'large_external_by_hash_only':large,
      'source_count':len(receipt['source_before']),'command_count':12,'helper_sha256':receipt['helper_sha256'],'receipt_sha256':sha(P/'receipt.json'),
      'mode_labels':{'0':'physicalP baseline','1':'rejected legacy advection control','2':'robust trace-only','3':'robust direct combined advection+trace'},
      'principal_source_diff_verified':'Only nonzero prescribedOmega-gradient fixture and geometric background subtraction differ from prior actual extractor copy.'}
    (DEST/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    print(str(DEST.relative_to(ROOT)),sha(DEST/'index.json'))
    print(len(files),'small files',index['bytes'],'bytes',len(large),'large hashes')


if __name__=='__main__':main()
