#!/usr/bin/env python3
"""Freeze completed projection evidence without executing any science."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
O=Path(__file__).resolve().parent;REPO=O.parents[2]
DEST=O/'immutable-J0-projection-constraint-defect-20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extra-review',type=Path,action='append',default=[])
    args=parser.parse_args()
    if DEST.exists():raise RuntimeError('immutable destination already exists')
    exclude={'freeze.stdout','freeze.stderr','frozen-readback.stdout','frozen-readback.stderr',
             'freeze-v2.stdout','freeze-v2.stderr','frozen-readback-v2.stdout','frozen-readback-v2.stderr'}
    files=[p for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not any(v.startswith('immutable-') for v in p.relative_to(O).parts)
        and not (p.parent==O and p.name in exclude)]
    contexts=set();historical={}
    receipts=[O/'attempt-1791558113840144000/receipt.json',O/'refined-h0-half/attempt-1791558482849376000/receipt.json',
        O/'exact-projected/attempt-1791559237496324000/receipt.json']
    for p in receipts:
        r=json.loads(p.read_text());assert r['sources_unchanged'] and r['source_before']==r['source_after']
        for name,value in r['source_before'].items():
            q=Path(name)
            if sha(q)!=value:
                # This exact versioned prose file changed after the run. No working-tree edit.
                if q!=REPO/'docs/hyperboloidal-continuum-constraint-rate-audit.md':
                    raise RuntimeError('unexpected input drift: '+str(q))
                ref='9bc9fc71b057bc74d1ead0c3b34119390591c1bc:'+str(q.relative_to(REPO))
                data=subprocess.check_output(['git','show',ref])
                if hashlib.sha256(data).hexdigest()!=value:raise RuntimeError('historic blob mismatch')
                historical[q]=(data,ref,value)
            if not q.is_relative_to(O):contexts.add(q)
    runtime=REPO/'build-layer-research/boundary/total-j-finite-rb-control-20261009/python-runtime-environment.json'
    if runtime.exists():contexts.add(runtime)
    contexts.update(p.resolve() for p in args.extra_review)
    DEST.mkdir();entries=[]
    for p in sorted(files)+sorted(contexts):
        relative=p.relative_to(O) if p.is_relative_to(O) else Path('context')/p.relative_to(REPO)
        target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if p in historical:
            data,ref,value=historical[p];target.write_bytes(data);assert sha(target)==value
            origin='git:'+ref
        else:
            shutil.copy2(p,target);assert sha(target)==sha(p);origin=str(p.resolve())
        large=(('calls' in relative.parts and p.suffix in ('.input','.stdout') and p.stat().st_size>0)
            or 'raw22-point-maps' in p.name or p.name in ('operator.npz','radial-bridge-release'))
        entries.append({'path':str(relative),'bytes':target.stat().st_size,'sha256':sha(target),
            'role':'large_payload' if large else 'source_or_receipt','origin':origin})
    index={'scope':'Completed J0/N8 finite-rb projection evidence: two failed FD attempts and distinct analytic projected-point PASS; general nongauge continuum comparator unresolved.',
      'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'launch_HEAD':'9bc9fc71b057bc74d1ead0c3b34119390591c1bc',
      'public_runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
      'historic_context_recovery':{str(p):{'git_blob':ref,'sha256':value} for p,(_,ref,value) in historical.items()},
      'files':entries,'total_files':len(entries),'total_bytes':sum(x['bytes'] for x in entries),
      'large_payload_files':sum(x['role']=='large_payload' for x in entries),
      'analytic_point_gate_passed':True,'both_FD_attempts_remain_failed':True,
      'general_nongauge_continuum_comparator_unresolved':True,'new_compilation_spectrum_propagation':False}
    (DEST/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'destination':str(DEST),'index_sha256':sha(DEST/'index.json'),
      **{k:index[k] for k in ('total_files','total_bytes','large_payload_files')}},indent=2))
if __name__=='__main__':main()
