#!/usr/bin/env python3
"""Recover only postprocessing of a successful half-step evolution."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from nonradiative_wide import sha
from nonradiative_wide_halfdt import load_extractor

def check_imports():
    extract,_=load_extractor()
    path=Path(sys.modules[extract.__module__].__file__).resolve()
    assert path==Path(__file__).resolve().parents[2]/'vis/python/scri_news.py',path
    assert 'matplotlib' not in sys.modules
    return str(path)

def recover(root,run):
    assert (run/'evolution-exit.txt').read_text().strip()=='0'
    original=int((run/'exit-code.txt').read_text());assert original!=0
    assert not (run/'analysis-recovery.json').exists()
    receipt=json.loads((run/'receipt.json').read_text());assert receipt['cycle']==3408
    checkpoint=sorted((run/'gauge-N12-halfdt/rst').glob('*.rst'))[-1]
    before=sha(checkpoint);assert before==receipt['final_sha256']
    imports=check_imports()
    # Preserve the failed analysis and its logs; rebuild only derived products.
    if (run/'analysis').exists(): (run/'analysis').rename(run/'analysis-attempt1')
    prior=run/'reference-pointwise-comparison.json'
    if prior.exists():prior.rename(run/'reference-pointwise-comparison-attempt1.json')
    driver=Path(__file__).with_name('nonradiative_wide_halfdt.py')
    command=['mpiexec','-n','2','-ppn','1',sys.executable,str(driver),'analyze',str(root),str(run)]
    result={'original_job':run.name.split('-')[-1],'recovery_job':os.environ['PBS_JOBID'].split('.')[0],
            'original_exit_status':original,'checkpoint_before_sha256':before,'command':command,
            'extractor_module':imports,'driver_sha256':sha(driver),'recovery_driver_sha256':sha(__file__),
            'snapshot_manifest_sha256':sha(root/'source-analysis-v2-sha256.json')}
    start=time.monotonic()
    with (run/'analysis-recovery.log').open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    result.update(exit_status=p.returncode,seconds=time.monotonic()-start,checkpoint_after_sha256=sha(checkpoint))
    (run/'analysis-recovery.json').write_text(json.dumps(result,indent=2)+'\n')
    assert before==result['checkpoint_after_sha256'],'Analysis changed the evolution checkpoint'
    return p.returncode

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-imports',action='store_true')
    p.add_argument('root',type=Path,nargs='?');p.add_argument('run',type=Path,nargs='?');a=p.parse_args()
    if a.check_imports:print(json.dumps({'extractor_module':check_imports()}))
    else:sys.exit(recover(a.root.resolve(),a.run.resolve()))
