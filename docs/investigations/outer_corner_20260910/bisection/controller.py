#!/usr/bin/env python3
"""Fresh N256 t=200 lapse bisection; one shared_interactive allocation per case."""
from pathlib import Path
from decimal import Decimal,getcontext
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,time,traceback
from criterion import classify,read_history,relative_width
getcontext().prec=40
R=Path(__file__).resolve().parent

def save(p,s):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def setparam(text,section,key,value):
    pat=r'(?ms)(^<'+re.escape(section)+r'>\s*\n)(.*?)(?=^<|\Z)'
    matches=list(re.finditer(pat,text))
    if len(matches)!=1:raise ValueError('Missing/duplicate section '+section)
    m=matches[0];line=r'(?m)^\s*'+re.escape(key)+r'\s*=.*$'
    if len(re.findall(line,m[2]))>1:raise ValueError('Duplicate key '+key)
    body=re.sub(line,key+' = '+str(value),m[2]) if re.search(line,m[2]) else m[2]+key+' = '+str(value)+'\n'
    return text[:m.start()]+m[1]+body+text[m.end():]

# Configuration under qualification; do not launch before endpoint evidence is reviewed.
def campaign_input(template,case):
    inp=template
    for section,key,value in [
        ('job','basename','lapse200'),('time','tlim',200),
        ('fastflow','num_horizons',0),('problem','stop_on_horizon','false'),
        ('problem','stop_on_dispersion','false'),
        ('mesh_refinement','amr_history_file',case/'amr_history.jsonl'),
        ('problem','brill_global_coefficients_file','initial.coefficients'),
        ('problem','constraint_summary_file','initial-constraints.dat'),
        ('z4c','boundary_rhs','full_constraint_bjorhus'),('z4c','extrap_order',2),
        ('z4c','vc_single_rank_device_sync','true'),
        ('z4c','history_constraint_radius',1000),('output5','dt',5)]:
        inp=setparam(inp,section,key,value)
    return inp

def input_parameters(text):
    result={};section=None
    for raw in text.splitlines():
        line=raw.split('#',1)[0].strip()
        if line.startswith('<') and line.endswith('>'):section=line[1:-1]
        elif '=' in line:
            key,value=line.split('=',1);key=(section,key.strip())
            if key in result:raise ValueError('Duplicate input parameter '+str(key))
            result[key]=value.strip()
    return result

def validate_adoption(case,amplitude,template,executable_hash):
    if (case/'run-status').read_text().strip()!='0':raise RuntimeError('Adopted evolution is incomplete or failed')
    if Decimal((case/'amplitude.txt').read_text())!=amplitude:raise RuntimeError('Adopted amplitude mismatch')
    provenance=json.loads((case/'provenance.json').read_text())
    if provenance.get('exe_sha256')!=executable_hash:raise RuntimeError('Adopted executable mismatch')
    if provenance.get('input_sha256')!=sha(case/'input.athinput'):raise RuntimeError('Adopted input changed')
    expected=input_parameters(campaign_input(template,case))
    actual=input_parameters((case/'input.athinput').read_text())
    # Basename only names output files; all physical, AMR and diagnostic settings match.
    expected.pop(('job','basename'));actual.pop(('job','basename'))
    history_key=('mesh_refinement','amr_history_file')
    for params in [expected,actual]:params[history_key]=str((case/params[history_key]).resolve())
    if expected!=actual:raise RuntimeError('Adopted run settings differ from campaign')
    checks={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0]
            for line in (case/'inputs.sha256').read_text().splitlines()}
    for name in ['input.athinput','initial.coefficients']:
        if checks.get(name)!=sha(case/name):raise RuntimeError('Adopted input checksum mismatch: '+name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--adopt-super',type=Path);a=ap.parse_args();base=a.baseline.resolve(strict=True)
    with (R/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (R/'state.json').exists():raise RuntimeError('Existing state; explicit recovery required')
        template=(base/'template.athinput').read_text()
        qualification=R/'BOUNDARY_QUALIFIED.json'
        if not qualification.exists():raise RuntimeError('Boundary qualification gate missing; do not launch')
        q=json.loads(qualification.read_text())
        if q.get('executable_sha256')!=sha(base/'athena.history_extrema') or q.get('approved_configuration')!={'boundary_rhs':'full_constraint_bjorhus','extrap_order':2,'vc_single_rank_device_sync':True}:
            raise RuntimeError('Qualification does not match executable/boundary configuration')
        if not re.search(r'amr_history_mode\s*=\s*record',template):raise RuntimeError('Expected live AMR')
        runner=(base/'run_cycle.sh').read_text().replace('campaign=$(dirname -- "$case_dir")','campaign='+str(base))
        (R/'run_cycle.sh').write_text(runner)
        state=dict(status='STARTING',pid=os.getpid(),host=os.uname().nodename,created=time.time(),target_time=200,criterion='global minLapse(t=200) < 0.01 => collapse; otherwise disperse',relative_tolerance='0.00001',baseline=str(base),executable_sha256=sha(base/'athena.history_extrema'),source_base=(base/'source-base.txt').read_text().strip(),source_patch_sha256=sha(base/'source.patch'),template_sha256=sha(base/'template.athinput'),boundary_qualification=q,boundary_qualification_sha256=sha(qualification),completed=[],active=None)
        save(R/'state.json',state)
        def run_case(amplitude,name):
            if (R/'STOP').exists():raise RuntimeError('STOP requested before submission')
            if sha(base/'athena.history_extrema')!=state['executable_sha256']:raise RuntimeError('Executable changed')
            adopted = name=='endpoint_super' and a.adopt_super is not None
            if adopted:
                case=a.adopt_super.resolve(strict=True)
                validate_adoption(case,amplitude,template,state['executable_sha256'])
                rc=0
            else:
                case=R/name;case.mkdir();(case/'amplitude.txt').write_text(str(amplitude)+'\n')
                inp=campaign_input(template,case)
                (case/'input.athinput').write_text(inp)
                cmd=['salloc','--account=m3328_g','--qos=shared_interactive','--constraint=gpu&hbm80g','--nodes=1','--ntasks=1','--cpus-per-task=32','--gpus=1','--time=04:00:00','--job-name=lapse200-'+name,'bash',str(R/'run_cycle.sh'),str(case)]
                save(case/'allocation-command.json',cmd);state.update(status='ALLOCATING',active={'amplitude':str(amplitude),'directory':str(case)});save(R/'state.json',state)
                with (case/'allocation.log').open('w') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    state['active']['salloc_pid']=proc.pid;save(R/'state.json',state)
                    rc=proc.wait()
            if rc or not (case/'run-status').exists() or (case/'run-status').read_text().strip()!='0':raise RuntimeError('Allocation/evolution failure; no bracket update: '+name)
            h=read_history(next(case.glob('*.hst')));result=classify(h,200)
            if adopted and abs(h[0]['time'])>1e-10:raise RuntimeError('Adopted endpoint must be a fresh evolution from t0')
            if 'Terminating on time limit' not in (case/'stdout.log').read_text():raise RuntimeError('Unexpected termination')
            if list(case.glob('*.termination.json')):raise RuntimeError('Premature physical/resource stop')
            checkpoints=sorted((case/'rst').glob('*.rst'))
            if not checkpoints or checkpoints[-1].stat().st_size<1024:raise RuntimeError('Missing final checkpoint')
            record=dict(amplitude=str(amplitude),classification=result,final=h[-1],directory=str(case),job_id=(case/'job-id.txt').read_text().strip(),checkpoint=str(checkpoints[-1]),checkpoint_sha256=sha(checkpoints[-1]),coefficient_sha256=sha(case/'initial.coefficients'),input_sha256=sha(case/'input.athinput'),finished=time.time(),adopted=adopted)
            save((R/'adopted-super-result.json') if adopted else (case/'result.json'),record);state['completed'].append(record);state.update(status='CLASSIFIED',active=None);save(R/'state.json',state)
            subprocess.run([sys.executable,str(R/'plot.py')],check=True)
            return result
        try:
            sub,sup=Decimal('-.047'),Decimal('-.05')
            for amp,name,expected in [(sub,'endpoint_sub','disperse'),(sup,'endpoint_super','collapse')]:
                result=run_case(amp,name)
                if result!=expected:raise RuntimeError('Endpoint no longer brackets under t=200 criterion; reassess range before bisection')
            state.update(sub=str(sub),super=str(sup));save(R/'state.json',state)
            for i in range(1,25):
                width=relative_width(sub,sup);state['relative_width']=str(width)
                if width<=Decimal('0.00001'):
                    state.update(status='COMPLETE',active=None,finished=time.time());save(R/'state.json',state);subprocess.run([sys.executable,str(R/'plot.py')],check=True);return
                amp=(sub+sup)/2
                result=run_case(amp,f'cycle_{i:02d}')
                if result=='collapse':sup=amp
                else:sub=amp
                state.update(sub=str(sub),super=str(sup),relative_width=str(relative_width(sub,sup)));save(R/'state.json',state)
            raise RuntimeError('Iteration limit')
        except BaseException as e:
            state.update(status='FAILED',error=repr(e),failed=time.time());save(R/'state.json',state);raise
if __name__=='__main__':main()
