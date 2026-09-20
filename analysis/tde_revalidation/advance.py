#!/usr/bin/env python3
"""Fail-closed, single-successor campaign controller; no scheduler polling loop."""
import argparse,fcntl,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
from validate_checkpoint import validate,read_header,check_supported,geometry
HERE=Path(__file__).resolve().parent
ROOT=Path('/lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review')
FAILURES=re.compile(r'Z4C_INVALID_STATE|DYNGRMHD_ERROR|NANS_IN_CONS|MPI_Abort|MPI_ABORT|segmentation fault|SYCL.*exception|### FATAL ERROR',re.I)

def require(ok,message):
    if not ok:raise ValueError(message)

def save(s):
    tmp=HERE/'state.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(HERE/'state.json')

def verify_package():
    m=json.loads((HERE/'launch-manifest.json').read_text())
    for name,digest in m['files'].items():
        require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,'Changed launch file: '+name)
    executable=Path(m['executable'])
    require(hashlib.sha256(executable.read_bytes()).hexdigest()==m['executable_sha256'],'Executable changed')
    require(m['source_commit']==json.loads((HERE/'state.json').read_text())['source_commit'],'State/source mismatch')
    return m

def validate_completed(run,job_record,ranks=456):
    require(job_record['state']=='F' and job_record['exit']==0,'PBS did not finish successfully')
    require((run/'exit_code.txt').read_text().strip()=='0','Application did not exit0')
    log=(run/'run.log').read_text();require(not FAILURES.search(log),'Recorded evolution/ghost/primitive failure')
    stops=re.findall(r'^Terminating on (wall clock limit|time limit)\s*$',log,re.M)
    end=re.findall(r'^time=([\deE+.-]+) cycle=(\d+)\s*$',log,re.M)
    require(len(stops)==1 and len(end)==1,'Missing/ambiguous clean final state')
    tf,cycle=float(end[0][0]),int(end[0][1]);hist={}
    for suffix in ['mhd.hst','user.hst','z4c.user.hst']:
        f=run/('trumpet_disruption.'+suffix);lines=f.read_text().splitlines()
        header=next((l for l in lines if '[1]=' in l),None);require(header,'Missing history columns')
        names=re.findall(r'\[\d+\]=([^\s]+)',header)
        rows=[[float(x)for x in line.split()]for line in lines if line.strip() and not line.startswith('#')]
        require(rows and all(len(r)==len(names) and all(math.isfinite(x)for x in r)for r in rows),'Nonfinite/incomplete history')
        require(all(b[0]>=a[0]for a,b in zip(rows,rows[1:])),'Nonmonotonic history')
        require(abs(rows[-1][0]-tf)<max(1e-7,abs(tf)*6e-7),'Final history/log mismatch')
        if 'bad-metric'in names:require(all(r[names.index('bad-metric')]==0 for r in rows),'Invalid metrics in history')
        hist[suffix]={'rows':len(rows),'last':dict(zip(names,rows[-1]))}
    # This full-payload validation inspects all ghosts, not just history reductions.
    result=validate(run,ranks,cycle=cycle)
    require(result['passed'],'Raw checkpoint fields or metric invalid')
    require(abs(result['time_M']-tf)<max(1e-7,abs(tf)*6e-7),'Final checkpoint/log mismatch')
    h=read_header(result['files'][0]['path']);p=h['params']
    require(p['problem']['outer_sponge_enabled']=='true' and float(p['z4c']['damp_kappa1'])==0 and float(p['z4c']['shift_eta'])==.02,'Unexpected checkpoint physics')
    require(float(p['time']['tlim'])==2500,'Unexpected checkpoint target')
    require(p['mesh_refinement']['refinement']=='adaptive','Expected AMR checkpoint')
    for axis in (1,2,3):
        require(float(p['mesh']['x%dmin'%axis])==-2048 and float(p['mesh']['x%dmax'%axis])==2048,'Wrong domain')
        require(int(p['meshblock']['nx%d'%axis])==32,'Wrong block size')
    require(int(p['problem']['amr_static_level'])==3 and float(p['problem']['amr_static_halfwidth'])==256,'Unexpected static refinement floor')
    ng,n,ns,lo,hi,shape,offset=check_supported(h)
    for b in geometry(h,n,ns,lo,hi):
        if all(b['min'][a]<256 and b['max'][a]>-256 for a in range(3)):
            require(b['relative_level']>=3,'Protected central region coarsened below level3')
    result.update(stopping_reason=stops[0],histories=hist,target_completed=result['time_M']>=2500-1e-7,
                  stability_clearance=False,classification='target_reached'if result['time_M']>=2500-1e-7 else'clean_walltime_stop')
    if stops[0]=='time limit':require(result['target_completed'],'Time limit below requested target')
    return result

def scheduler(job):
    r=subprocess.run(['qstat','-fx',str(job)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    require(r.returncode==0,'Cannot determine PBS state: '+r.stderr)
    state=re.search(r'job_state = (\w+)',r.stdout);require(state,'Missing PBS state')
    code=re.search(r'Exit_status = (-?\d+)',r.stdout)
    return {'state':state.group(1),'exit':int(code.group(1))if code else None}

def main(submit=False):
    s=json.loads((HERE/'state.json').read_text())
    require(s['phase']not in ['submission_uncertain','paused_failure','target_reached'],'Campaign phase requires review: '+s['phase'])
    verify_package();restart=None
    if s.get('current_job'):
        job=s['current_job'];q=scheduler(job)
        if q['state']!='F':print('Existing job %s is %s; no duplicate'%(job,q['state']));return
        run=ROOT/'runs'/('star_reval_'+job)
        try:
            result=validate_completed(run,q)
        except Exception as e:
            s.update(phase='paused_failure',failure={'job':job,'reason':str(e),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});save(s);raise
        require(result['time_M']>s.get('last_submitted_from_time_M',-1),'No forward progress')
        (HERE/('completion-'+job+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        restart=result['files'][0]['path'];s['last_verified_checkpoint']={'path':restart,'time_M':result['time_M'],'cycle':result['cycle'],'ranks':456,'blocks':result['blocks']};s['last_histories']=result['histories']
        if result['target_completed']:s['phase']='target_reached';save(s);print('2500M reached; no further submission');return
        s['phase']='ready_to_continue';save(s)
        if not submit:print(json.dumps(s['last_verified_checkpoint'],indent=2));return
    else:
        require(s['phase']=='prepared' and not s['jobs'],'Fresh launch only once')
        if not submit:print('Fresh campaign prepared, no submission requested');return
    queued=subprocess.run(['qselect','-u','hzhu','-q','debug-scaling','-s','Q'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    require(queued.returncode in(0,153),'Cannot inspect pending debug-scaling jobs')
    if queued.stdout.strip():print('Another debug-scaling job is pending; defer submission');return
    s['phase']='submission_uncertain';s['submission_attempt_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(s)
    args=['qsub']
    if restart:args+=['-v','RESTART_FILE='+restart]
    args+=['submit.pbs']
    try:r=subprocess.run(args,cwd=str(HERE),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
    except Exception:raise RuntimeError('Submission uncertain; reconcile scheduler/receipts before retrying')
    receipt={'utc':s['submission_attempt_utc'],'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    (HERE/('submission-'+str(len(s['jobs']))+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    require(r.returncode==0 and re.fullmatch(r'\d+(?:\.[\w.-]+)?',r.stdout.strip()),'Submission uncertain; reconcile before retrying')
    job=r.stdout.strip().split('.')[0];s['jobs'].append(job);s.update(current_job=job,phase='submitted',last_submitted_from_time_M=s['last_verified_checkpoint']['time_M']if restart else 0);save(s);print(r.stdout.strip())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--submit',action='store_true');ap.add_argument('--verify-package',action='store_true');a=ap.parse_args()
    with(HERE/'advance.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.verify_package:verify_package();print('Package/executable verified')
        else:main(a.submit)
