#!/usr/bin/env python3
"""Aurora-only, locked serial queue controller. Inspection is the default."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

WORKFLOW=Path(__file__).resolve().parent
CAMPAIGN=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/telegrapher_lapse_20260929')
STATE_ROOT=CAMPAIGN/'single_hole_20261001'
EXE=CAMPAIGN/'source/build/aurora-sycl-v5/src/athena'
sys.path.insert(0,str(WORKFLOW.parent/'telegrapher_aurora_20260929'))
from analyze import analyze, numeric_rows


def now(): return datetime.datetime.utcnow().isoformat()+'Z'
def command(args):
    return subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
def save(state):
    path=STATE_ROOT/'state.json.tmp'
    path.write_text(json.dumps(state,indent=2)+'\n')
    path.replace(STATE_ROOT/'state.json')
def scheduler(job=None):
    if job:
        ids=[job]
    else:
        selection=command(['qselect','-u','hzhu'])
        if selection.returncode: raise RuntimeError('PBS selection failed: '+selection.stderr)
        ids=selection.stdout.split()
        if not ids: return {}
    r=command(['qstat','-x','-f','-F','json']+ids)
    if r.returncode: raise RuntimeError('PBS query failed: '+r.stderr)
    return json.loads(r.stdout).get('Jobs',{})

def audit(run,case):
    result=analyze(run)
    history=list(run.glob('*.hst'))
    rows=numeric_rows(history[0]) if history else []
    result['complete']=bool(result.get('exit_status')=='0' and rows and
        result.get('evolution_time',-1)>=case['target_time']-1e-5 and
        'Terminating on time limit' in (run/'run.log').read_text())
    result['finite']=bool(rows and all(len(r)==19 and all(math.isfinite(x) for x in r) for r in rows))
    outer,inner=(16,2) if case['kind']=='g5' else (8,1)
    expected=4*math.pi/3*(outer**3-inner**3)
    sampled_volume=rows[0][17] if result['finite'] else 0
    result['shell_analytic_volume']=expected
    result['shell_sampled_volume']=sampled_volume
    result['shell_volume_relative_quadrature_error']=abs(sampled_volume-expected)/expected
    result['fixed_safe_shell']=bool(result['finite'] and all(
        r[15]==0 and r[16]==0 and r[18]==0 and r[14]>0 and
        abs(r[17]-sampled_volume)<expected*1e-10 for r in rows) and
        result['shell_volume_relative_quadrature_error']<0.01)
    result['horizons_ok']=bool(result.get('successes',0)>0 and result.get('failures',1)==0 and
        result.get('pending_searches',1)==0 and result.get('reported_successes_above_rms_tolerance',1)==0)
    horizons=list((run/'horizon').glob('BHaHAHA_diagnostics.ah*.gp'))
    hr=numeric_rows(horizons[0]) if horizons else []
    enclosing=[math.sqrt(sum(x*x for x in r[2:5]))+r[6] for r in hr if len(r)>=7]
    result['horizon_origin_radius_upper_bound']=max(enclosing) if enclosing else None
    result['horizon_excised']=bool(enclosing and all(math.isfinite(r) and r<inner for r in enclosing))
    result['input_matches']=hashlib.sha256((run/'input.athinput').read_bytes()).hexdigest()==case['input_sha256']
    result['passed']=all(result[k] for k in ('complete','finite','fixed_safe_shell','horizons_ok','horizon_excised','input_matches'))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--submit',action='store_true')
    p.add_argument('--queue',choices=('debug','capacity'),default='debug')
    args=p.parse_args()
    if not CAMPAIGN.exists(): raise SystemExit('Run this controller on Aurora, not locally.')
    STATE_ROOT.mkdir(exist_ok=True)
    with (STATE_ROOT/'advance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=json.loads((WORKFLOW/'cases.json').read_text())
        state=json.loads((STATE_ROOT/'state.json').read_text()) if (STATE_ROOT/'state.json').exists() else dict(
            created=now(),attempts=[],completed=[],status='prepared',
            held_legacy_jobs=['8883678','8888791','8888792'])
        save(state)  # Persist the initial prepared state even in inspection mode.
        jobs=scheduler()
        active={k:v for k,v in jobs.items() if v['job_state'] not in ('F','X')}
        own={k:v for k,v in active.items() if v.get('Job_Name','').startswith('tlp-')}
        if state.get('status')=='submission_uncertain':
            print('Reconcile saved submission intent against PBS before any retry.');return
        if own:
            print(json.dumps(dict(active={k:v['job_state'] for k,v in own.items()})));return
        pending=[a for a in state['attempts'] if a['status']=='submitted']
        for attempt in pending:
            detail=scheduler(attempt['job_id'])
            job=next(iter(detail.values()))
            if job['job_state']!='F':
                print('Awaiting definitive PBS completion: '+attempt['job_id']);return
            case=next(c for c in spec['cases'] if c['name']==attempt['case'])
            run=STATE_ROOT/'runs'/case['name']/attempt['job_id'].split('.')[0]
            try: result=audit(run,case)
            except (OSError,ValueError,KeyError,TypeError) as e: result=dict(passed=False,error=str(e))
            result['pbs_exit_status']=job.get('Exit_status')
            result['passed'] &= job.get('Exit_status')==0
            attempt.update(status='passed' if result['passed'] else 'needs_review',audit=result)
            if result['passed']: state['completed'].append(case['name'])
            else: state['status']='needs_review'
            save(state)
        if state['status']=='needs_review':
            print('Stopped for review; preserve failed/partial evidence. See state.json.');return
        todo=[c for c in spec['cases'] if c['name'] not in state['completed']]
        if not todo:
            state['status']='numerics_complete';save(state)
            print('All cases passed; collect results and prepare figures, presentation, manuscript.');return
        case=todo[0]
        if not args.submit:
            print(json.dumps(dict(next_case=case['name'],completed=len(state['completed']),status=state['status'])));return
        if args.queue=='debug' and any(j.get('queue')=='debug' for j in active.values()):
            print('Debug already occupied by user; defer or choose authorized capacity queue.');return
        nodes=2 if case['block']==64 else 1
        if sum(a.get('nodes',0) for a in state['attempts'])+nodes>spec['max_campaign_node_hours']:
            raise RuntimeError('Finite campaign node-hour reservation budget reached.')
        if sum(a['case']==case['name'] for a in state['attempts'])>=spec['max_attempts_per_case']:
            raise RuntimeError('Case attempt budget reached.')
        used=sum(p.stat().st_size for p in (STATE_ROOT/'runs').rglob('*') if p.is_file()) if (STATE_ROOT/'runs').exists() else 0
        if used>=spec['max_new_output_bytes']: raise RuntimeError('New campaign output limit reached.')
        marker=EXE.parent.parent/'executable.sha256'
        if not EXE.is_file() or not marker.is_file(): raise RuntimeError('GPU build not complete.')
        expected=marker.read_text().split()[0]
        actual=hashlib.sha256(EXE.read_bytes()).hexdigest()
        if actual!=expected: raise RuntimeError('GPU binary hash does not match completed build.')
        actual_input=hashlib.sha256((WORKFLOW/'inputs'/(case['name']+'.athinput')).read_bytes()).hexdigest()
        if actual_input!=case['input_sha256']: raise RuntimeError('Manifest/input mismatch.')
        idx=spec['cases'].index(case)
        name='tlp-{:02d}-{}'.format(idx,len(state['attempts']))
        cmd=['qsub','-A','CompactBinaryMerger','-q',args.queue,'-N',name,
             '-l','select={}'.format(nodes),'-l','walltime=01:00:00',
             '-o',str(STATE_ROOT/(name+'.pbs.log')),
             '-v','CASE='+case['name'],str(WORKFLOW/'run.pbs')]
        intent=dict(time=now(),case=case['name'],nodes=nodes,queue=args.queue,command=cmd,executable_sha256=actual)
        state.update(status='submission_uncertain',intent=intent);save(state)
        r=command(cmd)
        intent.update(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
        if r.returncode==0 and '.aurora-pbs-' in r.stdout.strip():
            intent.update(job_id=r.stdout.strip(),status='submitted')
            state['attempts'].append(intent);state['status']='running';state.pop('intent',None)
        elif r.returncode==38 or ('maximum number' in r.stderr.lower() and 'qsub:' in r.stderr):
            # Explicit PBS rejection created no job; retry only on a later heartbeat.
            state['last_rejection']=intent;state['status']='prepared';state.pop('intent',None)
        save(state)
        print(json.dumps(intent,indent=2))

if __name__=='__main__': main()
