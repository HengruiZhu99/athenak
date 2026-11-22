"""Idempotent Slurm dispatch for the single static-grid bracket."""
import csv
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from smr_checkpoint import validate_restart

ROOT=Path(__file__).resolve().parent
LOW=Decimal('.9508')
HIGH=Decimal('1.0600')
MID=(LOW+HIGH)/2
WIDTH=Decimal('.0001')


def plan(states):
    values={j:states.get(j,{}).get('status') for j in (LOW,MID,HIGH)}
    if 'needs_review' in values.values(): return 'needs_review',next(j for j,v in values.items() if v=='needs_review')
    if values[MID]=='indeterministic': return 'indeterministic',MID
    for j in (LOW,HIGH,MID):
        if values[j] in (None,'active'): return 'run',j
    collapsed=[j for j,status in values.items() if status=='collapse']
    dispersed=[j for j,status in values.items() if status=='dispersal']
    if not collapsed or not dispersed: return 'unbracketed',None
    lo,hi=max(collapsed),min(dispersed)
    if lo>=hi: return 'inconsistent',None
    while hi-lo>WIDTH:
        mid=(lo+hi)/2
        status=states.get(mid,{}).get('status')
        if status in (None,'active'): return 'run',mid
        if status=='collapse':lo=mid
        elif status=='dispersal':hi=mid
        else:return 'indeterministic',mid
    return 'complete',(lo,hi)


def advance():
    os.chdir(ROOT)
    cases=ROOT/'cases';cases.mkdir(exist_ok=True)
    states={}
    for f in cases.glob('*/state.json'):
        state=json.loads(f.read_text())
        segment=f.parent/f"segment-{state['segment']:03d}"
        mass=json.loads((f.parent/'ring.adm.json').read_text())['M_adm']
        validate_restart(segment/'final.rst',state['t_M']*mass,state['cycle'])
        states[Decimal(f.parent.name)]=state
    action,value=plan(states)
    print(action,value,flush=True)
    ledger=ROOT/'ledger.csv.tmp'
    with ledger.open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['spin','status','t_M','centroid_M','event_t_M'])
        for spin,v in sorted(states.items()):
            writer.writerow([spin,v['status'],v['t_M'],v['mean_Rcirc_M'],(v.get('event') or {}).get('t_M','')])
    ledger.replace(ROOT/'ledger.csv')
    if action!='run':return action
    if value!=MID and states.get(MID,{}).get('status') in (None,'active'):
        submit(cases,MID)
    submit(cases,value)
    return action


def submit(cases,value):
    spin=format(value,'f');case=cases/spin;case.mkdir(exist_ok=True)
    # Reconcile using Slurm as well as the on-disk record after interrupted submissions.
    name='smr288-'+spin
    queued=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%j'],text=True).splitlines()
    if name in queued or name+'-id' in queued:return
    submission=case/'jobs.txt'
    if submission.exists():
        old=submission.read_text().splitlines()[-1].split()[-1]
        report=subprocess.check_output(['sacct','-X','-j',old,'-n','-P','-o','State'],text=True).strip()
        if report!='COMPLETED':
            print('Inspect previous allocation before retry:',old,report);return
    dep=[]
    if not (case/'ring.adm.json').is_file():
        job=subprocess.check_output(['sbatch','--parsable','-J',name+'-id','-o',str(case/'cal-%j.log'),'calibrate.sh',spin],text=True).strip().split(';')[0]
        dep=['--dependency=afterok:'+job,'--kill-on-invalid-dep=yes']
    gpu=subprocess.check_output(['sbatch','--parsable',*dep,'-J',name,'-o',str(case/'job-%j.log'),'batch.sh',spin],text=True).strip().split(';')[0]
    with submission.open('a') as stream:print('gpu',gpu,file=stream)
    print('submitted',gpu,spin,flush=True)


if __name__=='__main__':
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        action=advance()
    if action!='run' and '--from-timer' in sys.argv:
        manager=['busctl','--user','call','org.freedesktop.systemd1','/org/freedesktop/systemd1',
                 'org.freedesktop.systemd1.Manager']
        timer='smr288-dispatch.timer'
        subprocess.run([*manager,'DisableUnitFiles','asb','1',timer,'false'],check=True)
        subprocess.run([*manager,'StopUnit','ss',timer,'replace'],check=True)
