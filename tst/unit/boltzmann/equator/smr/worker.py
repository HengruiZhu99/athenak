"""Run one spin in segments short enough to stop exactly two ADM masses after detection."""
from pathlib import Path
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile

from smr_checkpoint import validate_restart
from compare import stamp
from diagnostics import profile, read_binary
from endpoint_review import check_online_interiors
from witness_pair import FITS, fit_pair

ROOT = Path(__file__).resolve().parent
SUITE = Path(os.environ['SOURCE'])/'tst/unit/boltzmann/equator'
EXE = ROOT/'centered-finder/athena'
SEED = ROOT/'seed.json'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def first_crossing(rows):
    return next((r for r in rows if float(r['mean_Rcirc_M']) > 3.5), None)


def decide(state, strict_time, t, radius, missing, ambiguous, crossing=None):
    event = state.get('event')
    if strict_time is not None and (not event or event['kind'] != 'collapse'):
        event = dict(kind='collapse', t_M=strict_time)
    if event is None and crossing is not None and not missing and not ambiguous:
        event = dict(kind='dispersal', t_M=float(crossing['t_M']),
                     mean_Rcirc_M=float(crossing['mean_Rcirc_M']), criterion='radius_above_3.5')
    if event is None and abs(t-60)<1e-8:
        if radius > 2.5 and not missing and not ambiguous:
            event = dict(kind='dispersal', t_M=t, mean_Rcirc_M=radius, criterion='radius_at_60M')
        else:
            return None, 'indeterministic'
    finished = event and t >= event['t_M']+2-1.e-8
    if finished and event['kind']=='dispersal' and (missing or ambiguous):
        return event, 'needs_review'
    return event, event['kind'] if finished else 'active'


def analyze(work, dest, mass, state):
    files = sorted((work/'bin').glob('*.adm.*.bin'))
    if not files:
        raise RuntimeError('no saved geometry')
    bounds = sorted(json.loads((ROOT/'mesh-bounds.json').read_text()))
    first = read_binary(files[0])
    if sorted([list(v[:4]) for v in first['mb_geometry']]) != bounds:
        raise RuntimeError('native geometry differs from canonical 76-block SMR')
    rows = []
    for adm in files:
        rad = adm.with_name(adm.name.replace('.adm.','.tmunu.'))
        if stamp(adm) != stamp(rad):
            raise RuntimeError('unmatched radiation and ADM slice')
        rows.append(profile(rad,mass))
    with (dest/'centroids.csv').open('w') as out:
        writer = csv.DictWriter(out,rows[0]); writer.writeheader(); writer.writerows(rows)
    bycycle = {r['cycle']:p for r,p in zip(rows,files)}
    native = check_online_interiors(work,files,mass)
    save(dest/'online-interiors.json',native)
    verified = [r['cycle'] for r in native if r['verified']]
    branches = {}
    marginal = []
    scheduled = range((state['cycle']//120+1)*120, rows[-1]['cycle']+1, 120)
    horizon = work/'torus.cartoon_m0_horizon_0.txt'
    # A short terminal segment can finish before its first finder call.
    lines = horizon.read_text().splitlines() if scheduled or horizon.exists() else []
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        v=line.split(); c=int(v[0]); branches.setdefault(c,set()).add(v[2])
        if v[3]=='1': marginal.append(c)
    # Read the switch recorded by the solver so archived three-branch segments
    # and new centered segments are checked against their actual search policy.
    offcenter='true'
    for line in first['header']:
        name, sep, val = line.partition('=')
        if sep and name.strip()=='cartoon_search_offcenter_0':
            offcenter=val.strip().lower()
    expected={'origin'} if offcenter in ('false','0') else {'origin','plus','minus'}
    missing=[c for c in scheduled
             if branches.get(c)!=expected or c not in bycycle]
    if not state.get('event') or state['event']['kind']!='collapse':
        elapsed = fit_pair(SEED, dest, files)
        save(dest/'witness-timing.json', dict(mode='parallel', slices=len(files), parallel_s=elapsed))
        for degree, _ in FITS:
            for r in json.loads((dest/f'witness-l{degree}.json').read_text()):
                if r['all_negative']: verified.append(r['cycle'])
    crossing=first_crossing(rows)
    onset=[bycycle[crossing['cycle']]] if crossing is not None and not state.get('event') else []
    for p in set([files[-1]]+onset+([bycycle[min(verified)]] if verified else [])+
                 [bycycle[c] for c in marginal[:1] if c in bycycle]):
        for field in ['adm','tmunu','con']:
            src=p.with_name(p.name.replace('.adm.',f'.{field}.'))
            shutil.copy2(src,dest/src.name)
    first_time=stamp(bycycle[min(verified)])[0]/mass if verified else None
    # An unverified online marginal surface needs separate adjudication before a dispersal label.
    ambiguous=bool(marginal or native)
    return rows[-1],first_time,missing,ambiguous,crossing


def run(spin):
    case=ROOT/'cases'/spin
    meta=json.loads((case/'ring.adm.json').read_text()); mass=meta['M_adm']
    if abs(meta['spin_ratio']-float(spin))>1e-8 or meta['allow_counterrotation']:
        raise ValueError('wrong calibrated initial data')
    statefile=case/'state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(t_M=0.,cycle=0,status='active',segment=0,missing=False,ambiguous=False)
    published=sorted(case.glob('segment-*/state.json'))
    if published:
        newest=json.loads(published[-1].read_text())
        if newest['segment']>state['segment']:
            validate_restart(published[-1].parent/'final.rst',newest['t_M']*mass,newest['cycle'])
            save(statefile,newest)
            state=newest
    # Pending pre-upgrade allocations may have published their first segment.
    if published and not state.get('event') and state['status']=='active':
        for entry in published:
            with (entry.parent/'centroids.csv').open() as stream:
                crossing=first_crossing(csv.DictReader(stream))
            if crossing and not state['missing'] and not state['ambiguous']:
                state['event'],state['status']=decide(state,None,state['t_M'],state['mean_Rcirc_M'],
                                                      False,False,crossing)
                save(statefile,state)
                break
    while state['status']=='active':
        event=state.get('event')
        stop=event['t_M']+2 if event else 60.
        goal=min(stop,state['t_M']+1.)
        n=state['segment']+1
        work=case/f'work-{n:03d}-{os.environ["SLURM_JOB_ID"]}'
        if work.exists():
            work=Path(tempfile.mkdtemp(dir=case,prefix=work.name+'-resume-'))
        else:
            work.mkdir()
        (work/'ring.adm').symlink_to(case/'ring.adm')
        # Fully reduced segment directories are committed together with the restart.
        dest=work/'keep'; dest.mkdir()
        launch=['-i',str(ROOT/'smr.in')]
        prev=case/f'segment-{n-1:03d}'/'final.rst'
        if n>1:
            validate_restart(prev, state['t_M']*mass, state['cycle'])
            launch=['-r',str(prev),'-i',str(ROOT/'finder-restart.in')]
        cmd=['srun','--exact','-N1','-n8','-c4','--gres=gpu:8','--gpu-bind=mask_gpu:0xff',
             '--cpu-bind=none','--mpi=pmix','--kill-on-bad-exit=1','python3',
             str(SUITE.parent/'stellar_photon_rank.py'),str(EXE),*launch,
             f'time/tlim={goal*mass:.17g}']
        print('target',goal,'M', 'work',work,flush=True)
        with (work/'stdout').open('w') as log:
            subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'Terminating on time limit' not in (work/'stdout').read_text():
            raise RuntimeError('solver stopped early; work directory retained')
        rst=sorted((work/'rst').glob('*.rst'))[-1];validate_restart(rst)
        last,found,missing,ambiguous,crossing=analyze(work,dest,mass,state)
        validate_restart(rst, last['t_M']*mass, last['cycle'])
        if abs(last['t_M']-goal)>1e-8: raise RuntimeError('terminal time mismatch')
        all_missing=state['missing'] or bool(missing)
        all_ambiguous=state['ambiguous'] or ambiguous
        event,status=decide(state,found,goal,last['mean_Rcirc_M'],all_missing,all_ambiguous,crossing)
        state=dict(t_M=goal,cycle=last['cycle'],segment=n,status=status,event=event,
                   mean_Rcirc_M=last['mean_Rcirc_M'],missing=all_missing,ambiguous=all_ambiguous)
        save(dest/'state.json',state)
        for p in [work/'stdout',*work.glob('*.txt'),*work.glob('*.hst')]: shutil.copy2(p,dest/p.name)
        # There is always at least one full checkpoint until the replacement is published.
        rst.rename(dest/'final.rst'); dest.rename(case/f'segment-{n:03d}')
        save(statefile,state)
        if prev.is_file(): prev.unlink()
        shutil.rmtree(work)
        print(state,flush=True)
        if time.time()+5400>int(os.environ.get('SLURM_JOB_END_TIME',2**40)):
            break


if __name__=='__main__': run(sys.argv[1])
