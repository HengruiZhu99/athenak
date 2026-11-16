#!/usr/bin/env python3
"""Independent single-threaded analysis tasks, one on each allocated node."""
import json,os,sys,time,subprocess
from pathlib import Path
from nonradiative_native_analysis import run
root=Path(os.environ['NONRAD_ROOT']);out=Path(os.environ['NONRAD_ANALYSIS'])
rank_environment={key:os.environ[key] for key in ('PALS_RANKID','PMI_RANK','PMIX_RANK','OMPI_COMM_WORLD_RANK') if key in os.environ}
print('MPI rank environment:',rank_environment,flush=True)
assert rank_environment,'launcher did not identify global MPI rank'
assert len(set(rank_environment.values()))==1,rank_environment
rank=int(next(iter(rank_environment.values())));assert rank in (0,1)
task=sys.argv[1];start=time.monotonic()
if task=='pilot':
    name=('control-N12','gauge-N12')[rank]
    run(root/'runs/pilot-9093394'/name,out/name)
elif task=='analytic':
    from nonradiative_analytic_suite import run as analytic
    analytic(root,out,rank)
elif task=='analytic-followup':
    from nonradiative_analytic_suite import gauge_followup
    gauge_followup(root,out,rank)
elif task.startswith('native-N'):
    n=task.split('-')[1]
    candidates=list((root/'runs').glob('pair-'+n+'-*'));assert len(candidates)==1,candidates
    native=candidates[0];assert (native/'exit-code.txt').read_text().strip()=='0'
    if n=='N12' and rank==0:
        seeds=[next((native/name/'seed/rst').glob('*.rst')) for name in ('control-N12','gauge-N12','gauge-N12-halfdt')]
        subprocess.run([sys.executable,str(Path(__file__).with_name('nonradiative_seed_check.py')),str(seeds[0]),str(seeds[1]),str(out/'seed-halfdt-check.json'),'--halfdt',str(seeds[2])],check=True)
    name=('control-','gauge-')[rank]+n
    run(native/name,out/name)
    if n=='N20':
        from nonradiative_pointwise_summary import run as pointwise_summary,compare_half
        pointwise_summary(out/name)
        # Earlier allocations are complete before this sequential debug job.
        for earlier in ('N12','N16'):
            prior=list((root/'analysis').glob('native-'+earlier+'-*'))
            for location in prior:
                candidate=location/(('control-','gauge-')[rank]+earlier)
                if (candidate/'summary.json').exists():pointwise_summary(candidate)
                half=location/'gauge-N12-halfdt'
                if rank==0 and (half/'summary.json').exists():
                    pointwise_summary(half);compare_half(location/'gauge-N12',half)
    if n=='N12' and rank==0:
        run(native/'gauge-N12-halfdt',out/'gauge-N12-halfdt')
    if n=='N12' and rank==1:
        from nonradiative_analysis_controls import run as controls
        controls(out/name)
    if n=='N12':
        from nonradiative_analytic_suite import gauge_followup
        gauge_followup(root,out/'analytic-followup',rank)
else:
    raise ValueError(task)
(out/f'rank-{rank}-receipt.json').write_text(json.dumps(dict(task=task,rank=rank,seconds=time.monotonic()-start),indent=2)+'\n')
