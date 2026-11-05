"""Analyze preserved/admitted fresh-grid pair; never equate its identities."""
from pathlib import Path
import argparse, hashlib, json, statistics


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    plan=read(a.campaign/'plan.json');worker=read(a.campaign/'output/worker.json')
    terminal=read(a.campaign/'terminal-verification.json')
    assert terminal['terminal'] and terminal['scheduler_exit']==0 and terminal['worker_status']=='completed'
    assert worker['plan_sha256']==sha(a.campaign/'plan.json')
    records={}
    for path,v in worker['artifacts'].items():
        p=a.campaign/'output'/path
        assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],path
        records[path]=dict(bytes=v['bytes'],sha256=v['sha256'],verified=True)
    verification=dict(job_id=worker['job_id'],artifact_count=len(records),all_verified=True,artifacts=records)
    (out/'local-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    summary={};histories={}
    for case,verification in zip(worker['cases'],terminal['cases']):
        name=case['name'];d=a.campaign/'output'/name
        report=read(d/'state.report.json')
        assert verification['verified'] and verification['sha256']==sha(d/'state.checkpoint')
        assert report['status']=='completed' and report['steps']==600
        rows=[json.loads(x) for x in (d/'state.history.jsonl').read_text().splitlines()]
        histories[name]=rows
        perfs=[read(p) for p in sorted(d.glob('state.rank*.performance.json'))]
        assert len(perfs)==48 and {p['rank'] for p in perfs}==set(range(48))
        phases={}
        for key in sorted(set.union(*(set(p['exclusive_phases']) for p in perfs))):
            values=[p['exclusive_phases'].get(key,{}).get('seconds',0) for p in perfs]
            phases[key]=dict(min=min(values),mean=statistics.mean(values),max=max(values))
        gen=[]
        for p in sorted(d.glob('state.rank*.generations.jsonl')):
            r=[json.loads(x) for x in p.read_text().splitlines()]
            gen.append(r[-1])
        assert len(gen)==48
        active=sum(g['active'] for g in gen)
        nested={}
        for k in sorted(set.union(*(set(g.get('nested_seconds',{})) for g in gen))):
            vals=[g.get('nested_seconds',{}).get(k,0) for g in gen]
            nested[k]=dict(mean=statistics.mean(vals),max=max(vals))
        end=rows[-1];fixed=next(x for x in end['fixed'] if abs(x['cut']-.3)<1e-14)
        summary[name]=dict(configuration_identity=report['identity'],native_command_seconds=case['native_seconds'],
             physical_GPU_hours=24*case['native_seconds']/3600,GPU_tile_hours=48*case['native_seconds']/3600,
             active_cells=active,exclusive_rank_seconds=phases,nested_overlapping_rank_seconds=nested,
             cell_steps_per_second=600*active/phases['step']['max'],
             report_seconds=report['seconds'],endpoint_fixed03=fixed,
             endpoint_candidate_cut=end['exterior_cutoff'],endpoint_candidate_H=end['H_exterior'],
             endpoint_candidate_M=end['M_exterior'],endpoint_tracker=end['tracker'],
             horizon_accepted=sum(bool(x['horizon_found']) for x in rows),horizon_rows=len(rows),
             endpoint_mass=end['mass'],endpoint_horizon_expansion=end['horizon_trial_expansion_rms'],
             max_null=max(x['null'] for x in rows),max_ghost_null=max(x['ghost_null'] for x in rows),
             pole_max=end['pole_max'],clock_max=end['clock_max'],cfl_max=end['cfl'],
             floor_statistics=report['cumulative_floor_statistics'],
             bulk_sign_counts=report['cumulative_bulk_sign_counts'],
             surface_sign_counts=report['cumulative_surface_sign_counts'],
             boundary_sign_counts=report['cumulative_boundary_sign_counts'])
    old,new=summary['block12'],summary['block24']
    ratios=dict(command_seconds=new['native_command_seconds']/old['native_command_seconds'],
        step_seconds=new['exclusive_rank_seconds']['step']['max']/old['exclusive_rank_seconds']['step']['max'],
        cells=new['active_cells']/old['active_cells'],
        cell_throughput=new['cell_steps_per_second']/old['cell_steps_per_second'],
        fixed03_H=new['endpoint_fixed03']['H']/old['endpoint_fixed03']['H'],
        fixed03_M=new['endpoint_fixed03']['M']/old['endpoint_fixed03']['M'])
    result=dict(job_id=worker['job_id'],cases=summary,block24_over_block12=ratios,
        source_sha256=sha(Path(__file__)),plan_sha256=worker['plan_sha256'],
        qualification='Same four-node allocation/executable/dt/physical duration, distinct mesh/config identities. Largest-rank step timer and separate command elapsed. Nested timers overlap and are not additive. GPU-hours count 24 physical GPUs; 48 tile-hours also reported. Fresh 600 steps to .075M, not 10M stability/convergence or a strong-scaling experiment. Same finest spacing but 24^3 layout refines more volume and changes interfaces/scri spacing. No automatic adoption or restart.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=list(summary)
    ax[0].bar(names,[summary[x]['native_command_seconds'] for x in names],label='native command',alpha=.45)
    ax[0].bar(names,[summary[x]['exclusive_rank_seconds']['step']['max'] for x in names],label='max-rank stepping',width=.45)
    ax[0].set_ylabel('seconds');ax[0].legend()
    for name,rows in histories.items():
        for a,key in zip(ax[1:],['H','M']):
            a.semilogy([r['t_over_M'] for r in rows],
                [next(x for x in r['fixed'] if abs(x['cut']-.3)<1e-14)[key] for r in rows],'-o',label=name)
            a.set_xlabel('t/M');a.set_ylabel(key+' RMS, fixed r > .3');a.legend();a.grid(alpha=.2)
    fig.suptitle('Four-node 600-step block-layout comparison; different resolved volumes')
    for ext in ['png','pdf']:fig.savefig(out/('comparison.'+ext),dpi=170)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
