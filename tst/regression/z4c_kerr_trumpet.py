#!/usr/bin/env python3
"""Stage, metric, matter-response and MPI tests for the stationary Kerr trumpet.

Short correctness regression; does not certify perturbation stability. The
independent spherical-metric unit tests are in tst/unit/kerr_trumpet.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from z4c_background_balance import read_csv
from z4c_gauge_pulse import snapshot


def replace(text, key, value):
    return re.sub(r'^'+re.escape(key)+r'\s*=.*', f'{key} = {value}', text, flags=re.M)


def inspect(run, vacuum, refined, spin):
    operations = set()
    stages = set()
    rows_count = 0
    for path in run.glob('z4c_balance_rank*.csv'):
        for row in read_csv(path):
            rows_count += 1
            operations.add(row['operation']); stages.add(row['stage'])
            assert int(row['nonfinite']) == 0 and np.isfinite(float(row['max_abs'])), row
            if vacuum:
                assert float(row['max_abs']) == 0 and int(row['bit_mismatch']) == 0, row
    required = {'init_state', 'init_reconstructed', 'init_projected', 'init_recast',
                'rhs_full_vs_bg', 'volume_rhs', 'post_ko_rhs', 'pre_excision_rhs',
                'post_excision_rhs', 'pre_boundary_rhs', 'post_boundary_rhs',
                'post_rk', 'post_exchange', 'pre_physical_bc', 'post_physical_bc',
                'pre_projection', 'post_projection', 'post_recast'}
    if refined: required |= {'post_restrict', 'post_prolong'}
    assert required <= operations and {'1', '2', '3'} <= stages
    for path in run.glob('z4c_geometry_rank*.csv'):
        for row in read_csv(path):
            assert int(row['nonfinite']) == 0
            if vacuum:
                assert float(row['max_abs']) == 0 and int(row['bit_mismatch']) == 0, row
    det_error = trace_error = 0.
    for path in run.glob('z4c_algebraic_rank*.csv'):
        for row in read_csv(path):
            if row['operation'] in ['init_projected', 'post_projection']:
                assert int(row['nonfinite']) == 0
                det_error = max(det_error, float(row['det_error']))
                trace_error = max(trace_error, float(row['trace_A']))
    assert det_error < 2e-14 and trace_error < 2e-14, (det_error, trace_error)
    minima = np.full(4, np.inf)
    background_error = 0.
    for path in run.glob('z4c_snapshot_*.json'):
        meta = json.loads(path.read_text())
        state = np.fromfile(path.with_suffix('.bin'), '<f8').reshape(meta['shape'])
        bg = np.fromfile(path.with_suffix('.background.bin'), '<f8').reshape(meta['shape'])
        full = state if meta['compare_background'] else state + bg
        assert np.isfinite(full).all()
        gxx,gxy,gxz,gyy,gyz,gzz = [full[:,i] for i in range(1,7)]
        minor2 = gxx*gyy-gxy*gxy
        det = gxx*gyy*gzz + 2*gxy*gxz*gyz - gxx*gyz**2 - gyy*gxz**2 - gzz*gxy**2
        vals = [full[:,18].min(),full[:,0].min(),minor2.min(),det.min()]
        minima = np.minimum(minima, vals)
        assert min(vals) > 0, (path, vals)
        if meta['operation'] != 'init_state': continue
        for block, b in zip(meta['blocks'], bg):
            ng=meta['ng']; axes=[block['xmin'][i]+(np.arange(b.shape[3-i])-ng+.5)*block['dx'][i] for i in range(3)]
            z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
            xyz=np.array([x,y,z]);r=np.sqrt((xyz**2).sum(0));R=r+1
            n=xyz/r;w=np.array([-n[1],n[0],np.zeros_like(r)])
            Sigma=R**2+spin**2*n[2]**2;X=(R**2+spin**2)**2-spin**2*(x*x+y*y)
            chi=r*r/(Sigma*X)**(1/3)
            c=np.sqrt(1-spin**2)
            expected={0:chi,18:r*np.sqrt(Sigma/X)}
            for a in range(3): expected[19+a]=(c*(R*R+spin*spin)*xyz[a]-spin*(2*r+1+spin*spin)*r*w[a])/X
            for q,(a,b2) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                expected[1+q]=chi/r**2*(Sigma*(a==b2)+spin**2*(1+2*R/Sigma)*w[a]*w[b2]-spin*c*(n[a]*w[b2]+w[a]*n[b2]))
            background_error=max(background_error,max(float(np.max(abs(b[q]-v))) for q,v in expected.items()))
    assert background_error < 8e-14, background_error
    last=snapshot(run,'post_recast',2,3)
    response=max(float(np.max(abs(v[:18]))) for _,v in last.values())
    assert (response==0) if vacuum else (response>0)
    return dict(operations=sorted(operations),geometric_response=response,
                raw_full_min_alpha_chi_minor2_det=minima.tolist(),
                background_max_error=background_error,projection_det_error=det_error,
                projection_trace_error=trace_error,
                active_block_hashes={str(gid):hashlib.sha256(v.tobytes()).hexdigest() for gid,(_,v) in last.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ranks',type=int,nargs='+',default=[1,4]);p.add_argument('--launcher',default='mpiexec')
    p.add_argument('--cases',nargs='+',default=['vacuum','lapse','atmosphere'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    text=(Path(__file__).resolve().parents[1]/'inputs/z4c_puncture_background.athinput').read_text()
    updates=dict(bh_background='kerr_trumpet',bh_spin='.9',a='.9',extrap_order=2,
                 residual_lapse_damping='.1',shift_eta='.02',damp_kappa1=0)
    for k,v in updates.items():text=replace(text,k,v)
    results={}
    for refined in [False,True]:
        for case in args.cases:
            for ranks in args.ranks:
                name=f'{case}_{"refined" if refined else "uniform"}_r{ranks}'
                run=args.output.resolve()/name;run.mkdir(exist_ok=False);config=text
                if refined:
                    config=replace(replace(config,'refinement','static'),'max_nmb_per_rank',128)
                    config+='\n<refined_region0>\nlevel = 1\nx1min = -1\nx1max = 1\nx2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n'
                if case=='lapse':config=config.replace('<problem>','<problem>\nvacuum_gauge_pulse_amplitude = 1e-8')
                if case=='atmosphere':
                    for k,v in dict(zero_tmunu='false',zero_tmunu_feedback='false',dfloor='1e-14',pfloor='1e-26').items():config=replace(config,k,v)
                (run/'input.athinput').write_text(config)
                with (run/'run.log').open('w') as log:
                    subprocess.run([args.launcher,'-n',str(ranks),str(args.exe.resolve()),'-i','input.athinput'],cwd=run,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert 'Terminating on cycle limit' in (run/'run.log').read_text()
                q=inspect(run,case=='vacuum',refined,.9)
                if ranks!=args.ranks[0]:
                    ref=results[name.rsplit('_r',1)[0]+f'_r{args.ranks[0]}']
                    assert q==ref, f'MPI partition changed {name}'
                results[name]=q
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(name,'PASS',q['geometric_response'],flush=True)

if __name__=='__main__':main()
