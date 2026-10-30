#!/usr/bin/env python3
"""Verify retained job artifacts and summarize each bounded timestep attempt."""
from pathlib import Path
import argparse,hashlib,json,struct,csv,re

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def verify(directory, case_index=-1):
    read=lambda p:json.loads((directory/p).read_text())
    plan=read('plan.json');binding=read('binding.json');worker=read('worker-status.json')
    assert sha(directory/'plan.json')==binding['plan_sha256']==worker['plan_sha256']
    assert worker['source_manifest_sha256']==binding['source_manifest_sha256']
    assert worker['binaries']==binding['binaries'] and worker['affinity']['passed']
    assert plan['phase']=='timestep_experiment' and plan['n']==16
    assert plan['resources']['ranks']==12 and plan['resources']['nodes']==1
    assert plan['storage']['limit_bytes']==5_000_000_000
    case=worker['cases'][case_index];command=plan['commands'][case_index]
    prefix=Path(command['args'][1]).name
    verified=[]
    for relative,expected in case['artifacts'].items():
        path=directory/Path(relative).relative_to(Path(plan['output_dir']).parent)
        assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],path
        verified.append(dict(path=relative,**expected))
    report=read('output/'+prefix+'.report.json')
    assert report['experimental_timestep'] and report['checkpoint_schema']=='HYP3D_MPI_DT1'
    assert report['mpi_size']==12 and report['backend']=='SYCL'
    assert report['dt']==float(command['args'][4]) and float(command['args'][3]) in (.15,1)
    assert report['scri_penalty_cfl_max']<=.5 and report['scri_dissipation_cfl_max']<=2.2
    cp=directory/('output/'+prefix+'.checkpoint')
    with cp.open('rb') as f:header=f.read(72)
    magic,n,nf,kind,admitted,cells,surface,steps,t,dt=struct.unpack('<16s4i3Q2d',header)
    assert magic.rstrip(b'\0')==b'HYP3D_MPI_DT1' and (n,nf,kind)==(16,31,2)
    assert t==report['t'] and steps==report['steps'] and dt==report['dt']
    assert bool(admitted)==report['checkpoint_admitted']==(report['status']!='failed')
    assert cp.stat().st_size==152+16*(nf*cells+surface)
    assert not case['timed_out'] and not case['storage_error']
    fields=['status','error','steps','t','dt','diagnostic_time','diagnostics_current','max_null_surface','max_null_ghost','min_lapse','lapse_floor_node_interventions','scri_penalty_cfl_max','scri_unlimited_penalty_cfl_max','scri_penalty_scale_min','scri_dissipation_cfl_max','angular_unlimited_cfl_max','angular_epsilon_min','max_cfl','horizon_found','horizon_trial_radius','horizon_trial_expansion_rms','horizon_complete_surface','H_exterior_L2','Mr_exterior_L2']
    result={k:report.get(k) for k in fields}
    result.update(verified=True,job_id=worker['job_id'],dt_over_dx=dt/.01125,t_over_M=3*t,checkpoint_admitted=bool(admitted),artifacts=verified)
    (directory/('verification.json' if case_index==-1 else f'verification-{prefix}.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--case-index',type=int,default=-1);a=p.parse_args();verify(a.directory,a.case_index)
