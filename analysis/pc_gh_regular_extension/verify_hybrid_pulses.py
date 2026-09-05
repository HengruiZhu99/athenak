"""Flat subsidiary prediction with finite damping and actual-time projection jumps.

This oracle transports exactly between jumps; it is independent of the numerical
spatial/RK evolution. Convergence measures all 33 reduction and curl components.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from make_inputs import parse
from verify_pulses import table
from verify_smooth_pulses import prediction,rate_gradient


def projection_times(rows, integrator):
    """Read true end-step jumps, including the bracketed legacy schema."""
    steps=[]
    seen=set()
    in_projection=False
    for row in rows:
        if row['region']!='all': continue
        if row['operation']=='3' and row['quantity']=='Rw':
            in_projection=row['phase']=='before'
        # Initial diagnostic schema reused physical-BC operation 8.
        # Recover only the actual correction inside the 3-before/after
        # bracket; never treat boundary measurements as map events.
        correction=row['operation']=='100' or (row['operation']=='8' and in_projection)
        if correction and row['phase']=='after' and row['quantity']=='delta_p':
            assert int(row['stage'])=={'rk3':3,'rk4':4}[integrator]
            key=(int(row['cycle']),float(row['t_step']))
            assert key not in seen,('Duplicate projection event',key)
            seen.add(key);steps.append(float(row['t_step'])+float(row['dt']))
    return steps


def measure(run):
    params=parse((run/'used_input.athinput').read_text());pc=params['pc_gh']
    data=table(run,'final');t=float(data['time'][0])
    assert all(np.isfinite(data[key]).all() for key in data.dtype.names)
    E,C=prediction(data,params)
    steps=[]
    if pc.get('project_reduction_constraints','false')=='true':
        files=list(run.glob('*.hybrid.csv'));assert len(files)==1,files
        with files[0].open() as stream:
            steps=projection_times(csv.DictReader(stream),params['time']['integrator'])
        assert steps and abs(steps[-1]-t)<1e-12,(steps[-1:] or None,t)
        times=np.array(steps);assert np.all(np.diff(times)>0)
        tensor=np.zeros((len(data),3,11))
        tensor[:,:,0]=E[:,:3];tensor[:,:,1:7]=E[:,3:21].reshape(-1,3,6)
        tensor[:,:,7]=E[:,21:24];tensor[:,:,8:11]=E[:,24:].reshape(-1,3,3)
        points=np.column_stack([data[a] for a in 'xyz'])
        amplitude=np.ones(len(data));gradient=np.zeros_like(points)
        shift=float(params['problem'].get('pulse_shift',.5))
        policy=dict(pc,reduction_profile='smooth_core',reduction_rate='0',reduction_inner_rate='1')
        for time in times:
            along=points.copy();along[:,0]+=shift*(t-time)
            P,dP=rate_gradient(along,policy,time,shift)
            if pc.get('reduction_projection_profile','global')=='global':
                P[:]=1;dP[:]=0
            gradient=gradient*(1-P[:,None])-amplitude[:,None]*dP
            amplitude*=1-P
        C=C*amplitude[:,None]+np.column_stack([
            gradient[:,i,None]*tensor[:,j,:]-gradient[:,j,None]*tensor[:,i,:]
            for i,j in [(0,1),(0,2),(1,2)]])
        E=E*amplitude[:,None]
    observed_E=np.column_stack([data[f'E{n}'] for n in range(33)])
    observed_C=np.column_stack([data[f'C{n}'] for n in range(33)])
    scale=float(params['problem'].get('pulse_amplitude',1e-8))
    def norms(error):
        return dict(linf_over_amplitude=float(abs(error).max()/scale),
                    l2_over_amplitude=float(np.sqrt((error**2*data['volume'][:,None]).sum())/scale))
    result=dict(time=t,projection_steps=len(steps),reductions=norms(observed_E-E),
                curls=norms(observed_C-C),scope=__doc__)
    (run/'hybrid-pulse-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path,nargs='+')
    for run in ap.parse_args().runs: print(run,json.dumps(measure(run)),flush=True)
