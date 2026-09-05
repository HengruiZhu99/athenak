"""Prepare the agreed six-arm hybrid campaign, preserving source input hashes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from make_inputs import parse,write,ROOT

CANDIDATES={'C1':(1,1,False),'C4':(4,4,False),'C16':(16,16,False),
            'R4':(1,4,False),'R16':(1,16,False),'P1':(1,1,True)}


def candidate(b,name,core=.125,taper=.5,ceiling=.0125):
    outer,inner,projection=CANDIDATES[name]
    b['mesh']['nghost']='4'
    b['pc_gh'].update(reduction_system='advective',reduction_rate=str(outer),
        reduction_inner_rate=str(inner),reduction_profile='smooth_core' if inner>outer else 'constant',
        project_reduction_constraints=str(projection).lower(),project_gauge_constraints='false',
        reduction_projection_profile='smooth_core' if projection else 'global',
        reduction_core_radius=str(core),reduction_taper_radius=str(taper),
        research_dt_ceiling=str(ceiling),hybrid_monitor='true',reduction_monitor='true',
        spatial_order='6',dissipation='.3')
    b['time'].update(integrator='rk3',cfl_number='.2',ndiag='1')
    b['problem']['require_cuda']='true'
    if b['problem'].get('pgen_name') == 'regular_extension_pulse':
        b['problem']['pulse_allow_reduction_projection']=str(projection).lower()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    sources={};manifest=[]
    def load(path):
        text=path.read_text();sources[str(path)]=hashlib.sha256(text.encode()).hexdigest()
        return parse(text)
    def save(path,b,stage):
        assert int(b['mesh']['nghost']) >= int(b['pc_gh']['spatial_order'])//2+1
        assert sum(v.get('file_type')=='rst' for k,v in b.items() if k.startswith('output'))<=1
        write(out/path,b); manifest.append(dict(input=str(path),stage=stage))
    uniform=load(args.evidence/'della-cuda/single-discretization/pcgh-l1-o6-rk3-cfl0.2-uniform-r16-R8/used_input.athinput')
    core=load(args.evidence/'della-cuda/rate16-qualification/pcgh-l16-o6-rk3-core256-R8/used_input.athinput')
    for name,b in [('uniform-C1-original',uniform),('core256-C16-original',core)]:
        save(Path('reproduction')/(name+'.athinput'),b,'reproduction')
    for grid,base,end in [('uniform128',uniform,12),('core256',core,6)]:
        for name in CANDIDATES:
            b=deepcopy(base);candidate(b,name);b['time']['tlim']=str(end)
            # Checkpoints and exact dt history survive allocation boundaries.
            rst=[v for k,v in b.items() if k.startswith('output') and v.get('file_type')=='rst']
            if rst:
                rst[0]['dt']='1'
            else:
                used=[int(k[6:]) for k in b if k.startswith('output')]
                b['output'+str(max(used,default=0)+1)]=dict(file_type='rst',dt='1')
            save(Path('screen')/f'{grid}-{name}.athinput',b,'screen')
    oracle=load(ROOT/'analysis/pc_gh_regular_extension/oracle.athinput')
    for dim in [1,2,3]:
        for order in [2,6]:
            for mode in ['zero','full','taper','global']+(['overlap'] if dim==3 else []):
                b=deepcopy(oracle);candidate(b,'P1');b['time'].update(nlim='0',tlim='0')
                b['mesh']['nghost']='4' if order==6 else '2'
                for d in [1,2,3]:
                    b['mesh'][f'nx{d}']=b['meshblock'][f'nx{d}']='16' if d<=dim else '1'
                    b['mesh'][f'x{d}min']='-1';b['mesh'][f'x{d}max']='1'
                b['pc_gh'].update(spatial_order=str(order),dissipation='0',
                    project_reduction_constraints='false')
                b['problem']['pgen_name']='regular_projection_oracle'
                if mode=='zero': b['pc_gh']['reduction_center_x']='10'
                if mode=='full': b['pc_gh'].update(reduction_core_radius='4',reduction_taper_radius='5')
                if mode=='global': b['pc_gh']['reduction_projection_profile']='global'
                if mode=='overlap':
                    b['pc_gh']['reduction_follow_trackers']='true'
                    for n,x in enumerate([-.25,.25]):
                        b['pc_gh'].update({f'co_{n}':'true',f'co_{n}_type':'BH',f'co_{n}_x':str(x),f'co_{n}_y':'0',f'co_{n}_z':'0'})
                save(Path('oracle')/f'd{dim}-o{order}-{mode}.athinput',b,'oracle')
                if dim==3 and order==6 and mode in ['global','taper','overlap']:
                    mpi=deepcopy(b);mpi['mesh']['nx1']='32'
                    save(Path('mpi-oracle')/f'd3-o6-{mode}.athinput',mpi,'mpi-oracle')
    flat=load(args.evidence/'inputs-v2/flat/minkowski-advective-l1.athinput')
    for name in CANDIDATES:
        b=deepcopy(flat);candidate(b,name)
        b['time'].update(tlim='.25',nlim='-1')
        save(Path('flat')/f'{name}.athinput',b,'flat')
        for n in [32,64,128]:
            b=load(args.evidence/f'della-cuda/order6-wave-fullsteps/shifted-wave-advective-l16-n{n}/used_input.athinput')
            candidate(b,name)
            # Preserve the exact harmonic gauge and full-step resolution ladder.
            save(Path('waves')/f'{name}-n{n}.athinput',b,'waves')
        for family in ['p','Q','L','B']:
            for n in [256,512,1024]:
                b=load(args.evidence/f'smooth-controls/fixed-pulses/{family}-n{n}.athinput')
                candidate(b,name)
                b['mesh']['nghost']='4'
                b['time'].update(tlim='.25',nlim='-1')
                b['pc_gh']['reduction_center_x']='0'
                save(Path('pulses')/f'{name}-{family}-n{n}.athinput',b,'pulses')
    for name in ['R16','P1']:
        for n in [32,64,128]:
            b=load(args.evidence/f'smooth-controls/moving-waves/wave-n{n}.athinput')
            candidate(b,name);b['pc_gh']['reduction_follow_trackers']='true'
            save(Path('moving-waves')/f'{name}-n{n}.athinput',b,'moving-waves')
        for family in ['p','Q','L','B']:
            b=load(args.evidence/f'smooth-controls/moving-pulses/{family}-n32.athinput')
            candidate(b,name);b['pc_gh']['reduction_follow_trackers']='true'
            b['time']['tlim']='.25';b['output1']=dict(file_type='rst',dt='.125')
            save(Path('moving-pulses')/f'{name}-{family}-n32.athinput',b,'moving-pulses')
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sources,cases=manifest),indent=2)+'\n')


if __name__=='__main__':main()
