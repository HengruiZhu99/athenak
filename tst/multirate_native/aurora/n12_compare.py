"""Compare the two completed 10 M0 N12 runs at their synchronized endpoint."""
import argparse
import json
import math
from pathlib import Path
from n12_paths import trajectory_directory
import sys
import tempfile

import numpy as np

from compare import compare
from restart_roundtrip import parse, payload
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from endpoint_diagnostics import diagnostics
from scri_collect import collect
from scri_news import analyze
from scri_reader import read


def regular_cuts(receipts, cadence):
    """Select only scheduled cuts from successfully accepted segments."""
    paths = {}
    for segment in receipts:
        directory = Path(segment['checkpoint']).parent.parent
        for path in directory.glob('*.o20.*.r00000000.scri'):
            record = read(path)
            cycle = int(record['record'][0])
            if not segment['start_cycle'] <= cycle <= segment['cycle']:
                raise ValueError(f'scri cut {cycle} is outside its accepted segment')
            if cycle % cadence:
                continue  # final wall-time dumps are asynchronous between methods
            expected = cycle*segment['dt']
            if abs(float(record['time'][0])-expected) > max(1,cycle)*math.ulp(expected):
                raise ValueError(f'scri cut {cycle} has an inconsistent timestamp')
            paths.setdefault(cycle,path)
    expected = range(0,receipts[-1]['cycle']+1,cadence)
    if set(paths) != set(expected):
        raise ValueError(f'missing scheduled scri cuts: {sorted(set(expected)-paths.keys())}')
    return [paths[cycle] for cycle in expected]


def extract_news(directory, sources):
    """Recompute on the verified common cadence and retain the extraction results."""
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        scratch = Path(temporary)
        records = collect(sources[0].parent,scratch/'cache',capacity=len(sources),record_paths=sources)
        if len(records)!=len(sources):
            raise ValueError('incomplete common waveform series')
        analyze(scratch/'cache',scratch/'news',lmax=8)
        with np.load(scratch/'news'/'news.npz') as archive:
            result = {k:archive[k] for k in ('bondi_time','ell_m','news_modes')}
        output = directory/'news'
        output.mkdir(exist_ok=True)
        for path in sorted((scratch/'news').iterdir()):
            path.replace(output/path.name)
        (scratch/'cache'/'records.json').replace(directory/'regular-cuts.json')
    return result


def work_and_science(root, mode):
    case = trajectory_directory(root,mode)
    last = json.loads((case/'state.json').read_text())
    if not last['target_reached'] or abs(last['time_M0']-10)>last['cycle']*math.ulp(10.):
        raise ValueError('N12 mode has not reached the common 10 M0 endpoint: '+mode)
    receipts = sorted((json.loads(p.read_text()) for p in case.glob('segment-*/receipt.json')),
                      key=lambda x:x['start_cycle'])
    next_cycle = 0
    for item in receipts:
        if (item['start_cycle'] != next_cycle or item['cycle'] <= next_cycle or
                any(item[k]!=last[k] for k in ('binary_sha256','mode','manifest_sha256','input_source_sha256','dt','target_cycles'))):
            raise ValueError('N12 segment provenance or cycle gap in '+mode)
        next_cycle = item['cycle']
    if (next_cycle != last['cycle'] or last['mode']!=mode or
            last['cycle'] != last['target_cycles'] or last['checkpoint']!=receipts[-1]['checkpoint']):
        raise ValueError('missing final segment receipt in '+mode)
    end_dir = Path(last['checkpoint']).parent.parent
    ids, time, endpoint = diagnostics(end_dir,scri_only=True,output_number=20)
    if abs(3*time-last['time_M0'])>math.ulp(10.):
        raise ValueError('scri output and checkpoint times disagree')
    history = np.atleast_2d(np.loadtxt(next(end_dir.glob('*.z4c_hyperboloidal.hst'))))[-1]
    if abs(3*history[0]-10)>last['cycle']*math.ulp(10.):
        raise ValueError('constraint history ends at a different physical time')
    for name,x in zip(('hamiltonian_rms','momentum_rms','theta_rms','z_rms'), history[3:7]):
        endpoint[name]=x
    horizons = {}
    for path in sorted(end_dir.glob('*.horizon_summary_*.txt')):
        if not any(line.strip() and not line.lstrip().startswith('#') for line in path.read_text().splitlines()):
            continue
        row = np.atleast_2d(np.loadtxt(path))[-1]
        component = path.stem.rsplit('_',1)[-1]
        horizons[component] = dict(time_M0=3*float(row[1]),mass=float(row[2]),spin=float(row[6]),
                                   area=float(row[7]),expansion_rms=math.sqrt(float(row[8])))
    cadence = 6 if mode=='multirate' else 96
    sources = regular_cuts(receipts,cadence)
    news = extract_news(root/'n12-10M'/'analysis'/mode,sources)
    first_restart_dir=Path(receipts[0]['checkpoint']).parent
    initial=sorted(first_restart_dir.glob('*.rst'))[0]
    if parse(initial)['cycle'] != 0:
        raise ValueError('missing initial N12 checkpoint in '+mode)
    wall = sum(item['wall_seconds'] for item in receipts)
    return (dict(**last,simulated_M0=10.),ids,endpoint,news,
            dict(segments=len(receipts),elapsed_seconds=wall,seconds_per_M0=wall/10,
                 physical_gpus=12,nodes=2,scri_cuts=len(sources),initial_checkpoint=str(initial),last_reported_horizons=horizons))


def active_field_errors(left,right):
    """Separate evolved N12 bulk and surface points from zero restart padding."""
    a,b = (parse(Path(v['checkpoint'])) for v in (left,right))
    x,y = payload(a),payload(b)
    n,overlap = 12,3
    bulk = (n+2*overlap)**3
    if x.shape!=y.shape or x.shape[1:]!=(bulk+n*n,39):
        raise ValueError('unexpected N12 restart layout')
    topology=np.frombuffer(a['topology'],dtype='=i4').reshape(-1,5)
    result = {key:dict(points=0,squared=0.,max_scaled=0.) for key in
              ('cartesian_bulk','spherical_bulk','scri')}
    floors = {key:dict(count=0,nonpositive=0,nonfinite=0,max_correction=0.) for key in
              ('multirate','uniform')}
    for location,u,v in zip(topology,x,y):
        group='cartesian_bulk' if location[4]==0 else 'spherical_bulk'
        for key,first,second in ((group,u[:bulk],v[:bulk]),('scri',u[bulk:],v[bulk:])):
            mask=first[:,0]>0
            if not np.array_equal(mask,second[:,0]>0):
                raise ValueError('different native evolved-point support')
            if np.any(first[~mask]) or np.any(second[~mask]):
                raise ValueError('nonpositive accepted chi in a populated restart slot')
            if not mask.any():
                continue
            first,second=first[mask],second[mask]
            for state,name in ((first,'multirate'),(second,'uniform')):
                floor=floors[name]
                for field,index in (('count',33),('nonpositive',34),('nonfinite',35)):
                    floor[field] += int(state[:,index].sum())
                floor['max_correction']=max(floor['max_correction'],float(state[:,36].max()))
            f,g=first[:,:31],second[:,:31]
            scaled=np.abs(f-g)/np.maximum(1.,np.maximum(np.abs(f),np.abs(g)))
            row=result[key]
            row['points']+=len(f)
            row['squared']+=float(np.sum(scaled**2))
            row['max_scaled']=max(row['max_scaled'],float(scaled.max()))
    for row in result.values():
        row['rms_scaled']=math.sqrt(row.pop('squared')/(31*row['points']))
    return result,floors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    a,aid,ae,an,at=work_and_science(root,'multirate')
    b,bid,be,bn,bt=work_and_science(root,'uniform')
    if any(a[key]!=b[key] for key in ('manifest_sha256','input_source_sha256')):
        raise ValueError('N12 initial data or spatial input differs between methods')
    if not np.array_equal(aid,bid) or not np.array_equal(an['ell_m'],bn['ell_m']):
        raise ValueError('incompatible scri nodes or waveform modes')
    ta,tb=an['bondi_time'],bn['bondi_time']
    active=(ta>=tb.min()) & (ta<=tb.max())
    if active.sum()<2: raise ValueError('insufficient common Bondi-time coverage')
    # Interpolate the already-extracted news for comparison on common Bondi cuts.
    reference=np.column_stack([np.interp(ta[active],tb,v) for v in bn['news_modes'].T])
    error=an['news_modes'][active]-reference
    if not np.isfinite(error).all(): raise ValueError('nonfinite news difference')
    endpoint={}
    for key in ae:
        delta=ae[key]-be[key]
        if not np.isfinite(delta).all(): raise ValueError('nonfinite '+key)
        endpoint[key]=dict(rms_difference=float(np.sqrt(np.mean(delta**2))),
                           max_abs_difference=float(np.max(np.abs(delta))),
                           multirate_max_abs=float(np.max(np.abs(ae[key]))),
                           uniform_max_abs=float(np.max(np.abs(be[key]))),
                           multirate_rms=float(np.sqrt(np.mean(ae[key]**2))),
                           uniform_rms=float(np.sqrt(np.mean(be[key]**2))))
    report=dict(fields=compare(a,b),endpoint=endpoint,uniform=bt,multirate=at,
                elapsed_speedup_vs_native_uniform=bt['elapsed_seconds']/at['elapsed_seconds'],
                news=dict(common_bondi_cuts=int(active.sum()),common_bondi_range_M0=[3*float(ta[active][0]),3*float(ta[active][-1])],
                          max_mode_difference=float(np.max(np.abs(error))),
                          rms_mode_difference=float(np.sqrt(np.mean(np.abs(error)**2))),
                          uniform_peak=float(np.max(np.abs(reference)))))
    report['initial_fields']=compare(dict(checkpoint=at['initial_checkpoint'],simulated_M0=0.),
                                     dict(checkpoint=bt['initial_checkpoint'],simulated_M0=0.))
    report['active_fields'],report['chi_floor']=active_field_errors(a,b)
    (root/'n12-10M'/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
