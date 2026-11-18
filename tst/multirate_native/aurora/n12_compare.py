"""Compare the two completed 10 M0 N12 runs at their synchronized endpoint."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from compare import compare
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from endpoint_diagnostics import diagnostics
from scri_collect import collect
from scri_news import analyze


def work_and_science(root, mode):
    case = root/'n12-10M'/mode
    last = json.loads((case/'state.json').read_text())
    if not last['target_reached'] or abs(last['time_M0']-10)>last['cycle']*math.ulp(10.):
        raise ValueError('N12 mode has not reached the common 10 M0 endpoint: '+mode)
    receipts = sorted((json.loads(p.read_text()) for p in case.glob('segment-*/receipt.json')),
                      key=lambda x:x['start_cycle'])
    next_cycle = 0
    for item in receipts:
        if item['start_cycle'] != next_cycle or item['binary_sha256'] != last['binary_sha256']:
            raise ValueError('N12 segment provenance or cycle gap in '+mode)
        next_cycle = item['cycle']
    if next_cycle != last['cycle']:
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
                                   area=float(row[7]),expansion_rms=float(row[8]))
    analysis_dir = root/'n12-10M'/'analysis'/mode
    cache = analysis_dir/'cache'
    records = collect(case,cache,capacity=512)
    if abs(3*records[-1]['time']-10)>last['cycle']*math.ulp(10.):
        raise ValueError('waveform series does not reach the final cut')
    analyze(cache,analysis_dir/'news',lmax=8)
    with np.load(analysis_dir/'news'/'news.npz') as archive:
        news = {k:archive[k] for k in ('bondi_time','ell_m','news_modes')}
    wall = sum(item['wall_seconds'] for item in receipts)
    return (dict(**last,simulated_M0=10.),ids,endpoint,news,
            dict(segments=len(receipts),elapsed_seconds=wall,seconds_per_M0=wall/10,
                 physical_gpus=12,nodes=2,scri_cuts=len(records),last_reported_horizons=horizons))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    a,aid,ae,an,at=work_and_science(root,'multirate')
    b,bid,be,bn,bt=work_and_science(root,'uniform')
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
                           uniform_rms=float(np.sqrt(np.mean(be[key]**2))))
    report=dict(fields=compare(a,b),endpoint=endpoint,uniform=bt,multirate=at,
                elapsed_speedup_vs_native_uniform=bt['elapsed_seconds']/at['elapsed_seconds'],
                news=dict(common_bondi_cuts=int(active.sum()),common_bondi_range=[float(ta[active][0]),float(ta[active][-1])],
                          max_mode_difference=float(np.max(np.abs(error))),
                          rms_mode_difference=float(np.sqrt(np.mean(np.abs(error)**2))),
                          uniform_peak=float(np.max(np.abs(reference)))))
    (root/'n12-10M'/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
