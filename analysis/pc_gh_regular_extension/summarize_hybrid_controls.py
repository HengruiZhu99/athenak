"""Collect measured pulse convergence and strict-stop evidence without promotion."""
import argparse
import json
import math
import re
from pathlib import Path


def pulse_summary(root):
    result=[]
    for arm in ['C1','C4','C16','R4','R16','P1']:
        base=root/('pulses-projection' if arm=='P1' else 'pulses')/arm
        for family in ['p','Q','L','B']:
            cases=[base/f'{arm}-{family}-n{n}'/'hybrid-pulse-metrics.json' for n in [256,512,1024]]
            if not all(p.exists() for p in cases):
                result.append(dict(candidate=arm,family=family,status='incomplete'));continue
            rows=[json.loads(p.read_text()) for p in cases]
            item=dict(candidate=arm,family=family,files=[str(p) for p in cases])
            for sector in ['reductions','curls']:
                for norm in ['l2_over_amplitude','linf_over_amplitude']:
                    e=[r[sector][norm] for r in rows]
                    item[f'{sector}_{norm}']=e
                    item[f'{sector}_{norm}_orders']=[math.log(e[i]/e[i+1],2) for i in range(2)]
            item['projection_steps']=[r['projection_steps'] for r in rows]
            item['status']='measured; qualification requires all gates'
            result.append(item)
    return result


def stops(root):
    result=[]
    for group in ['reproduction','screen-controls','core-controls','screen-hybrids','core-hybrids']:
        for p in sorted((root/group).glob('**/provenance.json')):
            run=p.parent;logs=sorted(run.glob('segment-*.log'))
            lines=[]
            for log in logs:
                for line in log.read_text(errors='replace').splitlines():
                    if 'FATAL ERROR' in line or 'elapsed=' in line: lines.append(line)
            failures=[line for line in lines if 'FATAL ERROR' in line]
            last=next((line for line in reversed(lines) if 'elapsed=' in line),'')
            result.append(dict(run=str(run),completed=(run/'completed.json').exists(),
                               last_step=last,failure=failures[-1:] or None))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=dict(pulses=pulse_summary(a.root),punctures=stops(a.root))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['pulses']:
        print(r['candidate'],r['family'],r.get('reductions_l2_over_amplitude_orders'),
              r.get('curls_l2_over_amplitude_orders'))
    for r in result['punctures']: print(r['run'],r['completed'],r['last_step'],r['failure'])
