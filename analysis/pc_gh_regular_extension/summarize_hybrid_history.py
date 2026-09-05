"""Reduce uncensored stage/transfer CSVs, preserving extreme locations and jumps."""
import argparse
import csv
import json
from pathlib import Path


def summarize(run):
    files=list(run.glob('*.hybrid.csv'))
    assert len(files)==1,files
    envelope={};jumps=[];bracket=False;seen=set()
    with files[0].open() as stream:
        for row in csv.DictReader(stream):
            if row['region']=='all' and row['operation']=='3' and row['quantity']=='Rw':
                bracket=row['phase']=='before'
            if row['quantity'].startswith('delta_'):
                if row['operation']=='100' or (row['operation']=='8' and bracket):
                    key=tuple(row[k] for k in ['cycle','t_step','region','quantity'])
                    assert key not in seen,('duplicate jump',key)
                    seen.add(key);jumps.append(row)
                continue
            if int(row['block'])<0: continue
            key=tuple(row[k] for k in ['cycle','t_step','region','quantity'])
            old=envelope.get(key);value=float(row['max'])
            minimum=row['quantity'].startswith('min_')
            if old is None or (value<float(old['max']) if minimum else value>float(old['max'])):
                envelope[key]=row
    fields=list(next(iter(envelope.values())))
    with (run/'hybrid-stage-envelope.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(envelope.values())
    cumulative={}
    for row in jumps:
        key=row['region']+':'+row['quantity']
        cumulative[key]=cumulative.get(key,0)+float(row['coordinate_l1'])
    last_time=max(float(row['t_step']) for row in envelope.values())
    jump_times=[float(r['t_step'])+float(r['dt']) for r in jumps if r['region']=='all' and r['quantity']=='delta_p']
    duration=jump_times[-1] if jump_times else last_time
    result=dict(run=str(run),stage_envelope_rows=len(envelope),last_step_start=last_time,
                projection_steps=len(jump_times),correction_duration=duration,
                cumulative_coordinate_l1=cumulative,
                cumulative_coordinate_l1_per_time={k:v/duration for k,v in cumulative.items()} if duration>0 else {},
                scope='Envelope over RK and transfer states; extrema do not alone identify a causal source. Legacy operation-8 boundary mislabels are excluded.')
    (run/'hybrid-history-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path,nargs='+')
    for run in ap.parse_args().runs:summarize(run)
