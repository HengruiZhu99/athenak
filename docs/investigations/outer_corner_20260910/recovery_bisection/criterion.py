"""First early lapse event terminates; otherwise classify at t200."""
import math,re
from decimal import Decimal

def read_history(path):
    header=None;rows=[]
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            if '[1]=' in line:
                header={name:int(i)-1 for i,name in re.findall(r'\[(\d+)\]=(\S+)',line)}
            continue
        if not line.strip():continue
        values=[float(x) for x in line.split()]
        if header is None or len(values)!=len(header) or not all(map(math.isfinite,values)):
            raise ValueError('Invalid or nonfinite evolution history')
        row={k:values[i] for k,i in header.items()}
        if 'minLapse' not in row or 'time' not in row:raise ValueError('Global minimum lapse/time missing')
        if rows and row['time']<rows[-1]['time']:raise ValueError('History time decreased')
        rows.append(row)
    if not rows:raise ValueError('No history data')
    return rows

def first_early_crossing(rows):
    if not rows:raise ValueError('No history data')
    for row in rows:
        if not math.isfinite(row['time']) or not math.isfinite(row['minLapse']) or row['minLapse']<=0:
            raise ValueError('Invalid evolution history; no classification')
    return next((row for row in rows if row['minLapse']<1e-5),None)

def first_early_event(rows):
    first_early_crossing(rows)  # Validate all supplied data before classifying.
    dipped=False
    for row in rows:
        lapse=row['minLapse']
        if lapse<1e-5:return dict(classification='collapse',reason='early_global_lapse',sample=row)
        if lapse<.1:dipped=True
        if dipped and lapse>.8:return dict(classification='disperse',reason='early_lapse_recovery',sample=row)
    return None

def classify(rows,target_time):
    if not math.isfinite(target_time) or target_time<=0:raise ValueError('Invalid target time')
    event=first_early_event(rows)
    if event:return event['classification']
    final=rows[-1]
    if abs(final['time']-target_time)>1e-8*max(1,target_time):
        raise ValueError('Run did not reach requested time or an early event; do not update bracket')
    return 'collapse' if final['minLapse']<.01 else 'disperse'

def relative_width(sub,supercritical):
    a,b=Decimal(sub),Decimal(supercritical)
    if not a.is_finite() or not b.is_finite() or not b<a<0:raise ValueError('Invalid negative-amplitude bracket')
    return abs(a-b)/abs(b)
