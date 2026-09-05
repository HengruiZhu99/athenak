"""Compact a finished hybrid monitor, preserving completed states and stage envelopes.

Only stage-3 operation -2 before is a completed-step sample for the fixed RK3
stress input. Raw clocks are step starts; completed state time is t_step+dt.
All operations remain represented in half-M stage envelopes, with winning rows.
Checkpoint rollback branches are discarded by a two-pass epoch audit. Raw input
is never changed. Do not invoke on a file still being written.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def reduce(path, output):
    stamp=(path.stat().st_size,path.stat().st_mtime_ns)
    starts=[]; previous=None; digest=hashlib.sha256(); count=0
    with path.open('rb') as f:
        header=f.readline();digest.update(header)
        for line in f:
            digest.update(line); count+=1
            a,b,_=line.split(b',',2);key=(int(a),float(b))
            if previous is None or key<previous:starts.append(key)
            previous=key
    cuts=[None]*len(starts);future=None
    for n in range(len(starts)-1,-1,-1):
        cuts[n]=future
        future=starts[n] if future is None else min(future,starts[n])
    output.mkdir(parents=True,exist_ok=True)
    names=header.decode().strip().split(','); envelope={};discarded=0; kept=0
    pending={};changes={};unpaired_after=0
    epoch=-1;previous=None
    with path.open() as f,gzip.open(output/'completed-monitor.csv.gz','wt') as dst:
        next(f);dst.write(header.decode())
        for line in f:
            v=line.rstrip().split(',');key=(int(v[0]),float(v[1]))
            if previous is None or key<previous:epoch+=1
            previous=key
            if cuts[epoch] is not None and key>=cuts[epoch]:discarded+=1;continue
            if int(v[14])<0:continue
            if int(v[4])>=0 and v[4]!='100' and not v[7].startswith(('min_','delta_')):
                pair=(key,v[3],v[4],v[6],v[7])
                if v[5]=='before':pending[pair]=v
                else:
                    before=pending.pop(pair,None)
                    if before is None:unpaired_after+=1
                    else:
                        delta=float(v[8])-float(before[8])
                        token=(int(float(v[1])*2),v[3],v[4],v[6],v[7])
                        prior=changes.get(token)
                        if prior is None or delta>prior[0]:changes[token]=(delta,before,v)

            if v[3]=='3' and v[4]=='-2' and v[5]=='before':
                dst.write(line);kept+=1
            # Explicitly use step-start bins for stage states: they are not
            # claimed to lie at a single physical time during a multistage RK.
            token=(int(float(v[1])*2),v[3],v[4],v[5],v[6],v[7])
            old=envelope.get(token);minimum=v[7].startswith('min_')
            if old is None or (float(v[8])<float(old[8]) if minimum else float(v[8])>float(old[8])):
                envelope[token]=v
    with gzip.open(output/'stage-envelopes.csv.gz','wt') as f:
        w=csv.writer(f);w.writerow(['half_M_bin']+names)
        for key,row in sorted(envelope.items()):w.writerow([key[0]]+row)
    with gzip.open(output/'operation-norm-increases.csv.gz','wt') as f:
        w=csv.writer(f)
        w.writerow(['half_M_bin','stage','operation','region','quantity','delta_max_norm',
                    'before_max','after_max','cycle','t_step','dt',
                    'before_x','before_y','before_z','after_x','after_y','after_z'])
        for key,(delta,before,after) in sorted(changes.items()):
            w.writerow(list(key)+[delta,before[8],after[8],after[0],after[1],after[2]]+before[10:13]+after[10:13])
    if stamp!=(path.stat().st_size,path.stat().st_mtime_ns):
        raise RuntimeError('Source changed during reduction; outputs are invalid')
    audit=dict(source=str(path),source_bytes=stamp[0],source_sha256=digest.hexdigest(),
        raw_rows=count,epoch_starts=starts,epoch_cutoffs=cuts,discarded_rows=discarded,
        completed_rows=kept,envelope_rows=len(envelope),scope=__doc__,
        operation_norm_scope='Largest paired difference of maximum norms in each bin/stage/operation/region/quantity. This is not the norm of a vector correction, and before/after maxima can have different locations.',
        operation_bins=len(changes),unpaired_after=unpaired_after,unpaired_before=len(pending))
    (output/'monitor-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    p.add_argument('output',type=Path);a=p.parse_args();reduce(a.path,a.output)
