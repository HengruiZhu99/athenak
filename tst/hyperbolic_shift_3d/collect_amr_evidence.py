#!/usr/bin/env python3
"""Collect bounded qualification receipts without committing large binary states."""
import argparse,hashlib,json,pathlib,shutil
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--evidence',required=True);a=p.parse_args();runs=pathlib.Path(a.runs);out=pathlib.Path(a.evidence);out.mkdir(parents=True,exist_ok=True)
receipts=[];summaries={}
for name in ['matrix-v1','restart-v1','boost-v1','controls-v1','equilibrium-v1','checkpoint-rejection','checkpoint-rejection-v2']:
    source=runs/name;dest=out/name;dest.mkdir(exist_ok=True)
    for f in sorted(source.iterdir()):
        if f.suffix in ['.json','.jsonl','.log']:shutil.copy2(f,dest/f.name)
        if f.suffix in ['.bin','.checkpoint']:
            digest=hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
            receipts.append(dict(path=str(f.resolve()),bytes=f.stat().st_size,sha256=digest.hexdigest()))
        if f.name.endswith('.history.jsonl'):
            rows=[json.loads(line) for line in f.read_text().splitlines()]
            summaries[name+'/'+f.name]=dict(records=len(rows),final=rows[-1],min_chi=min(r['chi'] for r in rows),min_lapse=min(r['lapse'] for r in rows),max_det_residual=max(r['det_residual'] for r in rows),max_null=max(max(r['null'],r['ghost_null']) for r in rows),max_cfl=max(r['cfl'] for r in rows),horizon_rejected_records=sum(not r['horizon_found'] for r in rows) if 'boost' in f.name else None)
(out/'binary-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
(out/'evolution-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
summary=dict(status='experimental integration; dynamic boosted-puncture qualification fails',robust_unrestricted_boost_amr=False,backend='CPU Kokkos SERIAL',ranks=[1,2,4],dynamic_wave_restart='byte-identical across one-rank uninterrupted, two-rank uninterrupted, and two-to-four-rank restart',dynamic_boost='first regrid rejected: native point prolongation makes chi negative from positive donors',time_limit=dict(steps=6,dt=.00025,t_over_M=.0045),limitations=['No global mixed-level KO energy estimate','Native point transfer is fifth-order in field values; differentiated regrid defects can be lower order','Replicated host plans and complete-state regrid gathers','No GPU evidence, long-time stability or evolution convergence proof','Prescribed regrid schedule; production refinement estimator not qualified','No nonzero floor-event restart fixture'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
