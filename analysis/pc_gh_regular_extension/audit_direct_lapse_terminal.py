"""Audit a terminal stress run and locate its failed cell from saved static-mesh headers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def mesh_header(p):
    # Same verified little-endian vacuum PC-GH header ABI as sample_restart_sources.py.
    with p.open('rb') as f:
        prefix=f.read(65536);end=prefix.find(b'<par_end>\n');assert end>=0
        f.seek(end+10);blocks,root=struct.unpack('<ii',f.read(8))
        geometry=struct.unpack('<9d',f.read(72));mesh=struct.unpack('<19i',f.read(76));block=struct.unpack('<19i',f.read(76))
        time,dt,cycle=struct.unpack('<ddi',f.read(20));raw=f.read(blocks*16)
        locations=list(struct.iter_unpack('<4i',raw));assert len(locations)==blocks
    return dict(path=str(p),blocks=blocks,root_level=root,geometry=geometry,mesh_counts=mesh[1:4],
                ng=block[0],block_counts=block[1:4],time=time,dt=dt,cycle=cycle,
                location_sha256=hashlib.sha256(raw).hexdigest()),locations


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);a=p.parse_args()
    run=a.root/'stress/core256-R16';meta=json.loads((run/'provenance.json').read_text())
    exit_record=json.loads((a.root/'stress-exit.json').read_text())
    logs=sorted(run.glob('segment-*.log'));segments=[];fatals=[]
    for log in logs:
        text=log.read_text();found=[s for s in text.splitlines() if 'FATAL ERROR' in s];fatals.extend(found)
        segments.append(dict(log=str(log),sha256=sha(log),wall_stop='Terminating on wall clock limit' in text,
            fatal=found,command=json.loads(log.with_suffix('.log.command.json').read_text())))
    assert exit_record['exit_code']!=0 and fatals and not (run/'completed.json').exists()
    assert all(s['wall_stop'] and not s['fatal'] for s in segments[:-1])
    failure_time=float(re.search(r'failed at t=([\deE+.-]+)',fatals[-1])[1])
    cell=tuple(map(int,re.search(r'\(m,k,j,i\)=\((\d+),(\d+),(\d+),(\d+)\)',fatals[-1]).groups()))
    checks=[max(r.glob('rst/*.rst'),key=lambda x:x.stat().st_mtime_ns) for r in [a.baseline,run]]
    headers=[mesh_header(p) for p in checks]
    for key in ['blocks','root_level','geometry','mesh_counts','ng','block_counts','location_sha256']:
        assert headers[0][0][key]==headers[1][0][key],key
    h,locations=headers[1];m,k,j,i=cell;loc=locations[m];xyz=[];spacing=[]
    for d,index in enumerate([i,j,k]):
        root_blocks=h['mesh_counts'][d]/h['block_counts'][d]
        width=(h['geometry'][d+3]-h['geometry'][d])/(root_blocks*2**(loc[3]-h['root_level']))
        lower=h['geometry'][d]+loc[d]*width;dx=width/h['block_counts'][d]
        xyz.append(lower+(index-h['ng']+.5)*dx);spacing.append(dx)
    source=json.loads((a.root/'local-source-sha256.json').read_text())
    assert all(sha(a.root/'source'/p)==v for p,v in source.items())
    binary=Path(meta['binary']);assert sha(binary)==meta['binary_sha256']
    assert sha(run/'used_input.athinput')==meta['input_sha256']
    assert (run/'used_input.athinput').read_bytes()==(a.baseline/'used_input.athinput').read_bytes()
    result=dict(decision='FAIL',requested_target=6.,failure_time=failure_time,baseline_failure_time=5.187818,
        lifetime_difference=failure_time-5.187818,fatal_errors=fatals,exit_record=exit_record,
        segments=segments,head_driver_pid=int((a.root/'stress-driver.pid').read_text()),slurm_job=meta['slurm_job'],
        binary_sha256=meta['binary_sha256'],input_sha256=meta['input_sha256'],production_source_files_reverified=len(source),
        static_mesh_headers=[x[0] for x in headers],failed_cell=dict(index_mkji=cell,xyz=xyz,spacing=spacing,physical_level=loc[3]-h['root_level'],
        scope='Coordinates reconstructed from identical static logical block maps in valid checkpoints; the failed state itself was not checkpointed.'),
        downstream=dict(convergence='NOT RUN: stress gate failed',binary='NOT RUN: stress gate failed'))
    (a.root/'terminal-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
