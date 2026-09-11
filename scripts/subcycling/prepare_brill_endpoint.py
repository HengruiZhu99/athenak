"""Prepare isolated endpoint comparisons; does not submit or run simulations.

Uses the recovered campaign's checkpoint parser and parameter editor. Preserves
production files; copies only the checkpoint-compatible AMR history prefix.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--exe',type=Path,required=True)
p.add_argument('--ratio',type=int,default=16)
p.add_argument('--interval-cap',type=float,default=4e-5)
a=p.parse_args()
assert a.ratio in [2,4,8,16,32] and 0<a.interval_cap<1
campaign=a.campaign.resolve();output=a.output.resolve();exe=a.exe.resolve()
sys.path.insert(0,str(campaign))
from resume_case import checkpoint_info,sha
from workflow_common import setparam
source=campaign/'cycle_03_recovery_24000'
checkpoint=source/'rst/lapse200.00066.rst';info=checkpoint_info(checkpoint)
# Derive target from actual final history sample, never the eight-hour heuristic.
histories=sorted(source.glob('*.hst'));assert len(histories)==1
last=None;columns=None
with histories[0].open() as f:
    for line in f:
        if line.startswith('#'):
            names=re.findall(r'\[\d+\]=([^\s]+)',line)
            if names:
                assert columns is None or columns==names
                columns=names
        elif line.strip():last=line
assert last is not None and columns is not None
assert len(columns)==len(last.split())
original_final=dict(zip(columns,map(float,last.split())))
end=float(last.split()[0]);start=float(info['time']);assert start<end<start+.01
output.mkdir(parents=True,exist_ok=False)
with (source/'amr_history.jsonl').open('rb') as f:prefix=f.read(info['history_bytes'])
assert len(prefix)==info['history_bytes'] and prefix.endswith(b'\n')
recorded_source=json.loads(prefix.splitlines()[0])['source_id']
base=(source/'input.athinput').read_text()
manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
    initial_info=info,start=start,end=end,source_history=str(histories[0]),
    original_final_history_row=last.strip(),original_final=original_final,exe=str(exe),exe_sha256=sha(exe),cases=[])
# Keep integrator changes separate from time-subcycling effects. All cases
# use the same current executable; legacy_sync preserves its existing RK mode,
# not the historical production source revision.
for name,integrator,ratio in [('legacy_sync','rk4',0),
                              ('classical_sync','rk4_classical',0),
                              ('subcycled','rk4_classical',a.ratio)]:
    case=output/name;case.mkdir();(case/'amr_history.jsonl').write_bytes(prefix)
    text=base
    parameters=[('time','integrator',integrator),('time','nlim',-1),('time','tlim',end),
        ('time','subcycle_max_ratio',ratio),('time','subcycle_interval_cap',a.interval_cap),
        ('time','subcycle_cycle_unit','synchronization'),('time','ndiag',4),
        ('mesh_refinement','amr_history_mode','record'),
        ('mesh_refinement','amr_history_file',case/'amr_history.jsonl'),
        ('mesh_refinement','amr_history_compatible_source_id',recorded_source),
        ('mesh_refinement','max_nmb_per_rank',24000),
        ('problem','brill_global_coefficients_file',source/'initial.coefficients')]
    for section,key,value in parameters:text=setparam(text,section,key,value)
    # Same requested physical output cadence; synchronous outputs may overshoot
    # by a finest step. Both must reach the exact comparison endpoint.
    sections=re.findall(r'^<(output[^>]*)>',text,re.M)
    for section in sections:
        for key,value in [('cadence','time'),('dcycle',0),('dt',(end-start)/8),('last_time',start),('file_number',0)]:
            text=setparam(text,section,key,value)
    (case/'input.athinput').write_text(text)
    manifest['cases'].append(dict(name=name,directory=str(case),ratio=ratio,integrator=integrator,
        input_sha256=sha(case/'input.athinput'),amr_prefix_sha256=hashlib.sha256(prefix).hexdigest()))
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(output/'manifest.json')
