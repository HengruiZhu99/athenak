#!/usr/bin/env python3
"""Run a bounded, isolated live-AMR restart to test the runtime MOTS finder."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['athena','checkpoint','amr-history','output']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--tlim',required=True,type=float)
    p.add_argument('--cycle-limit',required=True,type=int)
    p.add_argument('--launcher',default='')
    p.add_argument('--cadence',type=int,default=4)
    p.add_argument('--lmax',type=int,default=128)
    p.add_argument('--l-start',type=int,default=8)
    p.add_argument('--radii',type=int,default=4)
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--outer05', action='store_true')
    p.add_argument('--ce-bracket', action='store_true')
    p.add_argument('--ce-steps', type=int, default=4)
    p.add_argument('--ce-iterations', type=int, default=96)
    p.add_argument('--checkpoint-cadence', type=int, default=0)

    args=p.parse_args()
    for name in ['athena','checkpoint','amr_history']:
        setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    run=args.output/'run';run.mkdir()
    # A saved slice authenticates the ledger prefix at its own checkpoint time.
    # Copy exactly that prefix; later records belong to the production future.
    with args.checkpoint.open('rb') as stream:
        header = stream.read(4*1024*1024).split(b'<par_end>')[0].decode('utf-8')
    carrier = re.search(r'<amr_history_restart>\s*(.*?)(?=\n<|\Z)', header, re.S)
    if not carrier:
        raise RuntimeError('Missing AMR restart carrier')
    fields = dict(re.findall(r'^\s*(\w+)\s*=\s*(\S+)', carrier[1], re.M))
    if fields.get('mode') != 'record':
        raise RuntimeError('Expected live AMR record mode')
    size = int(fields['history_bytes'])
    with args.amr_history.open('rb') as stream:
        prefix = stream.read(size)
    digest = 14695981039346656037
    for byte in prefix:
        digest = ((digest ^ byte)*1099511628211) & ((1 << 64)-1)
    if len(prefix) != size or f'{digest:016x}' != fields['history_digest']:
        raise RuntimeError('AMR ledger prefix does not match checkpoint')
    (run/'amr_history.jsonl').write_bytes(prefix)
    overlay=args.output/'restart.athinput'
    overlay.write_text(f'''<job>
basename = runtime_mots
<time>
tlim = {args.tlim}
nlim = {args.cycle_limit}
ndiag = 1
<mesh_refinement>
amr_history_file = {run/'amr_history.jsonl'}
<fastflow>
horizon_only = false
mots_detection = angular_candidate
mots_level_iterations = 96
mots_candidate_bound = 0.01
mots_promotion_max = 0.5
mots_angular_ratio = 0.8
mots_candidate_points = 1061
mots_shape_change = 0.02
mots_area_change = 0.01
lmax = {args.lmax}
ntheta = {2*args.lmax+4}
mots_l_start = {args.l_start}
flow_iterations_0 = {args.iterations}
find_interval_0 = {args.cadence}
mots_radius_count = {args.radii}
initial_radius_0 = 1.0
mots_epsilon2 = 1e-6
mots_epsilon_inf = 1e-5
mots_write_profiles = true
mots_profile_points = 1061
mots_discovery_interval = 8
mots_tracking_residual = 0.01
''')
    if args.outer05:
        if args.lmax != 64:
            p.error('--outer05 requires --lmax 64')
        text=overlay.read_text().replace('mots_detection = angular_candidate',
             'mots_detection = angular_l32\nmots_selection = outermost\n'
             'mots_radius_min = 0\ncartoon_axis_search_bound_0 = 8\n'
             'cartoon_axis_search_samples_0 = 257\nmots_seed_file = \nmots_seed_only = false')
        text=text.replace('mots_candidate_bound = 0.01','mots_candidate_bound = 0.05')
        text=text.replace('initial_radius_0 = 1.0','initial_radius_0 = 8')
        overlay.write_text(text)
    if args.ce_bracket:
        if not args.outer05 or args.ce_steps < 1 or args.ce_iterations < 1:
            p.error('--ce-bracket requires --outer05 and positive CE limits')
        with overlay.open('a') as f:
            f.write(f'ce_target = 0\nce_reference_radius = 0\nce_dense_points = 1061\nce_bracket = true\nce_steps = {args.ce_steps}\nce_iterations = {args.ce_iterations}\n')
            f.write('<problem>\nstop_on_horizon = false\nstop_on_mots_bracket = true\nstop_on_dispersion = false\n')
    if args.checkpoint_cadence:
        if args.checkpoint_cadence < 1:
            p.error('--checkpoint-cadence must be positive')
        blocks=re.findall(r'<(output\d+)>\s*(.*?)(?=\n<|\Z)',header,re.S)
        restart_blocks=[name for name,body in blocks if re.search(r'^\s*file_type\s*=\s*rst\s*(?:#.*)?$',body,re.M)]
        if len(restart_blocks)!=1:
            raise RuntimeError('Expected exactly one saved restart output block')
        with overlay.open('a') as f:
            f.write(f'<{restart_blocks[0]}>\ndcycle = {args.checkpoint_cadence}\n')
    # Preserve the restart's live-AMR policy and all evolution/physical parameters.
    # The only AMR change is the destination of its append-only history.
    command=shlex.split(args.launcher)+[str(args.athena),'-r',str(args.checkpoint),
                                      '-i',str(overlay),'-d',str(run)]
    source=Path(__file__).resolve().parents[2]
    revision_file=source/'source_revision.txt'
    revision=(revision_file.read_text().strip() if revision_file.exists() else
              subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip())
    m=dict(source_sha=revision,command=command,started=time.time(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
           checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
           executable=str(args.athena),executable_sha256=sha(args.athena),
           original_amr_history=str(args.amr_history),original_amr_history_sha256=sha(args.amr_history),
           copied_amr_history_bytes=size, copied_amr_history_digest=fields['history_digest'],
           overlay_sha256=sha(overlay),source_files={str(f.relative_to(source)):sha(f)
               for f in sorted((source/'src').rglob('*')) if f.is_file()})
    manifest=args.output/'manifest.json';manifest.write_text(json.dumps(m,indent=2)+'\n')
    with (args.output/'stdout.log').open('w') as out,(args.output/'stderr.log').open('w') as err:
        result=subprocess.run(command,stdout=out,stderr=err)
    m.update(finished=time.time(),returncode=result.returncode,
             checkpoint_unchanged=sha(args.checkpoint)==m['checkpoint_sha256'],
             original_amr_history_unchanged=sha(args.amr_history)==m['original_amr_history_sha256'])
    manifest.write_text(json.dumps(m,indent=2)+'\n')
    if not m['checkpoint_unchanged'] or not m['original_amr_history_unchanged']:
        raise RuntimeError('Production input changed during isolated test')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
