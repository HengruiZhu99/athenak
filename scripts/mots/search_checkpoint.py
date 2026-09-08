#!/usr/bin/env python3
"""Run one isolated frozen search; use --launcher for MPI/GPU launch commands."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--athena', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--launcher', default='', help='e.g. srun -n 2 -c 16')
    parser.add_argument('--lmax', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--radii', type=int, default=8)
    parser.add_argument('--radius-min', type=float, default=0.0)
    parser.add_argument('--radius-max', type=float, default=1.0)
    parser.add_argument('--axis-bound', type=float, default=8.0)
    parser.add_argument('--axis-samples', type=int, default=257)
    parser.add_argument('--seed', type=Path, help='center, coefficient count, then a_l0 values')
    parser.add_argument('--seed-only', action='store_true')
    parser.add_argument('--ntheta', type=int, default=0)
    parser.add_argument('--profile-points', type=int, default=0)
    parser.add_argument('--detection', choices=['strict','angular_candidate'], default='angular_candidate')
    parser.add_argument('--l-start', type=int, default=8)
    parser.add_argument('--level-iterations', type=int, default=96)
    args = parser.parse_args()
    if args.seed_only and not args.seed:
        parser.error('--seed-only requires --seed')
    if args.seed:
        args.seed = args.seed.resolve(strict=True)
    args.athena = args.athena.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    def git(*argv):
        proc = subprocess.run(['git', '-C', str(root), *argv], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else 'unavailable'
    revision = git('rev-parse', 'HEAD')
    if revision == 'unavailable' and (root/'source_revision.txt').exists():
        revision = (root/'source_revision.txt').read_text().strip()
    manifest = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
                'executable': str(args.athena), 'executable_sha256': sha(args.athena),
                'source_sha': revision, 'source_status': git('status', '--porcelain'),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'started': time.time(),
                'source_files': {str(p.relative_to(root)): sha(p) for p in sorted((root/'src').rglob('*')) if p.is_file()}}
    if args.seed:
        manifest['seed'] = {'path': str(args.seed), 'sha256': sha(args.seed),
                            'contents': args.seed.read_text()}
    overlay = args.output/'search.athinput'
    overlay.write_text(f'''<job>
basename = mots
<fastflow>
horizon_only = true
mots_detection = {args.detection}
mots_l_start = {min(args.l_start,args.lmax)}
mots_level_iterations = {args.level_iterations}
mots_candidate_bound = 0.01
mots_promotion_max = 0.5
mots_angular_ratio = 0.8
mots_candidate_points = 1061
mots_shape_change = 0.02
mots_area_change = 0.01
lmax = {args.lmax}
ntheta = {args.ntheta or 2*args.lmax+4}
flow_iterations_0 = {args.iterations}
mots_radius_count = {args.radii}
mots_radius_min = {args.radius_min}
initial_radius_0 = {args.radius_max}
cartoon_axis_search_bound_0 = {args.axis_bound}
cartoon_axis_search_samples_0 = {args.axis_samples}
mots_seed_file = {args.seed or ''}
mots_seed_only = {str(args.seed_only).lower()}
mots_profile_points = {args.profile_points}
mots_epsilon2 = 1e-6
mots_epsilon_inf = 1e-5
''')
    command = shlex.split(args.launcher) + [str(args.athena), '-r', str(args.checkpoint), '-i', str(overlay), '-d', str(args.output/'search')]
    manifest['command'] = command
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (args.output/'stdout.log').open('w') as out, (args.output/'stderr.log').open('w') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    manifest.update(finished=time.time(), returncode=result.returncode,
                    checkpoint_unchanged=sha(args.checkpoint)==manifest['checkpoint_sha256'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not manifest['checkpoint_unchanged']:
        raise RuntimeError('checkpoint changed during search')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
