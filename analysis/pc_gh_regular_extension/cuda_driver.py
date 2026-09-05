"""Build/run one explicit qualification input on Della CUDA, preserving evidence.

This runner intentionally does not infer that a successful exit passes a physics
qualification gate. Select later inputs only after reviewing earlier evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from make_inputs import parse, write

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(argv, cwd, log):
    with log.with_suffix(log.suffix+'.command.json').open('w') as file:
        json.dump(dict(argv=[str(x) for x in argv], cwd=str(cwd),
                       started=datetime.now(timezone.utc).isoformat()), file, indent=2)
    with log.open('x') as file:
        result = subprocess.run([str(x) for x in argv], cwd=cwd, stdout=file,
                                stderr=subprocess.STDOUT)
    print(f'{log}: exit={result.returncode}', flush=True)
    return result.returncode


def gpu_guard():
    host = socket.gethostname()
    if not (host.split('.')[0] == 'della-vis1' or os.environ.get('SLURM_JOB_ID')):
        raise SystemExit('Evolution runner requires della-vis1 or a Slurm GPU allocation')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used',
                             '--format=csv'], check=True, capture_output=True, text=True)
    return dict(host=host, gpu=result.stdout, slurm_job=os.environ.get('SLURM_JOB_ID'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def build(args):
    # Compilation can take place on a Della login node with the user's CUDA env.
    out = args.build.resolve()
    out.mkdir(parents=True, exist_ok=False)
    problem = {'oracle': '../../analysis/pc_gh_regular_extension/production_oracle',
               'z4c': 'z4c_one_puncture', 'binary': 'z4c_two_puncture'}[args.kind]
    wrapper = ROOT/'kokkos/bin/nvcc_wrapper'
    if not wrapper.is_file() or shutil.which('nvcc') is None:
        raise SystemExit('Load the Della CUDA/compiler environment and initialize Kokkos first')
    configure = ['cmake', '-S', ROOT, '-B', out, '-DPROBLEM='+problem,
                 '-DCMAKE_CXX_COMPILER='+str(wrapper), '-DCMAKE_BUILD_TYPE=Release',
                 '-DKokkos_ENABLE_CUDA=ON', '-DKokkos_ARCH_AMPERE80=ON',
                 '-DAthena_ENABLE_MPI=OFF', '-DAthena_ENABLE_OPENMP=OFF']
    if command(configure, ROOT, out/'configure.log'):
        raise SystemExit(1)
    if command(['cmake', '--build', out, '-j', str(args.jobs)], ROOT, out/'build.log'):
        raise SystemExit(1)
    binary = out/'src/athena'
    (out/'binary.sha256').write_text(sha(binary)+'  src/athena\n')
    command([binary, '-c'], ROOT, out/'executable-config.log')


def run(args):
    hardware = gpu_guard()
    build_dir = args.build.resolve()
    cache = (build_dir/'CMakeCache.txt').read_text()
    if 'Kokkos_ENABLE_CUDA:BOOL=ON' not in cache:
        raise SystemExit('Refusing a build without Kokkos CUDA enabled')
    binary = build_dir/'src/athena'
    destination = args.output.resolve()
    binary_hash = sha(binary)
    if args.resume:
        meta = json.loads((destination/'provenance.json').read_text())
        if meta['binary_sha256'] != binary_hash:
            raise SystemExit('Executable changed: use a distinct, documented restart experiment')
        if sha(destination/'used_input.athinput') != meta['input_sha256']:
            raise SystemExit('Input changed since the preserved experiment began')
    else:
        destination.mkdir(parents=True, exist_ok=False)
        params = parse(args.input.resolve().read_text())
        params['job']['basename'] = destination.name
        # Custom research pgen checks the *actual execution space* at startup.
        if params['problem']['pgen_name'] not in ['z4c_one_puncture', 'z4c_two_puncture']:
            params['problem']['require_cuda'] = 'true'
        write(destination/'used_input.athinput', params)
        shutil.copy2(build_dir/'CMakeCache.txt', destination/'CMakeCache.txt')
        meta = dict(**hardware, binary=str(binary), binary_sha256=binary_hash,
                    input_sha256=sha(destination/'used_input.athinput'),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                         cwd=ROOT, text=True).strip(),
                    created=datetime.now(timezone.utc).isoformat())
        (destination/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
        (destination/'source-diff.patch').write_bytes(subprocess.check_output(
            ['git', 'diff', 'HEAD', '--', 'src', 'analysis/pc_gh_regular_extension'], cwd=ROOT))
        (destination/'source-status.txt').write_bytes(subprocess.check_output(
            ['git', 'status', '--short'], cwd=ROOT))
    params = parse((destination/'used_input.athinput').read_text())
    segment = 0
    while True:
        segment += 1
        log = destination/f'segment-{segment:04d}.log'
        if log.exists():
            continue
        argv = [binary, '-i', destination/'used_input.athinput', '-t', args.wall_segment]
        restarts = sorted(destination.glob('rst/*.rst'), key=lambda p: p.stat().st_mtime_ns)
        if restarts:
            argv += ['-r', restarts[-1]]
        status = command(argv, destination, log)
        text = log.read_text(errors='replace')
        if status:
            raise SystemExit(status)
        is_oracle = int(params['time'].get('nlim', '-1')) == 0
        if 'Terminating on time limit' in text or (is_oracle and 'Terminating on cycle limit' in text):
            (destination/'completed.json').write_text(json.dumps(dict(
                segment=segment, numerical_exit='clean',
                qualification='requires physics analysis; survival is not acceptance'), indent=2)+'\n')
            return
        if 'Terminating on wall clock limit' not in text:
            raise SystemExit('Unrecognized termination: inspect the preserved log')
        if not list(destination.glob('rst/*.rst')):
            raise SystemExit('Clean wall stop without a restart; shorten the explicit test or add checkpoints')
        if params['problem']['pgen_name'] == 'z4c_one_puncture':
            # Existing control pgen ignores the restart flag and reinitializes.
            raise SystemExit('Z4c single-puncture pgen is not restart-safe; preserve this partial run')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('--kind', choices=['oracle', 'z4c', 'binary'], required=True)
    b.add_argument('--build', type=Path, required=True)
    b.add_argument('--jobs', type=int, default=8)
    r = sub.add_parser('run')
    r.add_argument('--build', type=Path, required=True)
    r.add_argument('--input', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--wall-segment', default='00:15:00')
    args = ap.parse_args()
    (build if args.action == 'build' else run)(args)


if __name__ == '__main__':
    main()
