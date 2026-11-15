#!/usr/bin/env python3
"""Prepare isolated native checkpoints for matched A/B smooth-pulse experiments."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from restart_roundtrip import parse
p = argparse.ArgumentParser()
p.add_argument('--writer', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
out, exe = args.output.resolve(), args.writer.resolve()
out.mkdir(parents=True, exist_ok=False)
manifest = []
for n in (12, 16, 20):
    for mode in ('A', 'B'):
        variants = [('control', 0), ('pulse', 1e-6)]
        if n == 12:
            variants.append(('halfamp', 5e-7))
        for variant, amplitude in variants:
            name = f'{mode}-N{n}-{variant}'
            d = out/name; d.mkdir()
            text = (root/f'build-scri-experiment/inputs/{mode}-N{n}.athinput').read_text()
            text += f'\n<job>\nbasename = scri_{name}\n<time>\nnlim = {1422*n//12}\ntlim = 1\n'
            text += '<fastflow>\nnum_horizons = 0\n<output1>\ndt = .0033333333333333333\n<output2>\ndt = 100\n'
            inp = d/'input.athinput'; inp.write_text(text)
            command = ['mpiexec', '-n', '2', str(exe), str(inp), str(amplitude)]
            print('Preparing', name, flush=True)
            with (d/'seed.log').open('w') as log:
                subprocess.run(command, cwd=d, stdout=log, stderr=subprocess.STDOUT,
                               timeout=900, check=True)
            checkpoint = next((d/'rst').glob('*.rst'))
            metadata = parse(checkpoint)
            assert metadata['cycle'] == 0 and metadata['time'] == 0
            manifest.append(dict(case=name, n=n, mode=mode, amplitude=amplitude,
                checkpoint=str(checkpoint.relative_to(out)),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), command=command,
                end_M0=3*metadata['dt']*(1422*n//12)))
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Prepared', len(manifest), 'pulse/control checkpoints', flush=True)
