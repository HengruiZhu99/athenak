#!/usr/bin/env python3
"""Compare full active-leaf field dumps from the same hierarchy test revision."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cpu', type=Path)
parser.add_argument('gpu', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
cpu_sha = (args.cpu / 'source-sha.txt').read_text().strip()
gpu_sha = (args.gpu / 'source-sha.txt').read_text().strip()
if cpu_sha != gpu_sha:
    raise RuntimeError('CPU/GPU source revisions differ')
rows = []
for case in ['corrector3', 'three-level', 'coarse-group', 'time-dependent', 'cpbc']:
    for steps in [4, 8, 16, 32]:
        name = f'{case}-values-{steps}.txt'
        cpu = list(map(float, (args.cpu / name).read_text().split()))
        gpu = list(map(float, (args.gpu / name).read_text().split()))
        if not cpu or len(cpu) != len(gpu):
            raise RuntimeError(f'Missing/mismatched fields: {name}')
        if not all(math.isfinite(x) for x in cpu + gpu):
            raise RuntimeError(f'Nonfinite fields: {name}')
        differences = [a-b for a, b in zip(cpu, gpu)]
        rows.append(dict(case=case, coarse_steps=steps, field_values=len(cpu),
                         max_abs_difference=max(map(abs, differences)),
                         rms_difference=math.sqrt(math.fsum(x*x for x in differences)/len(cpu))))
result = dict(source_sha=cpu_sha, coordinate_time=0.04, comparisons=rows,
              scope='Fixed hierarchy vacuum VC Cartoon small lapse pulse; prescribed gauge history. Not a production checkpoint comparison.')
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
