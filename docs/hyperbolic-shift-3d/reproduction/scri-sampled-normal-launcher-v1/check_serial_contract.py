#!/usr/bin/env python3
"""Read the actual local Serial log without relabeling it GPU evidence."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import preflight

source=HERE/'serial-component-ctest.log'
blocks=re.split(r'^\d+/\d+ Testing: hyperbolic_shift_3d_sampled_normal_device$',
                source.read_text(),flags=re.M)
if len(blocks)!=2:
    raise ValueError('Expected exactly one native sampled-normal test block')
match=re.search(r'^Output:\n-+\n(.*?)^<end of output>$',blocks[1],re.M|re.S)
if match is None or not re.search(r'^Test Passed\.$',blocks[1],re.M):
    raise ValueError('Missing completed native component output')
result=preflight.parse_component(match[1],'sampled_normal_device',0)
if result['backend']!='Serial' or result['passed'] or result['gpu_cpu_parity']:
    raise ValueError('Serial format validation must not admit a GPU result')
print(json.dumps(dict(status='format_verified',gpu_admission=False,
    scope='Actual Serial stdout metrics accepted; SYCL admission correctly rejected',
    log_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    parser_sha256=hashlib.sha256(Path(preflight.__file__).read_bytes()).hexdigest(),
    result=result),indent=2,allow_nan=False))
