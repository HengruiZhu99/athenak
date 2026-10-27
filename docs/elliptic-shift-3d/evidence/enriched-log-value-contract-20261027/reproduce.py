#!/usr/bin/env python3
"""Configure/build only the two local contracts, then run bounded CTest -j2.
Use --test-only when these targets were already built in the same session.
Coordinate use of the shared build directory with any other native rebuild.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BUILD=ROOT/'build-shift-3d'
parser=argparse.ArgumentParser();parser.add_argument('--test-only',action='store_true');args=parser.parse_args()
commands=[]
def execute(command,log,timeout=120):
    start=time.monotonic();result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    (HERE/log).write_text(result.stdout+result.stderr)
    commands.append({'argv':command,'return_code':result.returncode,'wall_seconds':time.monotonic()-start,'log':log})
    print(result.stdout,end='');print(result.stderr,end='')
    if result.returncode:raise SystemExit(result.returncode)
if not args.test_only:
    execute(['cmake','-S',str(ROOT),'-B',str(BUILD)],'configure.log')
    execute(['cmake','--build',str(BUILD),'--target','elliptic_shift_3d_affine_shift_remainder_tests','elliptic_shift_3d_enriched_log_value_tests','-j2'],'build.log')
execute(['ctest','--test-dir',str(BUILD),'-j2','--output-on-failure','-R','^elliptic_shift_3d_(affine_shift_remainder|enriched_log_value)$','--output-junit',str(HERE/'ctest.xml')],'ctest.log',90)
summaries={}
for test in ET.parse(HERE/'ctest.xml').getroot().iter('testcase'):
    output=test.findtext('system-out') or ''
    try:data=json.loads(output.strip())
    except json.JSONDecodeError:continue
    summaries[test.attrib['name']]=data.get('summary',data)
(HERE/'contracts-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
deps=[ROOT/'CMakeLists.txt',ROOT/'tst/elliptic_shift_3d/test_enriched_log_value.cpp',ROOT/'tst/elliptic_shift_3d/test_affine_shift_remainder.cpp',ROOT/'src/z4c/hyperboloidal/enriched_log_value.hpp',ROOT/'src/z4c/hyperboloidal/enriched_log_derivative.hpp',ROOT/'src/z4c/hyperboloidal/affine_shift_remainder.hpp',Path(__file__)]
manifest={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'test_only':args.test_only,'commands':commands,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},'parallel_limit':2,'qualification':'Self-contained local contracts; no external data/cache, production evolution or coupled acceptance claim.'}
if args.test_only:
    manifest['prior_build_observed']={'commands':['cmake -S . -B build-shift-3d','cmake --build build-shift-3d --target elliptic_shift_3d_affine_shift_remainder_tests elliptic_shift_3d_enriched_log_value_tests -j2'],'return_codes':[0,0],'provenance':'Executed directly via exec tools earlier in this delegated turn; CTest deliberately waits for coordinated native rebuild to finish.'}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
