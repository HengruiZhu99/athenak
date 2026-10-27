#!/usr/bin/env python3
"""Build/run the geometry-only refinement controls with a 30-second bound."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BUILD=ROOT/'build-shift-3d'
TEST=ROOT/'tst/elliptic_shift_3d/test_puncture_refinement.cpp'
flags=[]
for line in (BUILD/'CMakeFiles/elliptic_shift_3d_puncture_position_tests.dir/flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
        flags.extend(shlex.split(line.split('=',1)[1]))
binary=BUILD/'puncture_refinement_prototype'
compile_command=['/usr/bin/c++',*flags,str(TEST),'-o',str(binary),*map(str,(BUILD/'kokkos').glob('*/src/libkokkos*.a'))]
subprocess.run(compile_command,cwd=ROOT,check=True)
start=time.monotonic()
run_command=[str(binary)]
result=subprocess.run(run_command,cwd=ROOT,capture_output=True,text=True,timeout=30)
wall=time.monotonic()-start
(HERE/'results.json').write_text(result.stdout)
(HERE/'stderr.txt').write_text(result.stderr)
deps=[TEST,ROOT/'src/z4c/hyperboloidal/puncture_refinement.hpp',ROOT/'src/z4c/hyperboloidal/puncture_position.hpp',ROOT/'src/mesh/cubed_sphere/composite.hpp',ROOT/'src/mesh/cubed_sphere/map.hpp',Path(__file__)]
manifest={'return_code':result.returncode,'wall_seconds':wall,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compile_command':shlex.join(compile_command),'run_command':shlex.join(run_command),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(result.stdout,end='');print(result.stderr,end='')
raise SystemExit(result.returncode)
