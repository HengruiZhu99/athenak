#!/usr/bin/env python3
"""Analyze the declared fixed-physical-boundary scalar matrix after collection."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';collected=A/'cpu-schwarzschild-commensurate-v1-collected';prepared=A/'cpu-schwarzschild-commensurate-v1';R=E/'cpu-schwarzschild-commensurate';scripts=Path(__file__).resolve().parent
if R.exists():raise RuntimeError('Existing commensurate analysis is immutable; inspect it')
while not (collected/'collection-receipt.json').exists():time.sleep(15)
collection=json.loads((collected/'collection-receipt.json').read_text());manifest=json.loads((collected/'remote-manifest.json').read_text());plan=json.loads((prepared/'cpu-plan.json').read_text());expected=json.loads((prepared/'portable-source-manifest.json').read_text());result=collected/'result';worker=json.loads((result/'cpu-worker-receipt.json').read_text());source={name:hashlib.sha256((Path('src/test_scalar')/name).read_bytes()).hexdigest() for name in plan['source_sha256']};binary=hashlib.sha256((result/'test_scalar').read_bytes()).hexdigest()
if not collection['passed'] or manifest['source_files']!=expected or source!=plan['source_sha256'] or binary!=plan['binary_sha256']:raise RuntimeError('Fixed-boundary source/binary/collection check failed')
if not worker['all_cases_complete']:raise RuntimeError('Fixed-boundary wall checkpoint preserved; resume only the missing prefix with the exact binary and parameters')
R.mkdir();runs=[result/f'schwarzschild-n{n}-fixed-cube' for n in [24,36,48]];oracle=E/'radial-oracle/smooth'
subprocess.run([sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle),'--oracle-waveform','waveform-upwind-n1200.csv','--normal-coefficients',str(oracle/'normal-coefficients-n1200.txt'),'--output',str(R/'primary-review.json')],check=True)
primary=json.loads((R/'primary-review.json').read_text());print('FIXED-BOUNDARY MATRIX',json.dumps(dict(field_checks_passed=primary['field_checks_passed'],finest_accuracy_passed=primary['finest_accuracy_passed'])),flush=True)
