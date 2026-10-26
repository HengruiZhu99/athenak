#!/usr/bin/env python3
"""Bounded boundary-only restarts of the archived N24 wave-map failure.

The unmodified degree-two replay must already have reproduced the original
fatal line. These counterfactuals change only the ghost polynomial degree;
they cannot remove contamination accumulated before the saved checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree',type=int,nargs='+',choices=[2,3,4,5],default=[4])
    parser.add_argument('--resolution',type=int,choices=[24,32],default=24)
    parser.add_argument('--tlim',type=float)
    args=parser.parse_args(); degrees=args.degree
    root = Path(__file__).resolve().parents[2]
    name='old-wave-replay' if args.resolution==24 else 'old-wave-replay-N32'
    baseline = root/'build-cubed-sphere/research-20261012'/name
    evidence = root/'docs/validation/cubed-sphere-refinement-20261012'
    comparison = json.loads((evidence/name/('comparison.json' if args.resolution==24 else 'receipt.json')).read_text())
    assert comparison['terminal_error_byte_identical']
    binary, restart = baseline/'athena_frozen', baseline/'input.rst'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary,restart]}
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

    def run(degree):
        suffix='' if args.resolution==24 else '-N32'
        work = baseline.parent/('boundary-degree-%d%s' % (degree,suffix))
        work.mkdir(exist_ok=False)
        command = [str(binary),'-r',str(restart),'-d',str(work),
                   'time/ndiag=100','z4c/hyperboloidal_ghost_degree=%d' % degree,
                   'time/tlim=%.17g' % (args.tlim or (0.82 if args.resolution==24 else 1.6))]
        receipt = {'command':command,'input_hashes':hashes,
                   'scope':'Boundary-only restart from the exact replay checkpoint. Prior errors remain.'}
        start = time.monotonic()
        with (work/'run.log').open('w') as log, (work/'stderr.log').open('w') as err:
            try:
                proc = subprocess.run(command,stdout=log,stderr=err,timeout=150)
                receipt['exit_code'] = proc.returncode
            except subprocess.TimeoutExpired:
                receipt['timeout'] = True
        receipt['elapsed_seconds'] = time.monotonic()-start
        stderr = (work/'stderr.log').read_text()
        receipt['fatal_lines'] = [line for line in stderr.splitlines()
                                  if 'terminating due to uncaught exception' in line]
        (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        destination = evidence/work.name
        destination.mkdir(exist_ok=False)
        for name in ['run.log','stderr.log','receipt.json']:
            shutil.copy2(work/name,destination/name)
        return receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run,degrees))
    (evidence/('boundary-sensitivity-N%d-'%args.resolution+ '-'.join(map(str,degrees))+'.json')).write_text(
        json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
