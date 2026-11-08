#!/usr/bin/env python3
"""Write the ordinary adaptive GPU restart fixture; never launch or submit it."""
import argparse
import hashlib
import json
from pathlib import Path

SAFE_DT = .0002


def helper_text():
    # Only the timestep changes. The C++ fixtures own their criterion/driver
    # setup; keep their initial static construction and all physics parameters.
    return Path(__file__).with_name('n12_initialize.athinput').read_text()+'''
<time>
dt = .0002
'''


def restart_text():
    return helper_text()+'''
<job>
basename = amr_restart
<mesh_refinement>
refinement = adaptive
num_levels = 2
max_nmb_per_rank = 16
refinement_interval = 0
ncycle_check = 1
<amr_criterion0>
method = location
location_x1 = .99
location_x2 = 0
location_x3 = 0
location_rad = .005
<time>
evolution = dynamic
integrator = rk4_classical
tlim = 1
nlim = 4
ndiag = 1
<output1>
file_type = rst
dcycle = 1
single_file_per_rank = false
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('new input directory required')
    fixture=Path(__file__).with_name('n12_initialize.athinput')
    args.output.mkdir(parents=True)
    target=args.output/'amr_restart.athinput'
    target.write_text(restart_text())
    helper=args.output/'amr_helper.athinput'; helper.write_text(helper_text())
    report=dict(schema='native-amr-gpu-input-v2', capacity=16, fixed_dt=SAFE_DT,
        runtime_scope='isolated reviewed admission candidate, not the frozen N12 baseline',
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        helper_input_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        commands=dict(lifecycle=[str(helper.resolve()),'--diagnostics'],
            evolution=[str(helper.resolve())], prefix24=['-i',str(target.resolve()),'time/nlim=2'],
            continuous96=['-i',str(target.resolve())],
            resumed96=['-r','<verified-prefix-cycle2-shared-checkpoint>','time/nlim=4']),
        launched=False,scope='ordinary executable input only; site launcher/tile binding remains required')
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
