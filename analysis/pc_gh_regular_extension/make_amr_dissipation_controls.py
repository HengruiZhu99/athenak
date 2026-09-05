"""Prepare matched AMR controls at the existing production KO coefficient.

The fast-shift group changes only KO relative to the failed FD6 stress test.
The moving-shift group restores beta=.5 and the four-unit-time crossing. Its
lambda=16 seed decays below roundoff: report absolute transfer errors, not a
late-time fitted rate as evidence of continuum damping.
"""
import argparse
from pathlib import Path

from make_inputs import parse, write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite inputs')
    for group, shift, time in [('fast-shift', '8', '.25'),
                                ('moving-shift', '.5', '4')]:
        for family in ['p', 'Q', 'L', 'B']:
            for rate in [0, 16]:
                for n in [32, 48, 64]:
                    for smr in [0, 1]:
                        name = f'{family}-l{rate}-n{n}-smr{smr}.athinput'
                        b = parse((args.source/name).read_text())
                        b['pc_gh']['dissipation'] = '.3'
                        b['problem']['pulse_shift'] = shift
                        b['time']['tlim'] = time
                        b['output1']['dt'] = str(float(time)/2)
                        write(args.output/group/name, b)
    (args.output/'README.txt').write_text(__doc__.rstrip()+'\n')


if __name__ == '__main__':
    main()
