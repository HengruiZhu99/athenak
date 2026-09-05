"""Finite smooth inner relaxation controls, with fixed physical mask widths."""
import argparse
from pathlib import Path
from make_inputs import parse, write


def profile(b, core=.2, taper=.8, following=False):
    b['pc_gh'].update(reduction_profile='smooth_core', reduction_rate='2',
        reduction_inner_rate='16', reduction_core_radius=str(core),
        reduction_taper_radius=str(taper), reduction_follow_trackers=str(following).lower())
    if following:
        for n, x in enumerate([-.25, .25]):
            b['pc_gh'].update({f'co_{n}':'true', f'co_{n}_type':'BH', f'co_{n}_x':str(x),
                              f'co_{n}_y':'0', f'co_{n}_z':'0'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('qualification', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a control collection')
    q = args.qualification
    root = Path(__file__).resolve().parents[2]
    for tag, center in [('core','0'),('taper','.375'),('outside','2')]:
        b = parse((root/'analysis/pc_gh_regular_extension/oracle.athinput').read_text())
        profile(b,core=.2,taper=.5)
        b['pc_gh']['reduction_center_x'] = center
        write(args.output/'oracle'/f'{tag}.athinput',b)
    b = parse((root/'analysis/pc_gh_regular_extension/oracle.athinput').read_text())
    write(args.output/'oracle/constant.athinput',b)
    for family in ['p', 'Q', 'L', 'B']:
        for n in [256, 512, 1024]:
            b = parse((q/'half-mass-controls/pulses'/f'{family}-d1-l2-n256.athinput').read_text())
            b['mesh']['nx1'] = b['meshblock']['nx1'] = str(n)
            b['time']['tlim'] = '.5'
            profile(b)
            b['pc_gh']['reduction_center_x'] = '-.25'
            write(args.output/'fixed-pulses'/f'{family}-n{n}.athinput', b)
        for n in [32, 48, 64]:
            b = parse((q/'half-mass-controls/pulses'/f'{family}-d1-l2-n256.athinput').read_text())
            for a in [1, 2, 3]:
                b['mesh'].update({f'nx{a}':str(n),f'x{a}min':'-2',f'x{a}max':'2'})
                b['meshblock'][f'nx{a}'] = str(n)
            b['time'].update(tlim='.5', cfl_number='.1')
            b['problem']['pulse_radial'] = 'true'
            profile(b, following=True)
            b['output1'] = dict(file_type='rst', dt='.25')
            write(args.output/'moving-pulses'/f'{family}-n{n}.athinput', b)
    for n in [32, 64, 128]:
        b = parse((q/'half-mass-controls/waves'/f'shifted-wave-advective-l2-n{n}.athinput').read_text())
        profile(b, core=.1, taper=.35)
        write(args.output/'waves'/f'wave-n{n}.athinput', b)
        b['mesh']['nghost'] = '4'
        for a in [2,3]:
            b['mesh'][f'nx{a}'] = b['meshblock'][f'nx{a}'] = '8'
        b['pc_gh']['spatial_order'] = '6'
        b['time'].update(integrator='rk4', tlim='.2', cfl_number='.2')
        b['problem']['amp'] = '.1'
        profile(b, core=.1, taper=.35, following=True)
        b['output1'] = dict(file_type='rst',dt='.1')
        write(args.output/'moving-waves'/f'wave-n{n}.athinput', b)
    b = parse((q/'inputs-v2/flat/minkowski-advective-l1.athinput').read_text())
    profile(b, following=True)
    b['time']['integrator'] = 'rk4'
    write(args.output/'flat/moving-two-centers.athinput', b)
    # Keep constant-rate cases byte-for-byte identical to their old inputs.
    for family in ['p','Q','L','B']:
        b = parse((q/'half-mass-controls/pulses'/f'{family}-d1-l2-n256.athinput').read_text())
        write(args.output/'constant-regression'/f'{family}-n256.athinput', b)
    # These inputs are prepared for review, not automatically dispatched before controls.
    for n in [16,20,24]:
        b = parse((q/'half-mass-controls/single'/f'pcgh-m05-l2-smr-r{n}.athinput').read_text())
        profile(b, core=.125, taper=.5)
        write(args.output/'single'/f'pcgh-m05-smooth16-r{n}.athinput', b)
    (args.output/'README.txt').write_text(__doc__+'\n'
        'Outer lambda=2, inner lambda=16; p/Q/L/B use the same finite rate.\n'
        'Fixed pulses traverse the taper; moving 3D pulses use two overlapping masks.\n'
        'All projection options remain disabled. Prepared single-hole runs are gated.\n')


if __name__ == '__main__':
    main()
