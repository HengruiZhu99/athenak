"""Check binary per-hole damping at mass .5 with kappa=1 and eta=2 retained."""
import argparse
from pathlib import Path
import re

from make_inputs import parse,write


def half_scale(b):
    for block,values in b.items():
        for key,value in list(values.items()):
            geometry=bool(re.fullmatch(r'x[123](min|max)',key))
            geometry |= key.startswith(('center_','extent_','slice_x'))
            geometry |= key in ['mass','audit_r_min','audit_r_max','expected_finest_spacing',
                'physical_output_inner_radius','constraint_horizon_radius','constraint_horizon_buffer',
                'co_0_dump_radius','horizon0r_guess','horizon_dt','tlim','dt']
            if geometry: values[key]=str(float(value)/2)
    b['pc_gh']['reduction_rate']='2'
    # kappa and eta intentionally retain the binary values, not a complete
    # dimensional rescaling of the previous single-hole gauge.
    assert b['pc_gh']['kappa']=='1.0' and b['pc_gh']['shift_eta']=='2.0'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('qualification',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();q=args.qualification
    if args.output.exists(): raise SystemExit('Refusing to overwrite inputs')
    for n in [16,20,24]:
        for kind in ['single-large','single']:
            hierarchy='smr' if kind=='single-large' else 'uniform'
            name=f'pcgh-advective-l1-{hierarchy}-r{n}'+('-R64' if kind=='single-large' else '')
            b=parse((q/'inputs-v2'/kind/(name+'.athinput')).read_text())
            half_scale(b)
            write(args.output/'single'/f'pcgh-m05-l2-{hierarchy}-r{n}.athinput',b)
            if n==16 and kind=='single-large':
                b['pc_gh']['reduction_rate']='1'
                write(args.output/'single'/'pcgh-m05-l1-smr-r16.athinput',b)
    b=parse((q/'della-cuda/single-core/pcgh-advective-l1-smr-r16-dx1over256/used_input.athinput').read_text())
    half_scale(b)
    write(args.output/'single'/'pcgh-m05-l2-core512-R4.athinput',b)
    for n in [32,64,128]:
        b=parse((q/'inputs-v2/waves'/f'shifted-wave-advective-l1-n{n}.athinput').read_text())
        b['pc_gh']['reduction_rate']='2'
        write(args.output/'waves'/f'shifted-wave-advective-l2-n{n}.athinput',b)
    for family in ['p','Q','L','B']:
        for n in [256,512,1024]:
            b=parse((q/'inputs-v2/pulses'/f'{family}-d1-l1-n256.athinput').read_text())
            b['pc_gh']['reduction_rate']='2'
            b['mesh']['nx1']=b['meshblock']['nx1']=str(n)
            write(args.output/'pulses'/f'{family}-d1-l2-n{n}.athinput',b)
        for n in [32,48,64]:
            b=parse((q/'inputs-v2/amr-pulse'/f'{family}-l1-n{n}-smr1.athinput').read_text())
            b['pc_gh'].update(reduction_rate='2',boundedness_dcycle='1',constraint_dcycle='1')
            write(args.output/'amr'/f'{family}-l2-n{n}-smr1.athinput',b)
    (args.output/'README.txt').write_text(__doc__+'\n'
        'Single-hole lengths/times are halved: mass=.5, lambda=2, kappa=1, eta=2.\n'
        'The fine core reaches dx=1/512 in total-binary-mass units (m/256).\n'
        'All single evolutions keep FD2/RK4/CFL=.1/KO=.3 and projections disabled.\n'
        'This is a new control set, not an assertion of binary qualification.\n')


if __name__=='__main__': main()
