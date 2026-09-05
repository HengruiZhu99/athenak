"""Prepare focused FD6/RK3 controls for the constant rate-sixteen candidate."""
import argparse
from pathlib import Path

from make_inputs import parse,write


def configure(b):
    b['mesh']['nghost']='4'
    b['time'].update(integrator='rk3',cfl_number='.2')
    b['pc_gh'].update(spatial_order='6',reduction_monitor='true')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args()
    if args.output.exists(): raise SystemExit('Refusing to overwrite inputs')
    for mode,rate in [('legacy',0),('advective',0),('advective',16)]:
        oldrate=min(rate,1);tag=f'{mode}-l{rate}'
        b=parse((args.source/'flat'/f'minkowski-{mode}-l{oldrate}.athinput').read_text())
        configure(b);b['pc_gh']['reduction_rate']=str(rate)
        write(args.output/'flat'/f'minkowski-{tag}.athinput',b)
        for n in [32,64,128]:
            b=parse((args.source/'waves'/f'shifted-wave-{mode}-l{oldrate}-n{n}.athinput').read_text())
            configure(b);b['pc_gh']['reduction_rate']=str(rate)
            write(args.output/'waves'/f'shifted-wave-{tag}-n{n}.athinput',b)
    for family in ['p','Q','L','B']:
        for d,n,cfl in [(1,n,.2) for n in [256,512,1024]]+[(0,256,.2),(1,256,.1),(1,256,.05)]:
            b=parse((args.source/'pulses'/f'{family}-d{d}-l1-n256.athinput').read_text())
            configure(b);b['time'].update(tlim='.25',cfl_number=str(cfl))
            b['mesh']['nx1']=b['meshblock']['nx1']=str(n)
            b['pc_gh']['reduction_rate']='16';b['problem']['pulse_amplitude']='1e-7'
            write(args.output/'pulses'/f'{family}-d{d}-l16-n{n}-cfl{cfl}.athinput',b)
        for rate in [0,16]:
            for n in [32,48,64]:
                for smr in [0,1]:
                    b=parse((args.source/'amr-pulse'/f'{family}-l{min(rate,1)}-n{n}-smr{smr}.athinput').read_text())
                    configure(b);b['time']['tlim']='.25'
                    b['pc_gh'].update(reduction_rate=str(rate),boundedness_output='true',
                                       boundedness_dcycle='1',constraint_dcycle='1')
                    b['problem'].update(pulse_shift='8',pulse_amplitude='1e-7')
                    b['output1']=dict(file_type='rst',dt='.125')
                    write(args.output/'amr'/f'{family}-l{rate}-n{n}-smr{smr}.athinput',b)
    (args.output/'README.txt').write_text(
        'FD6/RK3 controls. Uniform compact pulses use lambda=16, beta=.5, t=.25.\n'
        'AMR uses beta=8, t=.25: same two-unit interface crossing distance, with lambda*t=4.\n'
        'The larger prescribed shift keeps the seeded packet measurable while testing stiff relaxation across an interface.\n'
        'This is a controlled flat transport experiment, not a black-hole shift model.\n')
