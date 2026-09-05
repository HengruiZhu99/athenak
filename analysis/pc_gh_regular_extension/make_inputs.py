"""Generate explicit, reviewable CUDA qualification inputs; does not run them."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse(text):
    blocks, current = {}, None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('<'):
            current = line.strip('<>')
            blocks[current] = {}
        elif '=' in line:
            key, value = line.split('=', 1)
            blocks[current][key.strip()] = value.strip()
    return blocks


def write(path, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '# Research candidate input; no numerical qualification implied.\n'
    for name, values in blocks.items():
        text += '\n<'+name+'>\n'
        text += ''.join(f'{key} = {value}\n' for key, value in values.items())
    path.write_text(text)


def load(name):
    return parse((ROOT/'tst/inputs'/name).read_text())


def reduction(blocks, rate, mode='advective'):
    blocks['pc_gh'].update(reduction_system=mode, reduction_rate=str(rate),
                          reduction_monitor='true', project_reduction_constraints='false',
                          project_gauge_constraints='false')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    out = ap.parse_args().output
    if out.exists():
        raise SystemExit('Refusing to overwrite an existing input collection')
    # Flat and gauge controls compare identical algorithms at zero/nonzero rates.
    for mode, rate in [('legacy', 0), ('advective', 0), ('advective', 1)]:
        tag = f'{mode}-l{rate}'
        b = load('pc_gh_minkowski.athinput')
        b['time'].update(tlim='1', nlim='-1')
        b['pc_gh'].update(gauge='z4c_mp_hyperbolic', shift_eta='0', dissipation='0')
        reduction(b, rate, mode)
        write(out/'flat'/f'minkowski-{tag}.athinput', b)
        for n in [32, 64, 128]:
            b = load('pc_gh_gauge_wave.athinput')
            b['mesh']['nx1'] = str(n)
            b['meshblock']['nx1'] = str(n)
            reduction(b, rate, mode)
            write(out/'waves'/f'shifted-wave-{tag}-n{n}.athinput', b)
    # One-dimensional smooth compact pulses, including transverse curl modes.
    for family in ['p', 'Q', 'L', 'B']:
        for direction in [0, 1]:
            for rate in [0, 1, 4]:
                for n in [64, 128, 256]:
                    b = load('pc_gh_gauge_wave.athinput')
                    b['mesh'].update(nx1=str(n), x1min='-4', x1max='4')
                    b['meshblock']['nx1'] = str(n)
                    b['time'].update(tlim='1', integrator='rk4', cfl_number='0.2')
                    b['pc_gh'].update(gauge='z4c_mp_hyperbolic', shift_eta='0', kappa='1')
                    reduction(b, rate)
                    b['problem'] = dict(pgen_name='regular_extension_pulse', pulse_family=family,
                        pulse_direction=str(direction), pulse_amplitude='1e-8',
                        pulse_width='0.75', pulse_shift='0.5')
                    write(out/'pulses'/f'{family}-d{direction}-l{rate}-n{n}.athinput', b)
        # Separate finite-rate stiffness and nonlinear-amplitude controls.
        for rate, stop, amp, tag in [(100, .02, '1e-8', 'stiff'),
                                     (1, 1, '1e-7', 'amplitude')]:
            b = parse((out/'pulses'/f'{family}-d1-l1-n128.athinput').read_text())
            b['pc_gh']['reduction_rate'] = str(rate)
            b['time']['tlim'] = str(stop)
            b['problem']['pulse_amplitude'] = amp
            write(out/'pulses'/f'{family}-{tag}.athinput', b)
    # Matched spacing across uniform and SMR: dx={1/8,1/10,1/12}M.
    # The 16M domain is an explicit finite-boundary control. A separate 128M
    # SMR domain below tests boundary contamination; no asymptotic claim follows
    # merely from convergence on the smaller domain.
    for r in [16, 20, 24]:
        for hierarchy in ['uniform', 'smr']:
            for formulation, mode, rate in [('pcgh', 'legacy', 0), ('pcgh', 'advective', 0),
                                            ('pcgh', 'advective', 1), ('z4c', '', 0)]:
                source = ('pc_gh_one_puncture_smr.athinput' if formulation == 'pcgh'
                          else 'z4c_one_puncture_control_smr.athinput')
                b = load(source)
                b.pop('refined_region4')
                b['mesh_refinement']['max_nmb_per_rank'] = '512'
                if hierarchy == 'uniform':
                    b = {name: values for name, values in b.items()
                         if name != 'mesh_refinement' and not name.startswith('refined_region')}
                for d in [1, 2, 3]:
                    b['mesh'][f'nx{d}'] = str(r if hierarchy == 'smr' else 8*r)
                    b['meshblock'][f'nx{d}'] = str(r//2 if hierarchy == 'smr' else 2*r)
                    for side in ['i', 'o']:
                        b['mesh'][f'{side}x{d}_bc'] = 'outflow'
                b['time'].update(tlim='6', integrator='rk4', cfl_number='0.1', ndiag='100')
                physics = b['pc_gh' if formulation == 'pcgh' else 'z4c']
                physics.update(extrap_order='2', dump_horizon_0='false')
                if formulation == 'pcgh':
                    physics.update(gauge='z4c_mp_hyperbolic', dissipation='0.3',
                                   boundedness_dcycle='1', constraint_dcycle='1')
                    reduction(b, rate, mode)
                    b['problem']['expected_finest_spacing'] = str(2/r)
                else:
                    physics['diss'] = '0.3'
                b['output4'] = dict(file_type='rst', dt='1')
                b['output5'] = dict(file_type='bin', variable='pcgh' if formulation == 'pcgh'
                                   else 'z4c', slice_x3='0', dt='0.25')
                tag = f'{formulation}-{mode}-l{rate}-{hierarchy}-r{r}'
                write(out/'single'/f'{tag}.athinput', b)
                if formulation == 'pcgh' and mode == 'advective' and hierarchy == 'smr':
                    # Same innermost dx, much more distant Sommerfeld boundary.
                    for d in [1, 2, 3]:
                        b['mesh'][f'x{d}min'], b['mesh'][f'x{d}max'] = '-64', '64'
                    for level in range(1, 7):
                        width = 64/(2**level)-1e-6
                        b[f'refined_region{level}'] = {'level': str(level)}
                        for d in [1, 2, 3]:
                            b[f'refined_region{level}'][f'x{d}min'] = str(-width)
                            b[f'refined_region{level}'][f'x{d}max'] = str(width)
                    b['time']['tlim'] = '20'
                    write(out/'single-large'/f'{tag}-R64.athinput', b)
    # A 3D compact pulse traverses x=0 at t=2, advecting left.
    # Use the 3D transfer path used by puncture runs; the 1D SMR initialization
    # probe crashed before evolution and is preserved as a fixture failure.
    for family in ['p', 'Q', 'L', 'B']:
        for rate in [0, 1]:
            for refined in [False, True]:
                for n in [32, 48, 64]:
                    b = parse((out/'pulses'/f'{family}-d1-l{rate}-n128.athinput').read_text())
                    for d in [1, 2, 3]:
                        b['mesh'][f'nx{d}'] = str(n)
                        b['mesh'][f'x{d}min'] = '-4'
                        b['mesh'][f'x{d}max'] = '4'
                        b['meshblock'][f'nx{d}'] = str(n//4)
                    b['time']['tlim'] = '4'
                    b['problem'].update(pulse_center_x='1', pulse_width='0.75', pulse_radial='true')
                    if refined:
                        b['mesh_refinement'] = dict(refinement='static', max_nmb_per_rank='512')
                        b['refined_region1'] = dict(level='1', x1min='0.000001', x1max='3.999999',
                            x2min='-3.999999', x2max='3.999999', x3min='-3.999999', x3max='3.999999')
                    write(out/'amr-pulse'/f'{family}-l{rate}-n{n}-smr{int(refined)}.athinput', b)
    # Exact established large-domain input, including its dynamic chi criterion.
    path = ROOT/'inputs/z4c/twopuncture/bbh_headon_pcgh_cuda_r128_t100.athinput'
    b = parse(path.read_text())
    b['job']['basename'] = 'pcgh_regular_advective_headon'
    reduction(b, 1)
    b['pc_gh'].update(kappa='1', boundedness_dcycle='10', constraint_dcycle='10')
    b['output2']['dt'] = '2'
    write(out/'binary'/'headon-t100.athinput', b)
    (out/'README.txt').write_text(
        'Generated inputs are test definitions, not passed gates.\n'
        'Run stages in order: oracle, flat/pulses, waves, single (+large domain), AMR pulse, binary.\n'
        'Only advance to binary after inspecting convergence, puncture powers, stability, and AMR injections.\n'
        'Primary runs disable both reduction and GH gauge projection.\n'
        'Binary differs from saved projected baseline in projection switches, reduction extension, and kappa=1;\n'
        'matched kappa/rate controls are required before attributing any improvement to reduction damping.\n')
    print(out.resolve())


if __name__ == '__main__':
    main()
