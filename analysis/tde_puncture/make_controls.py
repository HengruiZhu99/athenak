#!/usr/bin/env python3
"""Generate reproducible residual-puncture controls; does not submit jobs.

Wormhole background held fixed by subtraction, not a stationary trumpet.
Star data use the existing weak-field TOV superposition, not a solved binary
constraint problem. Orbit conversion neglects the star's O(1e-6) self metric.
"""
import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def setting(text, section, key, value):
    match = re.search(r'(<'+section+r'>\n)([^<]*)', text)
    assert match, section
    body = match[2]
    if re.search(r'^'+key+r'\s*=', body, re.M):
        body = re.sub(r'^'+key+r'\s*=.*', f'{key} = {value}', body, flags=re.M)
    else:
        body += f'{key} = {value}\n'
    return text[:match.start(2)] + body + text[match.end(2):]


def make_case(case, target=100.):
    s = (ROOT/'tst/inputs/z4c_puncture_background.athinput').read_text()
    configs = {
        'mesh_refinement': {'refinement': 'static', 'num_levels': 10, 'max_nmb_per_rank': 1024},
        'time': {'nlim': -1, 'tlim': target, 'ndiag': 100},
        'z4c': {'debug_balance': 'false', 'debug_snapshot_operations': '',
                'characteristic_bc_diagnostics': 'false', 'history_excise_ks_radius': .5},
        'mhd': {'zero_tmunu_feedback': 'true' if case in ['vacuum', 'lapse', 'lapse_double'] else 'false',
                'dfloor': 1.6e-21, 'pfloor': 1.6e-33},
        'problem': {'zero_tmunu': 'true' if case in ['vacuum', 'lapse', 'lapse_double'] else 'false',
                    'pure_background': 'false' if case == 'star' else 'true',
                    'amr_star_refine': 'false', 'amr_bh_refine_level': -1},
        'output1': {'dt': 1., 'data_format': '%20.15e'},
    }
    if case.startswith('lapse'):
        configs['problem'].update(vacuum_gauge_pulse_amplitude=2e-8 if case.endswith('double') else 1e-8,
                                  vacuum_gauge_pulse_x1=.75, vacuum_gauge_pulse_width=.5)
    if case == 'star':
        configs['problem'].update(rhoc=4.9451950378e-6, kappa=.00010641642727074857,
                                  npoints=200000, dr=5e-5, rho_cut=1.6e-21,
                                  star_center_x1=orbit()['isotropic_r0_M'],
                                  star_boost_x=orbit()['input_boost_x'],
                                  star_boost_y=orbit()['input_boost_y'])
    # Identical matter/vacuum meshes for an honest controlled comparison.
    for a in range(1, 4):
        configs.setdefault('mesh', {}).update({f'nx{a}': 16, f'x{a}min': -256, f'x{a}max': 256})
    for section, changes in configs.items():
        for key, value in changes.items():
            s = setting(s, section, key, value)
    s += '\n<refined_region0>\nlevel = 9\nx1min = -.5\nx1max = .5\nx2min = -.5\nx2max = .5\nx3min = -.5\nx3max = .5\n'
    x = orbit()['isotropic_r0_M']
    s += f'\n<refined_region1>\nlevel = 7\nx1min = {x-4}\nx1max = {x+4}\nx2min = -4\nx2max = 4\nx3min = -4\nx3max = 4\n'
    # Thin slices record the hole and star on the same plane. Volume histories
    # and metric diagnostics remain enabled, including unexcised maxima.
    s += '\n<output2>\nfile_type = bin\nvariable = z4c_residual\nslice_x2 = 0\ndt = 20\n'
    s += '\n<output3>\nfile_type = bin\nvariable = mhd_w_bcc\nslice_x3 = 0\ndt = 20\n'
    return s


def orbit():
    R = 206.6705864860798
    rp = 20.
    L = math.sqrt(2*rp*rp/(rp-2))
    f = 1-2/R
    r = .5*(R-1+math.sqrt(R*(R-2)))
    psi = 1+.5/r
    ur = -math.sqrt(2/R - f*L*L/(R*R))/math.sqrt(f)
    ut = L/R
    # Eulerian coordinate momentum on the isotropic spatial slice.
    ux, uy = ur/(psi*psi), ut/(psi*psi)
    # Existing weak-field boosted TOV generator gives w^i approximately Wv^i.
    flatW = math.sqrt(1+ux*ux+uy*uy)
    W = math.sqrt(1+(psi**4)*(ux*ux+uy*uy))
    assert abs(math.sqrt(f)*W-1) < 1e-14
    assert abs(psi**4*r*uy-L) < 1e-14
    return dict(MBH_Msun=2e5, Mstar_Msun=1, areal_r0_M=R, isotropic_r0_M=r,
                areal_rp_M=rp, separation_in_tidal_radii=1.5, E=1., L=L,
                input_boost_x=ux/flatW, input_boost_y=uy/flatW,
                background_E_check=math.sqrt(f)*W,
                approximation='Initial TOV superposition; orbit neglects stellar self metric; fixed wormhole is not a stationary Einstein background with this lapse.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=float, default=100.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for case in ['vacuum', 'lapse', 'lapse_double', 'atmosphere', 'star']:
        (args.output/(case+'.athinput')).write_text(make_case(case, args.target))
    (args.output/'orbit.json').write_text(json.dumps(orbit(), indent=2)+'\n')


if __name__ == '__main__':
    main()
