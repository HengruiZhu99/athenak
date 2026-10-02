#!/usr/bin/env python3
"""Finite, ordered single-hole matrix; identical physical mesh at each resolution."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'telegrapher_aurora_20260929/inputs_v3/g5_bhahaha_L6.athinput'


def parse(text):
    blocks = {}
    current = None
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('<'):
            current = line.strip('<>')
            blocks[current] = {}
        elif '=' in line and current:
            key, value = line.split('=', 1)
            blocks[current][key.strip()] = value.strip()
    return blocks


def dump(blocks):
    return '\n\n'.join('<{}>\n{}'.format(section, '\n'.join(
        '{} = {}'.format(k, v) for k, v in values.items()))
        for section, values in blocks.items()) + '\n'


def make(kind, gauge, block, width=None):
    b = parse(BASE.read_text())
    suffix = 'base' if width is None else 'w{}'.format(str(width).replace('.', 'p'))
    name = '{}_{}_b{}{}'.format(kind, gauge, block, '' if kind == 'g5' else '_'+suffix)
    b['job']['basename'] = name
    for a in (1, 2, 3):
        b['mesh']['nx'+str(a)] = 4*block
        b['meshblock']['nx'+str(a)] = block
    b['time'].update(tlim=4 if kind == 'g5' else 6, ndiag=50)
    b['z4c'].update(telegraph_lapse=str(gauge=='tel').lower(),
                    slow_start_lapse=str(gauge=='ssl').lower(),
                    history_interior_radius=16 if kind=='g5' else 8,
                    history_inner_radius=8 if kind=='g5' else 2,
                    history_boundary_speed=8, history_boundary_buffer=4)
    # Freeze the physical mesh: no tracker-driven AMR topology changes.
    b['mesh_refinement'].update(refinement='static', num_levels=7, max_nmb_per_rank=128)
    b.pop('amr_criterion0', None)
    b['refined_region1'].update(level=6 if kind=='g5' else 5,
                               x1min=-1, x1max=2 if kind=='g5' else 1,
                               x2min=-1,x2max=1,x3min=-1,x3max=1)
    if kind == 'pulse':
        b['problem'] = dict(pgen_name='z4c_one_puncture',punc_ADM_mass=1,
                            lapse_pulse_amplitude=0 if width is None else .2,
                            lapse_pulse_width=1 if width is None else width,
                            lapse_pulse_x=4, lapse_pulse_y=0,lapse_pulse_z=0)
        b['z4c']['co_0_reflevel'] = 5
        for number, (level, extent) in enumerate(((4,8),(3,8)), 2):
            b['refined_region'+str(number)] = dict(level=level,
                x1min=-extent,x1max=extent,x2min=-2 if level==4 else -extent,x2max=2 if level==4 else extent,
                x3min=-2 if level==4 else -extent,x3max=2 if level==4 else extent)
        b['bhahaha']['bah_dt'] = 2
    for n in range(1, 5):
        b['output'+str(n)]['dt'] = .05 if kind=='pulse' else .1
        b['output'+str(n)]['data_format'] = '%24.16e'
    b.pop('output5', None)  # No multi-GB volume checkpoints in this short matrix.
    text = dump(b)
    geometry = {k: v for k,v in b.items() if k.startswith('refined_region')}
    geometry['domain'] = {k:v for k,v in b['mesh'].items() if not k.startswith('nx')}
    return dict(name=name, kind=kind, gauge=gauge, block=block, width=width,
                target_time=b['time']['tlim'],
                input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                geometry_sha256=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()), text


def main():
    (ROOT/'inputs').mkdir(exist_ok=True)
    cases = []
    # All three gauges at each resolution; boosted study precedes pulse study.
    for block in (32,48,64):
        for gauge in ('tel','oplog','ssl'):
            meta, text = make('g5',gauge,block)
            cases.append(meta)
            (ROOT/'inputs'/(meta['name']+'.athinput')).write_text(text)
    for block, widths in ((32,(None,1,.5)),(48,(None,.5)),(64,(None,.5))):
        for gauge in ('tel','oplog','ssl'):
            for width in widths:
                meta, text = make('pulse',gauge,block,width)
                cases.append(meta)
                (ROOT/'inputs'/(meta['name']+'.athinput')).write_text(text)
    (ROOT/'cases.json').write_text(json.dumps(dict(cases=cases,
        max_attempts_per_case=2, max_campaign_node_hours=60,
        max_new_output_bytes=5*1024**3,
        note='30 short single-hole cases, maximum two 1-node 1-hour attempts per case; no automatic enlargement.'),indent=2)+'\n')

if __name__ == '__main__':
    main()
