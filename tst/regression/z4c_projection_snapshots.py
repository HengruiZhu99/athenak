#!/usr/bin/env python3
"""Validate read-only projection snapshots, MPI layout, and numerical noninterference."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
from z4c_background_restart import checkpoint


def snapshots(run, ranks, prefix="z4c_projection_"):
    records = {}
    for path in sorted(run.glob(prefix+'*.json')):
        meta = json.loads(path.read_text())
        assert meta['format_version'] == 1 and meta['layout'] == 'mnkji'
        assert meta['scalar_bytes'] == 8 and meta['byte_order'] in ('little', 'big')
        assert meta['level_convention'] == 'logical'
        assert meta['shape'][1] == 25 and len(meta['blocks']) == meta['shape'][0]
        assert all(b['level'] >= meta['root_level'] for b in meta['blocks'])
        count = 1
        for extent in meta['shape']:
            count *= extent
        data = path.with_suffix('.bin').read_bytes()
        background = path.with_suffix('.background.bin').read_bytes()
        assert len(data) == len(background) == count*meta['scalar_bytes']
        key = (meta['rank'], meta['cycle'], meta['stage'], meta['operation'])
        assert key not in records
        records[key] = (meta, data, background)
    assert {key[0] for key in records} == set(range(ranks))
    return records


def verify_zero(run, ranks):
    records = snapshots(run, ranks)
    for rank in range(ranks):
        roles = {k[3] for k in records if k[0] == rank}
        assert roles == {'init_reconstructed', 'init_projected',
                         'pre_projection', 'post_projection'}
        assert {k[2] for k in records if k[0] == rank and k[3] == 'pre_projection'} == {1,2,3}
    for key, (meta, data, background) in records.items():
        assert data == background, 'Vacuum snapshot is not bitwise equal to background'
        if key[3] == 'pre_projection':
            partner = records[(key[0],key[1],key[2],'post_projection')]
            assert data == partner[1] and background == partner[2]
            assert meta['blocks'] == partner[0]['blocks']
    selected = {}
    if list(run.glob('z4c_snapshot_*.json')):
        selected = snapshots(run, ranks, 'z4c_snapshot_')
        for key, (meta, data, background) in selected.items():
            assert not meta['compare_background']
            # Selected operations in this test are residual state/RHS arrays.
            assert not any(data), 'Zero residual snapshot contains nonzero bits'
    return {'selected_snapshots': len(selected), 'snapshots': len(records), 'exact_background': True,
            'exact_projection': True, 'ranks': ranks}


def final_state(run):
    records = [checkpoint(p) for p in (run/'rst').glob('*.rst')]
    assert records
    result = max(records, key=lambda r:r['cycle'])
    assert result['cycle'] == 1
    return result


def launch(args, run, text):
    run.mkdir(parents=True, exist_ok=False)
    (run/'input.athinput').write_text(text)
    with (run/'run.log').open('w') as log:
        subprocess.run([args.launcher,'-n',str(args.current_ranks),str(args.exe),
                        '-i','input.athinput'],cwd=str(run),stdout=log,
                       stderr=subprocess.STDOUT,check=True)
    assert 'Terminating on cycle limit' in (run/'run.log').read_text()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--launcher',default='mpiexec')
    p.add_argument('--ranks',type=int,nargs='+',default=[1,4])
    p.add_argument('--operations',default='',help='Comma-separated residual state/RHS operations')
    args=p.parse_args();args.exe=args.exe.resolve();args.output=args.output.resolve()
    source=Path(__file__).resolve().parents[1]/'inputs/z4c_ks_background.athinput'
    base=re.sub(r'^nlim\s*=.*','nlim = 1',source.read_text(),flags=re.M)
    base=base.replace('debug_balance = true','debug_balance = true\ndebug_projection_snapshots = true')
    if args.operations:
        base=base.replace('<z4c>','<z4c>\ndebug_snapshot_operations = '+args.operations)
    base+='\n<output2>\nfile_type = rst\ndt = 100\n'
    results={}
    for ranks in args.ranks:
        args.current_ranks=ranks
        for refined in (False,True):
            text=base
            if refined:
                text=text.replace('refinement = none','refinement = static')
                text=text.replace('max_nmb_per_rank = 8','max_nmb_per_rank = 128')
                text+='\n<refined_region0>\nlevel = 1\nx1min = -3.9\nx1max = -0.1\nx2min = -3.9\nx2max = -0.1\nx3min = -3.9\nx3max = -0.1\n'
            name=('refined' if refined else 'uniform')+'_r'+str(ranks)
            run=args.output/name;launch(args,run,text)
            results[name]=verify_zero(run,ranks)
    # The extra host copies and file writes must not change nonzero evolution.
    args.current_ranks=1
    pulse=base.replace('<problem>','<problem>\nouter_sponge_test_theta_pulse_amplitude = 1e-8\nouter_sponge_test_theta_pulse_radius = 1.3\nouter_sponge_test_theta_pulse_width = 0.3\nouter_sponge_test_theta_pulse_dipole_axis = 1')
    active=[]
    for enabled in (False,True):
        run=args.output/('pulse_snapshots_'+str(enabled).lower())
        text=pulse if enabled else pulse.replace('debug_projection_snapshots = true','debug_projection_snapshots = false')
        if not enabled and args.operations:
            text=text.replace('debug_snapshot_operations = '+args.operations,'debug_snapshot_operations =')
        launch(args,run,text);active.append(final_state(run))
        if enabled:snapshots(run,1)
        else:
            assert not list(run.glob('z4c_projection_*'))
            assert not list(run.glob('z4c_snapshot_*'))
    assert active[0]['total']==active[1]['total']
    for a,b in zip(active[0]['state'],active[1]['state']):
        assert struct.pack('<%dd'%len(a),*a)==struct.pack('<%dd'%len(b),*b)
    assert any(x!=0 for block in active[1]['state'] for x in block)
    results['nonzero_noninterference']={'saved_Z4c_bitwise_equal':True,
                                      'blocks':active[0]['total']}
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
