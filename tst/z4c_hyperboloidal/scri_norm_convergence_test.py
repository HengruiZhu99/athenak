#!/usr/bin/env python3
"""Manufactured norm-scaling/parser checks; launches no evolution."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
from scri_norm_convergence import compare, decay
from scri_run_summary import history
from scri_radial_partition import partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, base = args.output, args.native_baseline
    out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((base/'final-checkpoint.json').read_text())
    for n in [12,16,20]:
        directory = out/f'N{n}'; directory.mkdir()
        metadata = dict(receipt)
        metadata.update(identity=re.sub(r';n=12,12,12;', f';n={n},{n},{n};', metadata['identity']),
                        dt_native=metadata['dt_native']*12/n, cycle=metadata['cycle']*n//12,
                        fixture='Manufactured norm-scaling test; not an evolution checkpoint')
        (directory/'final-checkpoint.json').write_text(json.dumps(metadata))
        shutil.copy2(base/'input.athinput',directory/'input.athinput')
        shape = next(base.glob('*.horizon_shape_0.txt'))
        shutil.copy2(shape,directory/shape.name)
        paths = [next(base.glob('*.z4c_hyperboloidal.hst'))]
        paths += [next(base.glob(f'*.scri-shell-{g}.hst')) for g in range(4)]
        for path in paths:
            a, columns = history(path); a = a[-1:,:].copy(); a[:,1] *= 12/n
            selection = slice(8,12) if '.z4c_' in path.name else slice(3,7)
            a[:,selection] *= (12/n)**4
            band = next((line[2:] for line in path.read_text().splitlines()
                         if line.startswith('# Compact radius band ')), '')
            np.savetxt(directory/path.name, a, fmt='%.17e',
                       header=band+'\n'+' '.join(f'[{i+1}]={name}' for i,name in enumerate(columns)))
    shutil.copytree(out/'N12', out/'halfdt')
    path = out/'halfdt/final-checkpoint.json'; metadata = json.loads(path.read_text())
    metadata['dt_native'] /= 2; metadata['cycle'] *= 2
    path.write_text(json.dumps(metadata))
    for path in (out/'halfdt').glob('*.hst'):
        a, columns = history(path); a[:,1] /= 2
        band = next((line[2:] for line in path.read_text().splitlines()
                     if line.startswith('# Compact radius band ')), '')
        np.savetxt(path, a, fmt='%.17e', header=band+'\n'+' '.join(
            f'[{i+1}]={name}' for i,name in enumerate(columns)))
    paths = [out/f'N{n}' for n in [12,16,20]]
    result = compare(paths, out/'halfdt')
    error = max(abs(order-4) for region in result['regions'].values()
                for value in region.values() for order in value['pairwise_decay_orders'])
    assert error < 1e-10  # Complement subtraction has larger conditioning.
    assert all(value['N12_dt_change'] == 0 for region in result['regions'].values()
               for value in region.values())
    assert decay([1,1,1])['pairwise_decay_orders'] == [0.,0.]
    assert all(p < 0 for p in decay([1,2,4])['pairwise_decay_orders'])
    assert decay([0,0,0])['pairwise_decay_orders'] == [None,None]
    path = paths[-1]/'final-checkpoint.json'; original = path.read_text()
    for key, value in [('time_M0', receipt['time_M0']+.1),
                       ('identity', json.loads(original)['identity']+';different-physics=1')]:
        changed = json.loads(original); changed[key] = value
        path.write_text(json.dumps(changed))
        try:
            compare(paths)
        except AssertionError:
            pass
        else:
            raise AssertionError('mismatched '+key+' accepted')
        finally:
            path.write_text(original)
    inp = paths[0]/'input.athinput'; original = inp.read_text()
    inp.write_text(original.replace('npunct = 0','npunct = 1'))
    try:
        partition(paths[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('moving horizon input accepted')
    finally:
        inp.write_text(original)
    shell = next(paths[0].glob('*.scri-shell-0.hst')); original = shell.read_text()
    shell.write_text(original.replace('[0.8,0.88)','[0.7,0.88)'))
    try:
        partition(paths[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong shell bounds accepted')
    finally:
        shell.write_text(original)
    a, columns = history(shell); a[:,3:7] *= 100
    np.savetxt(shell,a,fmt='%.17e',header='Compact radius band [0.8,0.88)\n'+' '.join(
        f'[{i+1}]={name}' for i,name in enumerate(columns)))
    try:
        partition(paths[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('negative complement accepted')
    finally:
        shell.write_text(original)
    shape = next(paths[0].glob('*.horizon_shape_0.txt')); original = shape.read_text()
    a = np.atleast_2d(np.loadtxt(shape)); a[:,0] = .9*np.sqrt(4*np.pi)
    np.savetxt(shape,a)
    try:
        partition(paths[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('horizon intersecting outer shells accepted')
    finally:
        shape.write_text(original)
    sources = [Path(__file__), Path(__file__).with_name('scri_norm_convergence.py'),
               Path(__file__).with_name('scri_radial_partition.py')]
    report = dict(manufactured_fourth_order_max_error=error,
                  matched_halfdt_zero_difference=True, flat_growing_roundoff_cases_checked=True,
                  time_mismatch_rejected=True, physics_mismatch_rejected=True,
                  inner_complement_in_fourth_order_fixture=True,
                  moving_center_rejected=True, wrong_shell_bounds_rejected=True,
                  negative_complement_rejected=True, intersecting_horizon_rejected=True,
                  not_actual_N16_N20_evolution=True,
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
