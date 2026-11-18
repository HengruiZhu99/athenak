"""Exercise a real endpoint regrid, stable-GID transfer and restart in Scheme A."""
import argparse
import json
from pathlib import Path
import re
import tempfile
from qualify import identical, numerical_difference, run


def adaptive_input(ranks, *, uniform=False, intervals=3, base=2**-16, balanced=True):
    # Regrid the same initial state at the same physical cut in both integrators.
    # Each shell stays at its configured level as the Cartesian parent splits.
    ratio = 2 if uniform else 1
    text = Path(__file__).with_name('small.athinput').read_text()
    return text + f'''
<mesh_refinement>
refinement = adaptive
num_levels = 2
max_nmb_per_rank = {max(8, (90+ranks-1)//ranks)}
ncycle_check = {ratio}
refinement_interval = 0
<amr_criterion0>
method = location
location_x1 = .015
location_x2 = .013
location_x3 = .014
location_rad = .001
<time>
dt = {base/ratio:.17g}
nlim = {intervals*ratio}
<multirate>
spherical_levels = 1,1,0,1
balance = {"metis" if balanced else "none"}
uniform_reference = {str(uniform).lower()}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--launcher', default='mpiexec -n {ranks}')
    parser.add_argument('--fixed-partition', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        out = Path(tempfile.mkdtemp(prefix='adaptive-', dir=out))
    else:
        out.mkdir(parents=True)
    for name in ('mixed', 'uniform'):
        (out/(name+'.athinput')).write_text(adaptive_input(
            args.ranks, uniform=name=='uniform', balanced=not args.fixed_partition))
    def evolve(name, *options):
        return run(args.binary.resolve(), out, name, args.launcher, args.ranks, list(options))
    mixed, log = evolve('continuous', '-i', out/'mixed.athinput')
    events = re.findall(r'MULTIRATE_REGRID cycle=(\d+) created=(\d+) deleted=(\d+) blocks=(\d+)', log)
    if not events or events[0][0]!='1' or int(events[0][1])<=0 or mixed['blocks']<=25:
        raise AssertionError(f'expected actual first-endpoint refinement: {events}')
    grouped, _ = evolve('grouped', '-i', out/'mixed.athinput', 'multirate/complete_local_batches=true')
    grouped_check = identical(mixed, grouped)
    del grouped
    prefix, _ = evolve('prefix', '-i', out/'mixed.athinput', 'time/nlim=1')
    if prefix['blocks'] != int(events[0][3]):
        raise AssertionError('checkpoint predates the accepted refinement')
    resumed, _ = evolve('resumed', '-r', prefix['path'], 'time/nlim=3', 'output1/last_time=-1')
    restart = identical(mixed, resumed)
    del resumed, prefix
    uniform, _ = evolve('uniform', '-i', out/'uniform.athinput')
    if mixed['topology'] != uniform['topology'] or mixed['time'] != uniform['time']:
        raise AssertionError('different final topology or physical endpoint in uniform reference')
    uniform['cycle'] = mixed['cycle']  # comparison helper checks equal cycle as well as time
    error = numerical_difference(mixed, uniform)
    if error['max_scaled_field_difference'] > 1.e-7:
        raise AssertionError(f'adaptive reference comparison failed: {error}')
    del uniform
    coarsen_input = adaptive_input(args.ranks, intervals=2, balanced=not args.fixed_partition)
    coarsen_input = coarsen_input.replace('method = location',
                                          'method = min_max\nvariable = hyperboloidal_chi\nvalue_max = 10')
    coarsen_input += """
<refined_region0>
x1min = -.01
x1max = .01
x2min = -.01
x2max = .01
x3min = -.01
x3max = .01
level = 1
"""
    (out/'coarsen.athinput').write_text(coarsen_input)
    coarsened, coarsen_log = evolve('coarsen','-i',out/'coarsen.athinput')
    coarsened_blocks=coarsened['blocks']
    if coarsened_blocks!=25 or 'MULTIRATE_REGRID cycle=1 created=0 deleted=7 blocks=25' not in coarsen_log:
        raise AssertionError('field criterion failed to coarsen the eight Cartesian siblings')
    del coarsened
    # A restart before the first eligible check must retain the elapsed cooldown.
    delayed = (out/'mixed.athinput').read_text().replace('refinement_interval = 0',
                                                         'refinement_interval = 2')
    (out/'delayed.athinput').write_text(delayed)
    direct, log = evolve('delayed', '-i', out/'delayed.athinput','time/nlim=2')
    if 'MULTIRATE_REGRID cycle=2 created=7 deleted=0 blocks=32' not in log:
        raise AssertionError('unexpected cooldown refinement cut')
    prefix, _ = evolve('before_cut','-i',out/'delayed.athinput','time/nlim=1')
    if prefix['blocks'] != 25:
        raise AssertionError('refinement occurred before cooldown expired')
    continued, _ = evolve('from_before_cut','-r',prefix['path'],'time/nlim=2','output1/last_time=-1')
    cooldown_restart = identical(direct,continued)
    result = dict(complete_local_batches=grouped_check, events=events, coarsened_blocks=coarsened_blocks, cooldown_restart=cooldown_restart, final_blocks=mixed['blocks'], restart=restart,
                  uniform_fine_reference=error, time=mixed['time'])
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
