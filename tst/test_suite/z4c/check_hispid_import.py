"""Check a checkpoint's initial-time AthenaK import, without horizon searches.

Passing this check qualifies interchange only. Constraint and horizon
acceptance remain properties of separately retained physical investigations.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

import check_hispid_binary
import fastflow_storage
import hispid_sampler_proof
from check_hispid_binary import checkpoint_metadata
from hispid_sampler_proof import digest, import_evidence, validate_migration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--migration-proof')
    p.add_argument('--allow-diagnostic', action='store_true')
    p.add_argument('--domain-half-width', type=float, default=32.)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--template', default=str(Path(__file__).resolve().parents[2] / 'inputs/hispid.athinput'))
    a = p.parse_args()
    if not math.isfinite(a.domain_half_width) or a.domain_half_width <= 0 or a.timeout < 1:
        raise ValueError('positive finite domain and positive timeout required')
    exe = Path(a.executable).resolve(strict=True)
    template = Path(a.template).resolve(strict=True)
    source = checkpoint_metadata(Path(a.checkpoint).resolve(strict=True), require_two_active=False)
    if source['acceptance'] not in ('analytic_seed', 'preliminary', 'strong') and not (
            a.allow_diagnostic and source['acceptance'] == 'diagnostic'):
        raise ValueError('checked checkpoint or explicit diagnostic import required')
    if any(hole[0] > 0 and max(abs(x) for x in hole[1:4]) + radius >= a.domain_half_width
           for hole, radius in zip(source['holes'], source['inner_max'])):
        raise ValueError('domain must contain every active hole and modified ball')
    migration = validate_migration(a.migration_proof, source) if a.migration_proof else None
    bound = {str(path): digest(path) for path in (exe, template, Path(source['path']), Path(__file__))}
    for module in (check_hispid_binary, hispid_sampler_proof, fastflow_storage):
        path = Path(module.__file__).resolve(strict=True)
        bound[str(path)] = digest(path)

    def verify():
        if any(digest(path) != sha for path, sha in bound.items()):
            raise ValueError('bound import input, source, executable or log changed')
        if migration and validate_migration(a.migration_proof, source) != migration:
            raise ValueError('checkpoint sampler proof changed')

    verify()
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    input_path = root / 'import.athinput'
    input_path.write_bytes(template.read_bytes())
    bound[str(input_path)] = digest(input_path)
    cmd = [str(exe), '-i', str(input_path),
           'problem/hispid_filename=' + source['path'],
           'problem/hispid_source_sha256=' + source['source_library_sha256'],
           'problem/hispid_initial_horizons=false', 'fastflow/num_horizons=0',
           'problem/hispid_flat_control=false', 'problem/hispid_mesh_constraints=false',
           'time/nlim=0', 'time/tlim=0']
    if a.allow_diagnostic:
        cmd.append('problem/hispid_allow_diagnostic=true')
    if migration:
        cmd.append('problem/hispid_allow_library_migration=true')
    for d in (1, 2, 3):
        cmd += [f'mesh/nx{d}=8', f'meshblock/nx{d}=8',
                f'mesh/x{d}min={-a.domain_half_width}', f'mesh/x{d}max={a.domain_half_width}']
    result = dict(schema='hispid_initial_time_import_v1', purpose='interchange_only',
                  source=source, sampler_migration=migration, command=cmd,
                  bound_artifacts_sha256=bound, completed=False, passed=False,
                  physical_acceptance=False, horizon_search_performed=False,
                  criteria=dict(adm_z4c_relative_error=1e-11, time=0, cycle=0),
                  domain_half_width=a.domain_half_width)

    def save():
        verify()
        (root / 'import.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    log_path = root / 'run.log'
    with log_path.open('w') as log:
        try:
            run = subprocess.run(cmd, cwd=root, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=a.timeout)
            result['returncode'] = run.returncode
        except subprocess.TimeoutExpired:
            result['returncode'] = 'timeout'
    bound[str(log_path)] = digest(log_path)
    save()
    stdout = log_path.read_text()
    result['import'] = import_evidence(stdout, source, migration)
    result['zero_evolution_verified'] = bool(
        re.search(r'time=0\.000000e\+00 cycle=0', stdout) and 'MeshBlock-cycles = 0' in stdout)
    result['no_finder_verified'] = 'HiSpID horizon ' not in stdout and 'FastFlow harmonic_storage' not in stdout
    result['bound_inputs_unchanged'] = True
    result['completed'] = True
    result['passed'] = bool(result['returncode'] == 0 and result['import']['passed']
                            and result['zero_evolution_verified'] and result['no_finder_verified'])
    result['note'] = 'Import and round-trip evidence only; checkpoint physical acceptance is unchanged.'
    save()
    print(json.dumps({k: result[k] for k in ('passed', 'import', 'zero_evolution_verified', 'no_finder_verified')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
