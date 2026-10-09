"""Compile local polynomial evaluator and compare pinned core point actions."""
from pathlib import Path
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('error', category=RuntimeWarning)
P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((P/'plan.json').read_text())
pins = json.loads((P/'source-pins.json').read_text())
symbolic = json.loads((P/'symbolic-report.json').read_text())
assert symbolic['passed_exact_flat_core_envelope_derivation']
assert symbolic['source_sha256'] == sha(P/'derive_core.py')
assert symbolic['generated_header_sha256'] == sha(P/'core_envelope.hpp')
for record in pins.values():
    assert sha(record['path']) == record['sha256']
angular = Path(pins['angular_index']['path']).parent
original_builds = {}
for mode in ('release', 'debug'):
    short = json.loads((angular/('build-'+mode+'-latest.json')).read_text())
    old = Path(short['attempt'])/'receipt.json'
    assert sha(old) == short['receipt_sha256']
    receipt = json.loads(old.read_text())
    for path, digest in receipt['compiler_dependency_hashes'].items():
        assert sha(path) == digest, path
    for path, digest in receipt['link_archive_hashes'].items():
        assert sha(path) == digest, path
    original_builds[mode] = {'path': str(old), 'sha256': sha(old),
                             'dependencies_verified': len(receipt['compiler_dependency_hashes']),
                             'archives_verified': len(receipt['link_archive_hashes'])}

attempt = P/'local-attempt-001'
assert not attempt.exists(), 'Preserve all prior attempts'
attempt.mkdir()
for name in ('check_local.py', 'evaluate_core.cpp', 'core_envelope.hpp', 'plan.json'):
    shutil.copy2(P/name, attempt/name)
builds = {}
version = subprocess.check_output(['/usr/bin/c++', '--version'], text=True)
for mode in ('release', 'asan'):
    flags = ['-O3'] if mode == 'release' else ['-O0','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    exe = P/('core-envelope-'+mode)
    cmd = ['/usr/bin/c++', '-std=c++17', *flags, str(P/'evaluate_core.cpp'), '-o', str(exe)]
    started = time.monotonic()
    run = subprocess.run(cmd, text=True, capture_output=True)
    (attempt/('build-'+mode+'.stdout')).write_text(run.stdout)
    (attempt/('build-'+mode+'.stderr')).write_text(run.stderr)
    record = {'command': cmd, 'compiler_version': version, 'compiler_sha256': sha('/usr/bin/c++'),
              'exit_code': run.returncode, 'compile_seconds': time.monotonic()-started,
              'source_sha256': sha(P/'evaluate_core.cpp'), 'generated_header_sha256': sha(P/'core_envelope.hpp')}
    (attempt/('build-'+mode+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    assert run.returncode == 0, run.stderr
    depcmd = ['/usr/bin/c++', '-std=c++17', *flags, str(P/'evaluate_core.cpp'), '-M','-MT','core-envelope']
    dep = subprocess.check_output(depcmd, text=True)
    (attempt/('dependencies-'+mode+'.make')).write_text(dep)
    paths = sorted(set(str(Path(name).resolve()) for name in shlex.split(dep.replace('\\\n',' ').split(':',1)[1])))
    record.update(executable_sha256=sha(exe), compiler_dependency_hashes={path:sha(path) for path in paths})
    (attempt/('build-'+mode+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    builds[mode] = record

blocks = np.load(P/'core-envelope-blocks.npz')
saved = np.load(pins['angular_coefficients']['path'])
fitted = json.loads((angular/'angular-analysis-002.json').read_text())
fitted_comparisons = []
for J in plan['J_values']:
    for r in plan['saved_coefficient_radii']:
        for derivative in range(3):
            exact = sum(blocks['J%d'%J][derivative,power] * r**(2*power) for power in range(3))
            old = saved['J%d_r%.17g_B%d'%(J,r,derivative)]
            residual = exact-old
            fit = next(row for row in fitted['results'] if row['J']==J and row['r']==r)
            fitted_comparisons.append({'J':J,'r':r,'derivative':derivative,
                'absolute_L2':float(np.linalg.norm(residual)), 'absolute_max':float(np.max(np.abs(residual))),
                'scaled':float(np.linalg.norm(residual)/max(1,np.linalg.norm(exact))),
                'expected_L2':float(np.linalg.norm(exact)), 'saved_fit_raw_condition':fit['raw_condition'],
                'saved_fit_scaled_condition':fit['scaled_condition']})

queries = []
groups = []
for r in plan['native_probe_radii']:
    for direction in (plan['directions'] if r else [[0.,0.,0.]]):
        point = [r*v for v in direction]
        rho = sum(v*v for v in point)
        envelopes = [[1,0,0],[rho,1,0],[rho*rho,2*rho,2],
                     [rho**3,3*rho*rho,6*rho],
                     [1+.3*rho-.2*rho*rho+.1*rho**3,.3-.4*rho+.3*rho*rho,-.4+.6*rho]]
        for J in range(3):
            for m in range(J+1):
                for phase in range(2 if m else 1):
                    start = len(queries)
                    for column in range((8,16,20)[J]):
                        for envelope, w in enumerate(envelopes):
                            queries.append([J,m,column,phase,0,*point,*w])
                    groups.append({'J':J,'m':m,'phase':phase,'r':r,'point':point,
                                   'start':start,'stop':len(queries),'envelopes':len(envelopes)})
payload = ''.join(' '.join(format(value,'.17g') for value in row)+'\n' for row in queries)
(P/'queries.txt').write_text(payload)
(P/'query-metadata.json').write_text(json.dumps({'rows':len(queries),'groups':groups,
    'input_sha256':sha(P/'queries.txt'),'plan_sha256':sha(P/'plan.json')},indent=2)+'\n')

run_records = {}
outputs = {}
commands = {'native-release':[pins['bridge_release']['path'],'--batch'],
            'native-asan':[pins['bridge_asan']['path'],'--batch'],
            'polynomial-release':[str(P/'core-envelope-release')],
            'polynomial-asan':[str(P/'core-envelope-asan')]}
for name, command in commands.items():
    started = time.monotonic()
    run = subprocess.run(command,input=payload,text=True,capture_output=True)
    out = P/(name+'.txt')
    out.write_text(run.stdout)
    (P/(name+'.stderr')).write_text(run.stderr)
    record = {'command':command,'exit_code':run.returncode,'seconds':time.monotonic()-started,
              'executable_sha256':sha(command[0]), 'input_sha256':sha(P/'queries.txt'),
              'output_sha256':sha(out),'stderr_bytes':len(run.stderr.encode())}
    run_records[name] = record
    (P/'run-receipts.json').write_text(json.dumps(run_records,indent=2)+'\n')
    assert run.returncode == 0, run.stderr
    raw = np.fromstring(run.stdout,sep=' ')
    width = 49 if name.startswith('native') else 44
    assert raw.size == len(queries)*width
    outputs[name] = raw.reshape(len(queries),width)
    assert np.isfinite(outputs[name]).all()


def errors(a,b):
    residual = a-b
    row = np.linalg.norm(residual,axis=1)
    expected = np.linalg.norm(b,axis=1)
    return {'absolute_L2':float(np.linalg.norm(residual)), 'absolute_max':float(np.max(np.abs(residual))),
            'aggregate_scaled':float(np.linalg.norm(residual)/max(1,np.linalg.norm(b))),
            'per_action_scaled':float(np.max(row/np.maximum(1,expected))),
            'expected_L2':float(np.linalg.norm(b)), 'expected_action_norm_max':float(expected.max()),
            'expected_exact_zero_actions':int((expected==0).sum()),
            'expected_nearzero_nonzero_actions_below1e_minus12':int(((expected>0)&(expected<1e-12)).sum()),
            'expected_smallest_nonzero_action_norm':float(expected[expected>0].min()) if (expected>0).any() else None}


native = outputs['native-release']
polynomial = outputs['polynomial-release']
rhs_error = errors(native[:,22:44],polynomial[:,22:44])
input_error = errors(native[:,:22],polynomial[:,:22])
asan_native = errors(outputs['native-asan'][:,22:44],native[:,22:44])
asan_polynomial = errors(outputs['polynomial-asan'][:,22:44],polynomial[:,22:44])
normal_scale = np.maximum(1,np.linalg.norm(native[:,22:44],axis=1))
normal_error = float(np.max(np.abs(native[:,44:48])/normal_scale[:,None]))
assert np.all(native[:,48]==1), 'Probe left exact Cauchy core'
by_group = []
for group in groups:
    start,stop = group['start'],group['stop']
    by_group.append({**group,'rhs_error':errors(native[start:stop,22:44],polynomial[start:stop,22:44]),
                     'input_error':errors(native[start:stop,:22],polynomial[start:stop,:22])})
T = plan['tolerances']
checks = {'exact_symbolic_all_m':symbolic['all_exact_cartesian_residuals_zero'],
          'no_radial_denominators':not symbolic['r_or_rho_denominators'],
          'no_cross_L_conditions':not symbolic['imposed_cross_L_envelope_conditions'],
          'native_full22':rhs_error['per_action_scaled']<=T['native_full22_scaled'],
          'input_lift_layout':input_error['per_action_scaled']<=T['native_full22_scaled'],
          'actual_raw_normals':normal_error<=T['raw_algebraic_normal_scaled'],
          'saved_angular_core_blocks':max(v['scaled'] for v in fitted_comparisons)<=T['saved_fitted_coefficient_scaled'],
          'Release_ASan_native':asan_native['per_action_scaled']<=T['Release_ASan_numeric_scaled'],
          'Release_ASan_polynomial':asan_polynomial['per_action_scaled']<=T['Release_ASan_numeric_scaled'],
          'all_point_Omega_exact_one':bool(np.all(native[:,48]==1)),
          'all_processes_exit_zero':all(v['exit_code']==0 for v in run_records.values()),
          'all_stderr_empty':all(v['stderr_bytes']==0 for v in run_records.values())}
report = {'scope':plan['scope'],'passed_local_flat_core_envelope_gate':all(checks.values()),'checks':checks,
          'rows':len(queries),'exact_symbolic_counts':symbolic['counts'],
          'full22_rhs_error':rhs_error,'full22_input_lift_error':input_error,
          'native_Release_ASan_rhs_error':asan_native,'polynomial_Release_ASan_rhs_error':asan_polynomial,
          'raw_algebraic_normal_scaled':normal_error,'saved_fitted_core_comparisons':fitted_comparisons,
          'groups':by_group,'runs':run_records,'source_sha256':sha(__file__),
          'plan_sha256':sha(P/'plan.json'),'symbolic_report_sha256':sha(P/'symbolic-report.json'),
          'original_native_bridge_dependency_verification':original_builds,
          'python':sys.version,'numpy':np.__version__,'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
          'actual_radial_PDE_global_matrix_boundary_or_evolution_admitted':False}
np.savez_compressed(P/'local-actions.npz',**outputs)
report['actions_npz_sha256'] = sha(P/'local-actions.npz')
(P/'local-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('groups','saved_fitted_core_comparisons','runs')},indent=2),flush=True)
assert report['passed_local_flat_core_envelope_gate']
