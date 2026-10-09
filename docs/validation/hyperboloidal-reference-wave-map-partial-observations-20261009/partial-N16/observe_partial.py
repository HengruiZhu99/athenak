"""HELD stopped-failure observer; partial diagnostics can never accept a run.

Uses unchanged binary64 reader/ABI and accepted native probe. Never invokes or
changes the completed-result c1b948 analyzer, original input or failed receipt.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LONG = ROOT / 'build-layer-research/wave-map-native-t2-root-20261009'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def finite(x):
    if isinstance(x, dict):
        for value in x.values(): finite(value)
    elif isinstance(x, list):
        for value in x: finite(value)
    elif isinstance(x, float): assert math.isfinite(x), 'nonfinite JSON number'

def load(path):
    x = json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    finite(x)
    return x

def dump(path, x):
    finite(x)
    Path(path).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def checked(pins):
    for name, digest in pins.items(): assert sha(name) == digest, name

def diagnose_history(path):
    info = {'path': str(path), 'sha256': sha(path), 'original_bytes': path.stat().st_size,
            'complete_history_guards_satisfied': False}
    data = None
    try:
        data = np.atleast_2d(np.loadtxt(path))
        assert data.shape[1] == 15, 'history column count'
        assert np.isfinite(data).all(), 'nonfinite history values'
        assert (data[:, 1] > 0).all(), 'nonpositive history dt'
        assert (np.diff(data[:, 0]) > 0).all(), 'history times not strictly increasing'
        info.update(complete_history_guards_satisfied=True, rows=data.tolist(),
            last_history_time=float(data[-1, 0]), last_H_Mcon_Zcon_Theta=data[-1, 2:6].tolist())
    except Exception as exc:
        info['diagnostic_guard_failure'] = repr(exc)
        data = None
    return info, data

def array_observation(number, path, reader, probe, n, parameters, profile, history, out, previous_time):
    q = {'rst_path': str(path), 'rst_sha256': sha(path), 'partial_diagnostic_only': True,
         'accepted_native_run': False, 'diagnostic_guard_failures': [], 'probe_called': False}
    call = None
    try:
        rst = reader.read_restart(path)
        q.update(time=rst['time'], cycle=rst['cycle'], restart_header_dt=rst['dt'])
        assert rst['time'] > previous_time, 'restart times not strictly increasing'
        assert rst['mb_indcs']['ng'] == 3
        for axis in [1, 2, 3]:
            assert rst['mb_indcs'][f'nx{axis}'] == n
            assert rst['mesh_indcs'][f'nx{axis}'] == n
            assert rst['mesh_size'][f'x{axis}min'] == -1.1
            assert rst['mesh_size'][f'x{axis}max'] == 1.1
            assert rst['mesh_size'][f'dx{axis}'] == 2.2 / n
        for key in parameters:
            assert rst['parameters'][key] == parameters[key], key
        raw = np.asarray(rst['data'])
        assert raw.dtype == np.dtype('<f8')
        assert raw.shape == (1, 25, n + 6, n + 6, n + 6)
        h = rst['mesh_size']['dx1']
        first = -1.1 + (0.5 - 3) * h
        coordinates = first + np.arange(n + 6, dtype=float) * h
        z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
        mask = x*x + y*y + z*z < 1
        active = raw[0][:, mask]
        q['active_count'] = int(mask.sum())
        q['nonfinite_count25'] = np.count_nonzero(~np.isfinite(active), axis=1).tolist()
        q['finite_extrema25'] = []
        for row in active:
            values = row[np.isfinite(row)]
            q['finite_extrema25'].append({'min': float(values.min()) if len(values) else None,
                                        'max': float(values.max()) if len(values) else None})
        if not np.isfinite(active).all():
            index = np.argwhere(~np.isfinite(active))[0]
            q['first_nonfinite_field_active_index_value'] = [int(index[0]), int(index[1]), repr(float(active[tuple(index)]))]
            raise ValueError('nonfinite active full25 fields; no native constraint call')
        alpha = raw[0, 18][mask]
        chi = raw[0, 0][mask]
        q.update(alpha_min=float(alpha.min()), chi_min=float(chi.min()))
        assert (alpha > 0).all(), 'nonpositive active alpha; no native constraint call'
        assert (chi > 0).all(), 'nonpositive active chi; no native constraint call'
        metric = np.zeros((int(mask.sum()), 3, 3))
        for field, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            metric[:, i, j] = metric[:, j, i] = raw[0, 1 + field][mask]
        eigen = np.linalg.eigvalsh(metric)
        q['minimum_conformal_metric_eigenvalue'] = float(eigen.min())
        assert np.isfinite(eigen).all() and (eigen[:, 0] > 0).all(), 'non-SPD active metric; no native constraint call'
        q['minimum_Penrose_spatial_metric_eigenvalue'] = float((eigen / chi[:, None]).min())
        # These are field-admission eigenvalues, never a PDE/operator spectrum.
        payload = raw.tobytes(order='C')
        metadata = out / f'{number:04d}-array-input-metadata.json'
        dump(metadata, {'rst_path': str(path), 'rst_sha256': sha(path),
            'payload_sha256': hashlib.sha256(payload).hexdigest(), 'payload_bytes': len(payload),
            'schema': 'little endian binary64 [1,25,k,j,i], all stored cells',
            'rst_time': rst['time'], 'rst_dt': rst['dt'], 'cycle': rst['cycle'],
            'partial_diagnostic_only': True, 'accepted_native_run': False})
        command = [str(probe), '--snapshot', str(n)]
        result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        so = out / f'{number:04d}-probe.stdout'
        se = out / f'{number:04d}-probe.stderr'
        so.write_bytes(result.stdout)
        se.write_bytes(result.stderr)
        call = {'command': command, 'returncode': result.returncode, 'stdout_sha256': sha(so),
            'stderr_sha256': sha(se), 'input_metadata_sha256': sha(metadata),
            'partial_diagnostic_only': True, 'accepted_native_run': False}
        q['probe_called'] = True
        assert result.returncode == 0 and result.stderr == b'', 'native diagnostic probe failure'
        native = load(so)
        assert native['active_count'] == int(mask.sum())
        q['native_diagnostics'] = native
        q['gauge_pole_diagnostic_scope'] = 'The unchanged probe evaluates wave-map value-only gauge poles on the supplied fields; these are not the evolved C0 gauge poles when observing a C0 case.'
        reference = profile == 0
        tolerance = 1e-11 if reference else 1e-10
        if native['det_max'] > tolerance: q['diagnostic_guard_failures'].append('determinant threshold exceeded')
        if native['trace_max'] > tolerance: q['diagnostic_guard_failures'].append('A-trace threshold exceeded')
        expected = np.asarray(native['rms_H_Mcon_Zcon_Theta'])
        if rst['time'] == 0:
            if native['initial_profile_max_error_reference_small_large'][profile] > 2e-13:
                q['diagnostic_guard_failures'].append('runtime initializer threshold exceeded')
            if np.max(expected) > 1e-9: q['diagnostic_guard_failures'].append('initial constraint threshold exceeded')
        if reference:
            if max(native['reference_deviation_max25']) > 1e-10:
                q['diagnostic_guard_failures'].append('reference drift threshold exceeded')
            if np.max(expected) > 1e-9: q['diagnostic_guard_failures'].append('reference constraint threshold exceeded')
        if history is None:
            q['diagnostic_guard_failures'].append('complete history guards unavailable')
        else:
            matches = np.flatnonzero(np.abs(history[:, 0] - rst['time']) <= 1e-12)
            if len(matches) != 1:
                q['diagnostic_guard_failures'].append('RST/history exact-time pairing failed')
            else:
                row = history[int(matches[0])]
                error = np.max(np.abs(row[2:6] - expected) / np.maximum(1., np.abs(expected)))
                q.update(history_dt=float(row[1]), history_scaled_rms_error=float(error))
                if error > 2e-11: q['diagnostic_guard_failures'].append('history/native RMS threshold exceeded')
        q['structural_and_native_diagnostic_call_completed'] = True
    except Exception as exc:
        q['diagnostic_guard_failures'].append(repr(exc))
        q['structural_and_native_diagnostic_call_completed'] = False
    return q, call

def main():
    np.seterr(all='raise')
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'root invocation must disable bytecode writes'
    release_path = Path(sys.argv[1]).resolve()
    name = sys.argv[2]
    release = load(release_path)
    recipe_path = HERE / 'recipe.json'
    recipe = load(recipe_path)
    assert release['partial_native_snapshot_observation_authorized'] is True
    assert release['observer_sha256'] == sha(Path(__file__))
    assert release['recipe_sha256'] == sha(recipe_path)
    assert name in recipe['cases'] and name in release['cases']
    case_release = release['cases'][name]
    out = Path(case_release['attempt_directory']).resolve()
    assert out.parent == (HERE / 'attempts').resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'observer.py').write_bytes(Path(__file__).read_bytes())
    (out / 'release.json').write_bytes(release_path.read_bytes())
    (out / 'recipe.json').write_bytes(recipe_path.read_bytes())
    started = time.monotonic()
    record = {'partial_diagnostic_only': True, 'accepted_native_run': False, 'observer_completed': False,
        'case': name, 'observer_sha256': sha(Path(__file__)), 'recipe_sha256': sha(recipe_path),
        'root_release_sha256': sha(release_path),
        'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Stopped failed native run observations only. No completed-run acceptance, repaired state, future continuation or continuum-stability claim.'}
    observations = []
    calls = []
    try:
        protected = dict(recipe['fixed_pins'])
        protected[str(Path(__file__).resolve())] = release['observer_sha256']
        protected[str(recipe_path)] = release['recipe_sha256']
        protected[str(release_path)] = sha(release_path)
        checked(protected)
        spec = recipe['cases'][name]
        launch_path = LONG / 'batch001' / name / 'launch-receipt.json'
        assert sha(launch_path) == case_release['failed_launch_receipt_sha256']
        launch = load(launch_path)
        assert launch['passed_native_process_and_provenance'] is False
        assert launch['returncode'] != 0 or launch['error'] is not None, 'a successful completed case belongs to the completed-result wrapper'
        assert launch['sources_before_after_equal'] is True, 'unresolved original source provenance; observation held'
        assert launch['mode'] == spec['mode'] and launch['input_path'] == spec['input_path']
        assert launch['native_execution_authorization_sha256'] == recipe['native_execution_release_sha256']
        build = load(Path(spec['build_receipt']))
        assert build['passed_compile_link'] is True
        assert build['compiled_implementation'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'
        assert launch['executable'] == build['executable'] and launch['executable_sha256'] == build['executable_sha256']
        assert launch['build_receipt'] == spec['build_receipt'] and launch['build_receipt_sha256'] == sha(Path(spec['build_receipt']))
        assert launch['command'] == [build['executable'], '-i', spec['input_path']]
        directory = launch_path.parent / 'output'
        assert Path(launch['cwd']).resolve() == directory and Path(launch['output_directory']).resolve() == directory
        protected.update(build['source_before'])
        protected.update(build['all_compiler_dependency_sha256'])
        protected.update(spec['required_hashes'])
        before = launch_path.with_name('protected-inputs-before.json')
        after = launch_path.with_name('protected-inputs-after.json')
        assert before.read_bytes() == after.read_bytes()
        protected.update(load(before))
        for p in [before, after, launch_path]: protected[str(p)] = sha(p)
        actual_outputs = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        assert actual_outputs == set(launch['outputs']), 'stopped output inventory changed'
        for relative, item in launch['outputs'].items():
            p = directory / relative
            assert p.stat().st_size == item['bytes']
            protected[str(p)] = item['sha256']
        protected[str(launch['run_log'])] = launch['run_log_sha256']
        protected[str(launch['stderr_path'])] = launch['stderr_sha256']
        probe_recipe = load(Path(recipe['probe_recipe']))
        seam = load(Path(recipe['seam_receipt']))
        assert seam['passed_compile_and_fixed_t0_seam'] is True
        assert seam['probe_executable_sha256'] == recipe['probe_executable_sha256']
        protected.update(probe_recipe['source_before'])
        protected.update(seam['source_before'])
        protected.update(seam['all_compiler_dependency_sha256'])
        checked(protected)
        dump(out / 'protected-inputs-before.json', protected)
        (out / 'original-failed-launch-receipt.json').write_bytes(launch_path.read_bytes())
        spec_loader = importlib.util.spec_from_file_location('unchanged_binary64_restart_reader', recipe['reader'])
        reader = importlib.util.module_from_spec(spec_loader)
        spec_loader.loader.exec_module(reader)
        parameters = reader.parameters(Path(spec['input_path']).read_text())
        critical = {'mesh/nghost': '3', 'problem/mass': '0', 'problem/pulse_angular': 'true',
            'problem/pulse_width': '.35', 'z4c/hyperboloidal_curvature_radius': '.5',
            'z4c/hyperboloidal_layer_r0': '.05', 'z4c/hyperboloidal_layer_r1': '.95',
            'z4c/hyperboloidal_kappa1': '10', 'z4c/hyperboloidal_dissipation': '.1',
            'z4c/hyperboloidal_pole_cfl': '.03', 'z4c/hyperboloidal_ghost_degree': '2',
            'z4c/hyperboloidal_symmetric_ghosts': 'true',
            'z4c/hyperboloidal_physical_trace_lapse': 'true', 'z4c/hyperboloidal_preferred_source': 'false',
            'time/integrator': 'rk3', 'time/cfl_number': '.1'}
        for key, value in critical.items(): assert parameters[key] == value, key
        assert float(parameters['time/tlim']) == 2.0
        n = int(parameters['mesh/nx1'])
        assert n in [16, 24, 32]
        for section in ['mesh', 'meshblock']:
            for axis in [1, 2, 3]: assert int(parameters[f'{section}/nx{axis}']) == n
        for axis in [1, 2, 3]:
            assert float(parameters[f'mesh/x{axis}min']) == -1.1
            assert float(parameters[f'mesh/x{axis}max']) == 1.1
        profile = {(0., 0.): 0, (.02, .01): 1, (.2, .1): 2}[(float(parameters['problem/lapse_pulse']), float(parameters['problem/shift_pulse']))]
        history_paths = sorted(directory.glob('*.z4c.user.hst'))
        history_info = {'complete_history_guards_satisfied': False, 'diagnostic_guard_failure': 'exactly one history file unavailable'}
        history = None
        if len(history_paths) == 1: history_info, history = diagnose_history(history_paths[0])
        dump(out / 'history-observation.json', history_info)
        paths = sorted(directory / relative for relative in launch['outputs'] if relative.endswith('.rst'))
        assert paths, 'no complete saved restart file available to observe'
        previous = -1.
        for number, path in enumerate(paths):
            q, call = array_observation(number, path, reader, Path(recipe['probe_executable']), n, critical, profile, history, out, previous)
            if 'time' in q: previous = q['time']
            observations.append(q)
            if call is not None: calls.append(call)
        stdout_text = Path(launch['run_log']).read_text()
        console = [{'cycle': int(c), 'time_printed': float(t), 'dt_printed': float(d)}
            for c, t, d in re.findall(r'cycle=(\d+)\s+time=([\d.eE+\-]+)\s+dt=([\d.eE+\-]+)', stdout_text)]
        console_info = {'matched_rows': len(console), 'precision_scope': 'Six-digit console values; not exact RST/history dt.'}
        if console:
            console_info.update(first=console[0], last=console[-1],
                minimum_printed_dt=min(q['dt_printed'] for q in console),
                every_matched_printed_dt_positive=all(q['dt_printed'] > 0 for q in console))
        dump(out / 'console-observation.json', console_info)
        dump(out / 'snapshot-observations.json', observations)
        checked(protected)
        assert actual_outputs == {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        dump(out / 'protected-inputs-after.json', protected)
        times = [q['time'] for q in observations if 'time' in q]
        t0_present = any(q.get('time') == 0 for q in observations)
        record.update(observer_completed=True, N=n, native_returncode=launch['returncode'], native_error=launch['error'],
            original_target_time=2.0, target_reached_in_available_saved_times=bool(times and max(times) == 2.0),
            last_available_saved_time=max(times) if times else None, saved_restart_files=len(paths),
            genuine_t0_snapshot_available=t0_present,
            t0_scope_limit=None if t0_present else 'Genuine t0 snapshot unavailable; initializer/initial-constraint checks cannot be supplied.',
            arrays_with_diagnostic_guard_failures=sum(bool(q['diagnostic_guard_failures']) for q in observations),
            native_probe_calls=len(calls), calls=calls, protected_before_after_equal=True,
            failed_launch_receipt_sha256=sha(launch_path),
            observations_sha256=sha(out / 'snapshot-observations.json'),
            history_observation_sha256=sha(out / 'history-observation.json'),
            console_observation_sha256=sha(out / 'console-observation.json'),
            native_stderr_sha256=launch['stderr_sha256'], native_stdout_sha256=launch['run_log_sha256'])
    except Exception as exc:
        record.update(observer_protocol_error=repr(exc), completed_snapshot_observations=observations, calls=calls)
    record['seconds'] = time.monotonic() - started
    dump(out / 'receipt.json', record)
    print(json.dumps({'observer_completed': record['observer_completed'], 'accepted_native_run': False,
        'partial_diagnostic_only': True, 'case': name, 'receipt': str(out / 'receipt.json'),
        'protocol_error': record.get('observer_protocol_error')}), flush=True)
    assert record['observer_completed'] is True, 'partial observer protocol stop; never a native acceptance verdict'

if __name__ == '__main__':
    main()
