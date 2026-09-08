#!/usr/bin/env python3
"""Isolated N256 bisection using policy-accepted MOTS candidates by t=50.

This measures a finite-resolution, finite-time search threshold, not certified
MOTS nonexistence. It never reads lapse to classify a run. The input template
preserves the previous campaign's physical settings and live AMR policy.
"""
import argparse
import csv
from decimal import Decimal, getcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

getcontext().prec = 40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def setparam(text, section, key, value):
    pattern = r'(?ms)(^<'+re.escape(section)+r'>\s*\n)(.*?)(?=^<|\Z)'
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise RuntimeError('Missing/duplicate section '+section)
    match = matches[0]
    body = match[2]
    line = r'(?m)^\s*'+re.escape(key)+r'\s*=.*$'
    if len(re.findall(line, body)) > 1:
        raise RuntimeError('Duplicate key '+key)
    body = re.sub(line, key+' = '+str(value), body) if re.search(line, body) else body+key+' = '+str(value)+'\n'
    return text[:match.start()]+match[1]+body+text[match.end():]


def classify(case, bound=.01, require_outermost=False):
    if require_outermost:
        if not list(case.glob("*.mots_selection.csv")):
            raise RuntimeError("Missing runtime outermost-selection diagnostics")
        for path in case.rglob("*.mots_selection.csv"):
            if any(r["status"] == "ambiguous_enclosure" for r in csv.DictReader(path.open())):
                raise RuntimeError("Ambiguous enclosure is not nondetection")
    header = None
    final = None
    for line in next(case.glob('*.hst')).read_text().splitlines():
        if line.startswith('#'):
            if '[1]=' in line:
                header = {name: int(i)-1 for i, name in re.findall(r'\[(\d+)\]=(\S+)', line)}
        elif line.strip():
            values = [float(x) for x in line.split()]
            if not header or len(values) != len(header) or not all(map(math.isfinite, values)):
                raise RuntimeError('Invalid evolution history')
            final = {key: values[i] for key, i in header.items()}
    if not final:
        raise RuntimeError('Missing evolution history')
    accepted = []
    for path in case.glob('*.mots_candidates.csv'):
        accepted += [r for r in csv.DictReader(path.open()) if r['policy_accepted'] == '1']
    if accepted:
        if not all(0 <= float(r['time']) <= 50.0000001 and
                   math.isfinite(float(r['epsilon2'])) and float(r['epsilon2']) <= bound
                   for r in accepted):
            raise RuntimeError('Invalid accepted-candidate evidence')
        termination = json.loads(next(case.glob('*.termination.json')).read_text())
        if termination['outcome'] not in ('mots_candidate', 'collapse'):
            raise RuntimeError('Candidate/termination disagreement')
        if not all(math.isfinite(v) for v in termination.values() if isinstance(v, (float, int))):
            raise RuntimeError('Nonfinite termination diagnostics')
        return dict(classification='candidate_detected', accepted=accepted,
                    termination=termination, final=final)
    if list(case.glob('*.termination.json')):
        raise RuntimeError('Unexpected stopping condition without candidate')
    if not final or abs(final['time']-50) > 1e-8:
        raise RuntimeError('Nondetection run did not reach t=50')
    if 'Terminating on time limit' not in (case/'stdout.log').read_text():
        raise RuntimeError('Unexpected evolution termination')
    if not list(case.glob('*.mots_candidates.csv')):
        raise RuntimeError('Finder produced no evidence')
    # A cycle-cadenced runtime search need not land exactly on t=50. Always
    # include the final saved slice, as in the recovered endpoint survey.
    analysis = case/'final-mots'
    frozen = json.loads((analysis/'search/frozen_mots.json').read_text())
    manifest = json.loads((analysis/'manifest.json').read_text())
    if (manifest.get('returncode') != 0 or not manifest.get('checkpoint_unchanged') or
            not frozen.get('active_state_unchanged') or not frozen.get('mesh_unchanged') or
            abs(frozen['time']-50) > 1e-8):
        raise RuntimeError('Invalid final-slice analysis')
    candidates = [r for r in csv.DictReader((analysis/'search/mots.mots_candidates.csv').open())
                  if r['policy_accepted'] == '1']
    if bool(candidates) != frozen['candidate_detected']:
        raise RuntimeError('Final-slice candidate evidence disagrees')
    if candidates:
        if not all(math.isfinite(float(r['epsilon2'])) and float(r['epsilon2']) <= bound
                   for r in candidates):
            raise RuntimeError('Invalid final-slice candidate norm')
        return dict(classification='candidate_detected', final=final,
                    accepted=candidates, frozen_final=frozen)
    return dict(classification='no_candidate_through_t50', final=final, frozen_final=frozen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['survey', 'baseline', 'athena', 'output']:
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--search-script', type=Path,
                   default=Path(__file__).with_name('search_checkpoint.py'))
    p.add_argument('--relative-tolerance', default='0.000001')
    p.add_argument('--outer05', action='store_true',
                   help='Monotonic L8/16/32, L64 RMS0.05, geometric outermost selection')
    args = p.parse_args()
    tolerance = Decimal(args.relative_tolerance)
    if not tolerance.is_finite() or not 0 < tolerance < 1:
        p.error('relative tolerance must be finite and between zero and one')
    bound = .05 if args.outer05 else .01

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'state.json').exists():
        raise RuntimeError('Existing state requires explicit recovery')
    survey = json.loads(args.survey.read_text())
    if survey.get('status') != 'complete':
        raise RuntimeError('Saved-slice survey is incomplete')
    cases = sorted(survey['cases'], key=lambda c: Decimal(c['amplitude']))
    positives = [c for c in cases if c['classification'] == 'candidate_detected']
    negatives = [c for c in cases if c['classification'] == 'no_candidate_on_saved_slices']
    if not positives or not negatives or len(positives)+len(negatives) != len(cases):
        raise RuntimeError('No complete two-sided candidate bracket')
    sup = max(Decimal(c['amplitude']) for c in positives)
    sub = min(Decimal(c['amplitude']) for c in negatives)
    if sub <= sup:
        raise RuntimeError('Nonmonotone saved-slice classifications; do not bisect')
    template = (args.baseline/'template.athinput').read_text()
    if not re.search(r'amr_history_mode\s*=\s*record', template):
        raise RuntimeError('Expected live AMR recording in original template')
    state = dict(status='STARTING', created=time.time(), pid=os.getpid(),
                 criterion=('angular_l32 outermost RMS<=0.05 by t=50' if args.outer05 else 'angular_candidate by t=50'), spatially_validated=False,
                 sub=str(sub), super=str(sup), relative_tolerance=str(tolerance),
                 survey_sha256=sha(args.survey), executable_sha256=sha(args.athena),
                 controller_sha256=sha(Path(__file__)),
                 baseline_template_sha256=sha(args.baseline/'template.athinput'), completed=[])
    save(root/'state.json', state)
    # Preserve qualified initial-data generation and binding logic byte-for-byte.
    oldrunner = (args.baseline/'run_cycle.sh').read_text()
    runner = oldrunner.replace('campaign=$(dirname -- "$case_dir")',
                              'campaign='+str(args.baseline.resolve()))
    runner = runner.replace('"$campaign/athena.history_extrema"', '"'+str(args.athena.resolve())+'"')
    runner += '\nif [[ ! -f "$case_dir/mots_bisect.termination.json" ]]; then\n'
    runner += '  checkpoint=$(ls "$case_dir"/rst/*.rst | sort | tail -n 1)\n'
    runner += ('  "$python" "'+str(args.search_script.resolve())+'" --athena "'+
               str(args.athena.resolve())+'" --checkpoint "$checkpoint" '+
               '--output "$case_dir/final-mots" --lmax 128 --l-start 8 --radii 4 '+
               '--iterations 500 --profile-points 1061 '+
               '--launcher "${step[*]}" > final-search.log 2>&1\nfi\n')
    if args.outer05:
        runner = runner.replace('--lmax 128 --l-start 8 --radii 4',
                                '--lmax 64 --l-start 8 --radii 8 --radius-max 8 '
                                '--detection angular_l32 --candidate-bound 0.05 '
                                '--selection outermost')
    else:
        runner = runner.replace('--iterations 500 --profile-points 1061',
                                '--candidate-bound 0.01 --selection residual '
                                '--iterations 500 --profile-points 1061')
    (root/'run_cycle.sh').write_text(runner)
    finder = dict(lmax=128, ntheta=260, mots_l_start=8, flow_iterations_0=500,
                  mots_radius_count=4, mots_detection='angular_candidate',
                  mots_level_iterations=96, mots_candidate_bound=.01,
                  mots_promotion_max=.5, mots_angular_ratio=.8, mots_candidate_points=1061,
                  mots_shape_change=.02, mots_area_change=.01, mots_epsilon2=1e-6,
                  mots_epsilon_inf=1e-5, mots_write_profiles='true', mots_profile_points=1061)
    if args.outer05:
        finder.update(lmax=64, ntheta=132, mots_detection='angular_l32',
                      mots_candidate_bound=.05, mots_selection='outermost',
                      mots_radius_count=8, mots_radius_min=0, initial_radius_0=8)
    else:
        finder['mots_selection'] = 'residual'
    state['finder'] = finder
    save(root/'state.json', state)
    try:
        for iteration in range(1, 21):
            width = abs(sub-sup)/abs(sup)
            state['relative_width'] = str(width)
            if width <= tolerance:
                state.update(status='COMPLETE', finished=time.time())
                save(root/'state.json', state)
                return
            if (root/'STOP').exists():
                raise RuntimeError('STOP requested')
            if sha(args.athena) != state['executable_sha256']:
                raise RuntimeError('Executable changed during bisection')
            amplitude = (sub+sup)/2
            case = root/('cycle_%02d'%iteration)
            case.mkdir()
            (case/'amplitude.txt').write_text(str(amplitude)+'\n')
            inp = template
            for section, key, value in [('job','basename','mots_bisect'),
                    ('mesh_refinement','amr_history_file',case/'amr_history.jsonl'),
                    ('problem','brill_global_coefficients_file','initial.coefficients'),
                    ('problem','constraint_summary_file','initial-constraints.dat')]:
                inp = setparam(inp, section, key, value)
            for key, value in finder.items():
                inp = setparam(inp, 'fastflow', key, value)
            (case/'input.athinput').write_text(inp)
            command = ['salloc','--account=m3328_g','--qos=shared_interactive',
                       '--constraint=gpu&hbm80g','--nodes=1','--ntasks=1',
                       '--cpus-per-task=32','--gpus=1','--time=04:00:00',
                       '--job-name=mots-bisect-%02d'%iteration,
                       'bash',str(root/'run_cycle.sh'),str(case)]
            state.update(status='ALLOCATING', active=dict(iteration=iteration,
                         amplitude=str(amplitude), directory=str(case)))
            save(root/'state.json', state)
            save(case/'allocation-command.json', command)
            with (case/'allocation.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if (root/'STOP').exists():
                        process.send_signal(2)
                        process.wait()
                        raise RuntimeError('STOP requested; allocation interrupted')
                    time.sleep(10)
                if process.returncode:
                    raise RuntimeError('Allocation/evolution failed: '+str(process.returncode))
            if (case/'run-status').read_text().strip() != '0':
                raise RuntimeError('Evolution failed')
            result = classify(case, bound, args.outer05)
            result.update(amplitude=str(amplitude), directory=str(case),
                          job_id=(case/'job-id.txt').read_text().strip(),
                          input_sha256=sha(case/'input.athinput'),
                          coefficients_sha256=sha(case/'initial.coefficients'))
            if result['classification'] == 'candidate_detected':
                sup = amplitude
            else:
                sub = amplitude
            save(case/'result.json', result)
            state['completed'].append(result)
            state.update(status='CLASSIFIED', active=None, sub=str(sub), super=str(sup))
            save(root/'state.json', state)
        raise RuntimeError('Bisection iteration safety limit')
    except BaseException as error:
        state.update(status='FAILED', error=str(error), failed=time.time())
        save(root/'state.json', state)
        raise


if __name__ == '__main__':
    main()
