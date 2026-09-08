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


def classify(case):
    accepted = []
    for path in case.glob('*.mots_candidates.csv'):
        accepted += [r for r in csv.DictReader(path.open()) if r['policy_accepted'] == '1']
    if accepted:
        if not all(0 <= float(r['time']) <= 50.0000001 and
                   math.isfinite(float(r['epsilon2'])) and float(r['epsilon2']) <= .01
                   for r in accepted):
            raise RuntimeError('Invalid accepted-candidate evidence')
        termination = json.loads(next(case.glob('*.termination.json')).read_text())
        if termination['outcome'] not in ('mots_candidate', 'collapse'):
            raise RuntimeError('Candidate/termination disagreement')
        return dict(classification='candidate_detected', accepted=accepted, termination=termination)
    if list(case.glob('*.termination.json')):
        raise RuntimeError('Unexpected stopping condition without candidate')
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
    if not final or abs(final['time']-50) > 1e-8:
        raise RuntimeError('Nondetection run did not reach t=50')
    if 'Terminating on time limit' not in (case/'stdout.log').read_text():
        raise RuntimeError('Unexpected evolution termination')
    if not list(case.glob('*.mots_candidates.csv')):
        raise RuntimeError('Finder produced no evidence')
    return dict(classification='no_candidate_through_t50', final=final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['survey', 'baseline', 'athena', 'output']:
        p.add_argument('--'+key, required=True, type=Path)
    args = p.parse_args()
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
                 criterion='angular_candidate by t=50', spatially_validated=False,
                 sub=str(sub), super=str(sup), relative_tolerance='0.000001',
                 survey_sha256=sha(args.survey), executable_sha256=sha(args.athena),
                 baseline_template_sha256=sha(args.baseline/'template.athinput'), completed=[])
    save(root/'state.json', state)
    # Preserve qualified initial-data generation and binding logic byte-for-byte.
    oldrunner = (args.baseline/'run_cycle.sh').read_text()
    runner = oldrunner.replace('campaign=$(dirname -- "$case_dir")',
                              'campaign='+str(args.baseline.resolve()))
    runner = runner.replace('"$campaign/athena.history_extrema"', '"'+str(args.athena.resolve())+'"')
    (root/'run_cycle.sh').write_text(runner)
    finder = dict(lmax=128, ntheta=260, mots_l_start=8, flow_iterations_0=500,
                  mots_radius_count=4, mots_detection='angular_candidate',
                  mots_level_iterations=96, mots_candidate_bound=.01,
                  mots_promotion_max=.5, mots_angular_ratio=.8, mots_candidate_points=1061,
                  mots_shape_change=.02, mots_area_change=.01, mots_epsilon2=1e-6,
                  mots_epsilon_inf=1e-5, mots_write_profiles='true', mots_profile_points=1061)
    try:
        for iteration in range(1, 21):
            width = abs(sub-sup)/abs(sup)
            state['relative_width'] = str(width)
            if width <= Decimal('0.000001'):
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
            result = classify(case)
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
