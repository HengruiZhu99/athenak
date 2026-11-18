"""Continue the isolated two-node comparison from verified native segment receipts."""
import fcntl
import json
import os
from pathlib import Path
from n12_paths import trajectory_directory
import re
import subprocess
import sys

from n12 import save


def next_mode(root, current, job):
    states = {}
    for mode in ('uniform', 'multirate'):
        path = trajectory_directory(root,mode)/'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            receipt = Path(state['checkpoint']).parent.parent/'receipt.json'
            if (state != json.loads(receipt.read_text()) or state['mode'] != mode
                    or not 0 < state['cycle'] <= state['target_cycles']
                    or state['target_reached'] != (state['cycle'] == state['target_cycles'])):
                raise ValueError(f'inconsistent {mode} segment receipt')
            states[mode] = state
    if current not in states or states[current]['job'] != job.split('.')[0]:
        raise ValueError('this allocation did not commit a new N12 segment')
    unfinished=[m for m in ('uniform','multirate') if m not in states or not states[m]['target_reached']]
    if not unfinished:
        return None
    # Both methods target the same physical endpoint, with different cycle
    # counts. Advance the lagging trajectory to provide common science cuts.
    return min(unfinished,key=lambda m: states[m]['cycle']/states[m]['target_cycles']
               if m in states else 0.0)


def continue_comparison(root, current, job):
    base = root/'n12-10M'
    with (base/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mode = next_mode(root, current, job)
        if mode is None:
            print('Both N12 trajectories reached their requested endpoint.', flush=True)
            return
        marker = base/f'successor-{job.split(".")[0]}.json'
        if marker.exists():
            if 'job' not in json.loads(marker.read_text()):
                raise RuntimeError(f'check PBS for the in-flight submission recorded in {marker}')
            print(marker.read_text(), flush=True)
            return
        variables = [f'MR_N12_MODE={mode}', 'MR_N12_CHAIN=1']
        if all((trajectory_directory(root,m)/'state.json').is_file() for m in (mode,'multirate')):
            variables.append('MR_N12_RESUME_PINNED=1')
        pending = os.environ.get('MR_N12_NEXT_CASES')
        if pending:
            # Spread longer strong-scaling suites over the existing one-hour
            # science allocations. A later suite reuses matching hashed receipts.
            cases = pending.split(':')
            if len(cases)==1 and Path(pending).name==pending:
                description=json.loads((root/pending).read_text())
                if isinstance(description,dict) and 'suites' in description:
                    cases=description['suites']
                    if not isinstance(cases,list) or not cases or not all(isinstance(s,str) for s in cases):
                        raise ValueError('invalid ordered benchmark suite list')
            for name in cases:
                if Path(name).name!=name or not (root/name).is_file():
                    raise ValueError('missing benchmark case file '+name)
            variables.append('MR_CASES='+cases[0])
            if len(cases)>1:
                variables.append('MR_N12_NEXT_CASES='+':'.join(cases[1:]))
        command = ['qsub', '-N', 'mr-n12-'+mode, '-W', 'depend=afterok:'+job,
                   '-v', ','.join(variables), str(Path(__file__).with_name('debug.pbs'))]
        # Preserve a submission in flight across an interruption between qsub and
        # its receipt. The scheduler can then be reconciled without a second job.
        save(marker,dict(parent=job,mode=mode,command=command))
        try:
            result = subprocess.run(command,cwd=root,text=True,capture_output=True,check=True)
        except subprocess.CalledProcessError as exc:
            marker.unlink()
            raise RuntimeError('PBS submission failed: '+exc.stderr) from exc
        child = result.stdout.strip()
        if not re.fullmatch(r'\d+(?:\.[\w.-]+)?',child):
            raise ValueError('unexpected PBS job identifier '+child)
        save(marker,dict(parent=job,mode=mode,job=child,command=command))
        print(marker.read_text(), flush=True)


if __name__ == '__main__':
    continue_comparison(Path(sys.argv[1]).resolve(),sys.argv[2],os.environ['PBS_JOBID'])
