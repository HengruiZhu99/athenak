#!/usr/bin/env python3
"""One documented diagnostic retry after the completed first gamma5 run."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import advance


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    root=advance.STATE_ROOT
    with (root/'advance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'state.json').read_text())
        if state.get('exterior_retry_prepared'):
            print('Diagnostic recovery already prepared; use advance.py, do not reset it.');return
        assert state['status']=='needs_review'
        attempts=[a for a in state['attempts'] if a['case']=='g5_tel_b32']
        assert len(attempts)==1
        old=attempts[0]
        assert old['job_id'].split('.')[0]=='8890549' and old['status']=='needs_review'
        job=next(iter(advance.scheduler(old['job_id']).values()))
        assert job['job_state']=='F' and job['Exit_status']==0
        assert not any(j.get('Job_Name','').startswith('tlp-') and j['job_state'] not in ('F','X')
                       for j in advance.scheduler().values())
        audit=old['audit']
        assert all(audit[k] for k in ('complete','finite','horizons_ok','horizon_excised','input_matches'))
        assert not audit['fixed_safe_shell']
        run=root/'runs/g5_tel_b32/8890549'
        rows=advance.numeric_rows(next(run.glob('*.hst')))
        assert all(len(r)==19 and r[15]==0 and r[18]==0 and r[17]==rows[0][17] for r in rows)
        assert any(r[16]>0 for r in rows)
        build=advance.EXE.parent.parent
        marker=build/'executable.sha256'
        if not marker.exists(): print('Corrected GPU build still pending; keep needs_review.');return
        source=(build/'source_commit.txt').read_text().strip()
        assert source.startswith('b29589dd'),source
        digest=hashlib.sha256(advance.EXE.read_bytes()).hexdigest()
        assert digest==marker.read_text().split()[0] and digest!=old['executable_sha256']
        spec=json.loads((advance.WORKFLOW/'cases.json').read_text())
        case=spec['cases'][0]
        assert hashlib.sha256((run/'input.athinput').read_bytes()).hexdigest()==case['input_sha256']
        reason=dict(time=advance.now(),old_job=old['job_id'],source_commit=source,
                    executable_sha256=digest,input_sha256=case['input_sha256'],
                    reason='Full-grid audit sampled excised puncture interior; repeat unchanged evolution once to certify the 3D unexcised exterior. Retain global flag and original evidence.',
                    maximum_case_attempts=2)
        print(json.dumps(reason,indent=2))
        if not args.apply: return
        backup=root/'state_before_exterior_retry.json';assert not backup.exists()
        backup.write_text(json.dumps(state,indent=2)+'\n')
        old['status']='diagnostic_superseded'
        state['exterior_retry_prepared']=reason
        state['status']='prepared'
        advance.save(state)

if __name__=='__main__': main()
