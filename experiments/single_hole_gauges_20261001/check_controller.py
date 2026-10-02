#!/usr/bin/env python3
"""Check submission guards without contacting PBS or launching jobs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import advance


def test(state, jobs, expected_calls, submit=True, returncode=0):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        binary=root/'build/src/athena';binary.parent.mkdir(parents=True)
        binary.write_bytes(b'test')
        (binary.parent.parent/'executable.sha256').write_text(hashlib.sha256(b'test').hexdigest())
        if state is not None: (root/'state.json').write_text(json.dumps(state))
        calls=[]
        def fake(cmd):
            calls.append(cmd)
            class Result: pass
            r=Result();r.returncode=returncode
            r.stdout='999.aurora-pbs-test\n' if not returncode else ''
            r.stderr=('qsub: maximum number of jobs' if returncode==38 else 'connection lost') if returncode else ''
            return r
        with patch.multiple(advance,CAMPAIGN=root,STATE_ROOT=root,EXE=binary), \
             patch.object(advance,'scheduler',return_value=jobs), \
             patch.object(advance,'command',side_effect=fake), \
             patch('sys.argv',['advance.py']+(['--submit'] if submit else [])), \
             contextlib.redirect_stdout(io.StringIO()):
            advance.main()
        assert len(calls)==expected_calls,calls
        out=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else None
        return out

base=dict(attempts=[],completed=[],status='prepared')
assert test(base,{},1)['attempts'][0]['job_id']=='999.aurora-pbs-test'
test(base,{'1':dict(job_state='Q',Job_Name='tlp-00-0',queue='debug')},0)
test(base,{'1':dict(job_state='R',Job_Name='unrelated',queue='debug')},0)
test(dict(base,status='submission_uncertain'),{},0)
test(dict(base,status='needs_review'),{},0)
test(base,{},0,submit=False)
assert test(base,{},1,returncode=38)['status']=='prepared'
assert test(base,{},1,returncode=1)['status']=='submission_uncertain'
print('8 controller checks passed: duplicates, queue occupancy, ambiguous submission, failures, dry run, PBS rejection.')
