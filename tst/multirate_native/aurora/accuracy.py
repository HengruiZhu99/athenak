"""Run the native endpoint accuracy and restart gates on bound Aurora tiles."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from binding import launcher


def qualify(root, executable, runs):
    suite = Path(__file__).resolve().parents[1]
    python = root.parent/'n12-full-scri-20261115/analysis-env/bin/python'
    with executable.open('rb') as stream:
        binary_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    receipts = []
    for check,tiles in runs:
        if check not in ('order','restart') or tiles not in (2,24):
            raise ValueError('invalid GPU accuracy check or tile count')
        parent = root/'runs'/f'accuracy-{binary_sha[:12]}-{check}-{tiles}tiles'
        parent.mkdir(parents=True,exist_ok=True)
        receipt = parent/'receipt.json'
        if not receipt.exists():
            attempt = 1
            while (parent/f'run-{attempt}').exists(): attempt+=1
            directory = parent/f'run-{attempt}'
            launch = launcher(tiles)
            script = 'temporal_convergence.py' if check=='order' else 'adaptive.py'
            command = [str(python),str(suite/script),'--binary',str(executable),
                       '--ranks',str(tiles),'--output',str(directory),'--launcher',shlex.join(launch)]
            if check=='order':
                command+=['--adaptive','--radial-probe','--cached-stage-packets']
            with (parent/f'run-{attempt}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                data=json.loads((directory/'results.json').read_text())
                if check=='order':
                    subprocess.run([str(python),str(suite/'endpoint_diagnostics.py'),str(directory),
                                    '--scri-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
                    data['scri']=json.loads((directory/'diagnostics.json').read_text())
                    if not data['passed'] or any(value is None or not value>=3.7
                            for orders in data['scri']['orders'].values() for value in orders):
                        raise ValueError('GPU endpoint temporal order gate failed in '+str(directory))
            encoded=dict(binary_sha256=binary_sha,check=check,tiles=tiles,physical_gpus=tiles//2,
                         directory=str(directory),command=command,result=data)
            temporary=receipt.with_suffix('.tmp')
            temporary.write_text(json.dumps(encoded,indent=2,allow_nan=False)+'\n')
            temporary.replace(receipt)
        saved=json.loads(receipt.read_text())
        if (saved['binary_sha256'],saved['check'],saved['tiles'])!=(binary_sha,check,tiles):
            raise ValueError('GPU accuracy receipt provenance mismatch')
        receipts.append(saved)
    return receipts
