"""Resume an allocation-limited screen only; never restart a scientific failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run',type=Path,required=True);ap.add_argument('--build',type=Path,required=True)
a=ap.parse_args()
if (a.run/'completed.json').exists():
    print('Screen already reached its requested endpoint; no resume.')
    raise SystemExit(0)
logs=sorted(a.run.glob('segment-*.log'))
for path in logs:
    if 'FATAL ERROR' in path.read_text(errors='replace'):
        print('Strict scientific failure preserved; no resume:',path)
        raise SystemExit(0)
if not logs or not list(a.run.glob('rst/*.rst')):
    raise SystemExit('No resumable evidence; inspect setup failure')
cmd=[sys.executable,str(Path(__file__).with_name('cuda_driver.py')),'run','--resume',
     '--build',str(a.build),'--input',str(a.run/'used_input.athinput'),'--output',str(a.run)]
status=subprocess.run(cmd).returncode
(a.run/'allocation-resume-status.json').write_text(json.dumps(dict(command=cmd,exit_code=status),indent=2)+'\n')
raise SystemExit(status)
