"""Run the new independent audits, preserving exact commands and exit status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('output', type=Path)
ap.add_argument('--production', type=Path,
                help='Existing zero-step principal/source CSV directory to cross-check')
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
commands = [[sys.executable, str(HERE/name)] for name in
            ['symbol.py', 'subsidiary.py', 'hyperbolicity.py', 'projector_limits.py', 'wormhole_subsidiary.py']]
if args.production:
    commands += [[sys.executable, str(HERE/name), str(args.production.resolve())]
                 for name in ['verify_production_symbol.py', 'verify_fourier.py']]
records = []
for argv in commands:
    log = args.output/(Path(argv[1]).stem+'.log')
    with log.open('w') as file:
        result = subprocess.run(argv, stdout=file, stderr=subprocess.STDOUT)
    records.append(dict(argv=argv, exit_code=result.returncode, log=str(log)))
    (args.output/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    print(Path(argv[1]).name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
