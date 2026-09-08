#!/usr/bin/env python3
"""Archive completed bisection diagnostics and checkpoint locations, not field dumps."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    state = json.loads((args.campaign/'state.json').read_text())
    inventory = []
    with tarfile.open(args.output, 'x:gz') as archive:
        def data(name, value):
            content = (json.dumps(value, indent=2)+'\n').encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        data('state.json', state)
        archive.add(args.campaign/'run_cycle.sh', arcname='run_cycle.sh')
        for record in state['completed']:
            case = Path(record['directory'])
            for path in sorted(case.iterdir()):
                if path.is_file() and '.mots_surface_' not in path.name and '.mots_dense_' not in path.name:
                    archive.add(path, arcname=str(Path(case.name)/path.name))
            if (case/'final-mots').exists():
                archive.add(case/'final-mots', arcname=case.name+'/final-mots')
            else:
                for row in record.get('accepted', []):
                    for kind, extension in [('surface', 'csv'), ('dense', 'json')]:
                        path = case/('mots_bisect.mots_'+kind+'_'+row['cycle']+'_'+row['id']+'.'+extension)
                        archive.add(path, arcname=str(Path(case.name)/path.name))
            checkpoints = sorted(case.glob('rst/*.rst'))
            if not checkpoints:
                raise RuntimeError('Completed run lacks checkpoint: '+str(case))
            inventory.append(dict(amplitude=record['amplitude'], directory=str(case),
                checkpoints=[dict(path=str(f), bytes=f.stat().st_size) for f in checkpoints],
                latest_sha256=sha(checkpoints[-1])))
        data('checkpoint_inventory.json', inventory)


if __name__ == '__main__':
    main()
