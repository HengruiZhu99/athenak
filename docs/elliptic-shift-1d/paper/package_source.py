#!/usr/bin/env python3
"""Package the portable paper, excluding TeX build debris and QA renders."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    paths = {p for p in root.iterdir() if p.is_file() and
             p.suffix in {'.tex', '.py', '.md', '.pdf', '.json'} and
             p.name != 'revision-manifest.json'}
    for name in ['sections', 'data', 'figures']:
        paths.update(p for p in (root/name).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths = sorted(paths)
    manifest = [dict(path=str(p.relative_to(root)),
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    manifest_path = root/'revision-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    target = root.parent/'elliptic-shift-prd-source.zip'
    temporary = target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths+[manifest_path]:
            archive.write(path, 'elliptic-shift-prd/'+str(path.relative_to(root)))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for record in manifest:
            content = archive.read('elliptic-shift-prd/'+record['path'])
            assert hashlib.sha256(content).hexdigest() == record['sha256']
        assert len(archive.namelist()) == len(manifest)+1
    temporary.replace(target)
    print(json.dumps(dict(archive=str(target), manifest_files=len(manifest),
                          crc_and_all_hashes_verified=True), indent=2))


if __name__ == '__main__':
    main()
