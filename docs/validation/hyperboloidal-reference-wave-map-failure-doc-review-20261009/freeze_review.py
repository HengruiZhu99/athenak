#!/usr/bin/env python3
"""One-shot compact source/receipt index; no scientific input is opened."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
destination = HERE / 'index.json'
assert not destination.exists()
files = {}
for path in sorted(HERE.rglob('*')):
    if path.is_file():
        data = path.read_bytes()
        assert len(data) <= 1048576
        data.decode('utf-8')
        files[str(path.relative_to(HERE))] = {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
index = {'scope': 'Independent source/scalar/copy-catalog review only; one original document rounding correction remains required',
         'files': files, 'file_count': len(files),
         'bytes': sum(x['bytes'] for x in files.values()),
         'raw_RST_BIN_reads': 0, 'scientific_calls': 0}
destination.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'index_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                  'file_count': len(files), 'bytes': index['bytes']}))
