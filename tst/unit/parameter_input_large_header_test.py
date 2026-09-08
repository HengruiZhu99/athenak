#!/usr/bin/env python3
"""Exercise large parameter headers and markers split across I/O buffers."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--athena', required=True, type=Path)
args = p.parse_args()
with tempfile.TemporaryDirectory() as directory:
    for offset, marker in [(4094, True), (65534, True), (110495, True), (110495, False)]:
        first = b'<job>\nanswer = 42\n#'
        last = b'\nlast = tail\n'
        content = first + b'x'*(offset-len(first)-len(last)) + last
        if marker:
            content += b'<par_end>\n\x00not a parameter: binary payload\n'
        path = Path(directory)/'header.athinput'
        path.write_bytes(content)
        result = subprocess.run([str(args.athena.resolve()), '-i', str(path), '-n'],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or not all(re.search(pattern, result.stdout) for pattern in
                                        [r'answer\s*=\s*42', r'last\s*=\s*tail']):
            raise RuntimeError(f'Header regression failed at {offset}:\n'
                               +result.stdout+result.stderr)
print('Large headers, split markers, and plain-input EOF passed')
