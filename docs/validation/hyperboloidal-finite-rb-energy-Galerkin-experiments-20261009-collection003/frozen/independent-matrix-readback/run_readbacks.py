"""Execute only independent algebraic readback of explicitly pinned outputs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT / 'build-layer-research/boundary/total-j-finite-rb-control-20261009'
V = HERE / 'verify_operator.py'
EXPECTED_VERIFIER = '855245e3e4429dc713d817285ec79f05dda112ec159b9b821c9983e95987d95b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(V) == EXPECTED_VERIFIER
cases = (
    ('failed-global-Q64', 'J0-N8-rb.98-Q64-a12x24-readback002',
     '26ddb5877c20d4ee5277427989ef9aaf1a9c0ca416a33cf756e53d96f5ae1d6f',
     'fae33d587b03d382f68dcc940f4f2df17d0840d6537681d9ec45f8842875dcc3'),
    ('segmented-Q64-sector', 'J0-segmented-Q64-sector-readback001',
     '8b4b9a5b33151d86359aa9e35f0ae44dc436ef4ab2d6c17796e77983f9422a27',
     '8449fc9b52f0137498e35b0f5e9b8e35b73998eebe37726bf1eaf4cdd2649d04'),
)
results = []
for label, folder, npz_sha, metadata_sha in cases:
    dest = HERE / label
    assert not dest.exists()
    dest.mkdir()
    command = [sys.executable, str(V), '--npz', str(P/folder/'operator.npz'),
               '--expect-sha256', npz_sha,
               '--metadata', str(P/folder/'matrix-metadata.json'),
               '--expect-metadata-sha256', metadata_sha,
               '--output', str(dest/'result.json')]
    (dest/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    start = time.monotonic()
    run = subprocess.run(command, capture_output=True)
    seconds = time.monotonic()-start
    (dest/'stdout').write_bytes(run.stdout)
    (dest/'stderr').write_bytes(run.stderr)
    result = json.loads((dest/'result.json').read_text())
    receipt = {'command': command, 'returncode': run.returncode,
               'seconds': seconds, 'status': result['status'],
               'source_sha256': sha(V), 'runner_sha256': sha(Path(__file__)),
               'npz_sha256': npz_sha, 'metadata_sha256': metadata_sha,
               'result_sha256': sha(dest/'result.json'),
               'stdout_sha256': sha(dest/'stdout'),
               'stderr_bytes': len(run.stderr),
               'scientific_kernel_or_assembler_run': False,
               'generator_spectrum_or_propagation': False}
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    results.append(receipt)
print(json.dumps(results, indent=2))
