"""Standard-library metadata-only capture; no scientific imports or calls."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'build-layer-research/continuum'


def pin(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


sources = [
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/PLAN.md',
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/KIRCHHOFF-PENCIL.md',
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/VALUES-PLAN.md',
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/flat_ivp_values.py',
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/values-recipe.json',
    BASE / 'native-angular-pulse-flat-IVP-held-20261009/run_values_once001.py',
    BASE / 'physical-RWM-native-pulse-IVP-pencil-20261009/DERIVATION.md',
    BASE / 'physical-RWM-native-pulse-IVP-pencil-20261009/index.json',
]
before = [pin(p) for p in sources]
assert before[3]['sha256'] == (
    '89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7')
write('source-pins.json', {'scope': 'source-only metadata pins', 'files': before})
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                               text=True).strip()
after = [pin(p) for p in sources]
assert after == before
write('receipt.json', {
    'completed': True, 'status': 'PENCIL_SOURCE_ONLY_FORMULAS_CONSISTENT',
    'head': head, 'inputs_unchanged': True,
    'source_files': len(sources),
    'scientific_imports_or_execution': False,
    'actual_values_outputs_read': False,
    'kernel_or_native_queries': False,
    'new_execution_admitted': False,
    'command': [sys.executable, str(Path(__file__).resolve())],
    'derivation': pin(OUT / 'DERIVATION.md'),
    'limitations': ['No derivative implementation or numerical accuracy gate',
                    'No inverse/native-time/caustic or stability claim'],
})
files = [pin(p) for p in sorted(OUT.iterdir()) if p.is_file()]
write('index.json', {'scope': 'immutable pencil/source-only note',
                     'files': files, 'file_count': len(files),
                     'bytes': sum(p['size'] for p in files),
                     'no_future_mutation': True})
print(json.dumps({'index': pin(OUT / 'index.json'),
                  'receipt': pin(OUT / 'receipt.json'),
                  'derivation': pin(OUT / 'DERIVATION.md')}))
