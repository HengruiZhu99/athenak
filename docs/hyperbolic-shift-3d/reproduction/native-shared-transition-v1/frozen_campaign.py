"""Load an explicitly selected, provenance-checked analysis campaign helper."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load(path):
    path = Path(path).resolve()
    provenance = json.loads(path.with_name('provenance.json').read_text())
    for name in (path.name, 'control.py'):
        item = path.with_name(name)
        identity = provenance['files'][name]
        if item.stat().st_size != identity['bytes'] or hashlib.sha256(item.read_bytes()).hexdigest() != identity['sha256']:
            raise ValueError('Frozen helper provenance mismatch: ' + str(item))
    def module(file, name):
        spec = importlib.util.spec_from_file_location(name, file)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    frozen_control = module(path.with_name('control.py'), 'frozen_analysis_control')
    previous = sys.modules.get('control')
    try:
        sys.modules['control'] = frozen_control
        return module(path, 'frozen_analysis_campaign')
    finally:
        if previous is None:
            sys.modules.pop('control', None)
        else:
            sys.modules['control'] = previous
